"""GDPR Deletion Pipeline.

Celery task that processes UserDataDeletion requests asynchronously.
Triggered by the compliance endpoint after creating a deletion request.

Deletion order (FK-safe):
  1. MemoryAttachment, MemorySemanticNode rows for the user's memories
  2. MemoryMetadata rows (scope=org_data or all)
  3. Anonymize user_id references in AuditEvent (legal hold — no hard delete)
  4. UserRole, ConsentRecord rows
  5. For scope=account or all: anonymize User.email / full_name in place
  6. Mark UserDataDeletion.status = completed
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from celery.utils.log import get_task_logger
from sqlalchemy import delete, select, update

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.models.compliance import UserDataDeletion
from app.services.compliance_service import anonymize_user_id

logger = get_task_logger(__name__)


def _broker_enabled() -> bool:
    broker = celery_app.conf.broker_url
    return bool(broker) and not str(broker).startswith("memory://")


def _run_async(async_fn, *args, **kwargs):
    coro = async_fn(*args, **kwargs)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    return asyncio.run(coro)


async def _process_deletion(request_id: str) -> dict[str, Any]:
    """Core async deletion logic. Returns summary dict."""
    async with async_session_factory() as session:
        # Load the deletion request
        stmt = select(UserDataDeletion).where(UserDataDeletion.id == request_id)
        result = await session.execute(stmt)
        req = result.scalar_one_or_none()

        if req is None:
            logger.warning("Deletion request not found", extra={"request_id": request_id})
            return {"ok": False, "reason": "request_not_found"}

        if req.status not in ("pending", "processing"):
            logger.info(
                "Deletion request already resolved",
                extra={"request_id": request_id, "status": req.status},
            )
            return {"ok": True, "skipped": True, "status": req.status}

        user_id = req.user_id
        org_id = req.organization_id
        scope = req.deletion_scope
        anon_id = anonymize_user_id(user_id)

        try:
            req.status = "processing"
            await session.commit()

            deleted: dict[str, int] = {}

            # ── 1. Remove memory attachments owned by this user ──────────
            try:
                from app.models.memory import MemoryMetadata
                from sqlalchemy import and_

                # Collect memory ids for this user+org
                mem_stmt = select(MemoryMetadata.id).where(
                    and_(
                        MemoryMetadata.organization_id == org_id,
                        MemoryMetadata.created_by == user_id,
                    )
                )
                mem_ids = list((await session.execute(mem_stmt)).scalars().all())

                if mem_ids and scope in ("org_data", "all"):
                    # Delete memories (cascade handles attachments/semantic nodes)
                    del_stmt = delete(MemoryMetadata).where(
                        MemoryMetadata.id.in_([str(m) for m in mem_ids])
                    )
                    del_result = await session.execute(del_stmt)
                    deleted["memories"] = del_result.rowcount
                    logger.info(
                        "Deleted memories",
                        extra={"count": deleted["memories"], "user_id": user_id},
                    )
            except Exception:
                logger.exception("Failed to delete memories", extra={"user_id": user_id})

            # ── 2. Anonymize user_id in AuditEvent (legal hold) ──────────
            try:
                from app.models.audit import AuditEvent
                anon_stmt = (
                    update(AuditEvent)
                    .where(AuditEvent.user_id == user_id)
                    .values(user_id=anon_id)
                )
                anon_result = await session.execute(anon_stmt)
                deleted["audit_anonymized"] = anon_result.rowcount
            except Exception:
                logger.exception("Failed to anonymize audit events", extra={"user_id": user_id})

            # ── 3. Remove ConsentRecords ──────────────────────────────────
            try:
                from app.models.compliance import ConsentRecord
                del_consent = delete(ConsentRecord).where(
                    ConsentRecord.user_id == user_id,
                    ConsentRecord.organization_id == org_id,
                )
                cr = await session.execute(del_consent)
                deleted["consent_records"] = cr.rowcount
            except Exception:
                logger.exception("Failed to delete consent records", extra={"user_id": user_id})

            # ── 4. Remove UserRole entries ────────────────────────────────
            try:
                from app.models.user import UserRole
                del_roles = delete(UserRole).where(
                    UserRole.user_id == user_id,
                    UserRole.organization_id == org_id,
                )
                rr = await session.execute(del_roles)
                deleted["user_roles"] = rr.rowcount
            except Exception:
                logger.exception("Failed to delete user roles", extra={"user_id": user_id})

            # ── 5. Anonymize User record (account/all scope) ──────────────
            if scope in ("account", "all"):
                try:
                    from app.models.user import User
                    anon_email = f"{anon_id}@deleted.invalid"
                    upd = (
                        update(User)
                        .where(User.id == user_id)
                        .values(
                            email=anon_email,
                            full_name="DELETED",
                            is_active=False,
                        )
                    )
                    await session.execute(upd)
                    deleted["user_anonymized"] = 1
                except Exception:
                    logger.exception("Failed to anonymize user record", extra={"user_id": user_id})

            await session.commit()

            # ── 6. Mark request completed ─────────────────────────────────
            req.status = "completed"
            req.deleted_at = datetime.now(timezone.utc)
            await session.commit()

            logger.info(
                "Deletion request completed",
                extra={"request_id": request_id, "deleted": deleted},
            )
            return {"ok": True, "request_id": request_id, "deleted": deleted}

        except Exception as exc:
            logger.exception("Deletion pipeline failed", extra={"request_id": request_id})
            try:
                req.status = "failed"
                req.error_detail = str(exc)[:2000]
                await session.commit()
            except Exception:
                pass
            return {"ok": False, "request_id": request_id, "error": str(exc)}


@celery_app.task(bind=True, name="app.tasks.gdpr_pipeline.process_deletion_task")
def process_deletion_task(self, deletion_request_id: str) -> dict[str, Any]:
    """Process a GDPR user data deletion request asynchronously."""
    if not _broker_enabled():
        return {"ok": True, "skipped": True, "reason": "broker_disabled"}

    logger.info("Starting deletion task", extra={"request_id": deletion_request_id})
    try:
        return _run_async(_process_deletion, deletion_request_id)
    except Exception as exc:
        logger.exception("process_deletion_task raised", extra={"request_id": deletion_request_id})
        raise exc
