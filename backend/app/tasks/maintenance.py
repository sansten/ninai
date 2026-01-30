"""Maintenance / scheduled tasks.

Includes Logseq nightly export trigger (enqueue per-memory export tasks).
"""

from __future__ import annotations

import asyncio
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from celery.utils.log import get_task_logger
from sqlalchemy import and_, or_, select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import async_session_factory, get_tenant_session
from app.models.memory import MemoryMetadata
from app.models.memory_logseq_export import MemoryLogseqExport
from app.models.organization import Organization
from app.models.export_job import ExportJob
from app.services.org_logseq_export_config_service import OrgLogseqExportConfigService
from app.services.export_job_service import ExportJobService
from app.services.audit_service import AuditService


logger = get_task_logger(__name__)


def _run_async(coro):
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


def _broker_enabled() -> bool:
    broker = celery_app.conf.broker_url
    return bool(broker) and not str(broker).startswith("memory://")


def _enqueue_logseq_export(*, org_id: str, memory_id: str, initiator_user_id: str | None, trace_id: str | None):
    if not _broker_enabled():
        return None

    return celery_app.send_task(
        "app.tasks.memory_pipeline.logseq_export_task",
        kwargs={
            "org_id": org_id,
            "memory_id": memory_id,
            "initiator_user_id": initiator_user_id,
            "trace_id": trace_id,
            "storage": "long_term",
        },
    )


async def _nightly_logseq_export_async(*, batch_size: int, lookback_hours: int) -> dict:
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        org_ids = (
            await session.execute(select(Organization.id).where(Organization.is_active.is_(True)))
        ).scalars().all()

    enqueued = 0
    orgs_processed = 0

    service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    for org_id in org_ids:
        orgs_processed += 1

        async with get_tenant_session(
            user_id=service_user_id,
            org_id=str(org_id),
            roles=service_roles,
            clearance_level=0,
            justification="nightly_logseq_export",
        ) as tenant_session:
            cfg_svc = OrgLogseqExportConfigService(tenant_session)
            cfg = await cfg_svc.get_config(organization_id=str(org_id))

            since = getattr(cfg, "last_nightly_export_at", None)
            if since is None:
                since = now - timedelta(hours=max(1, int(lookback_hours or 24)))

            # Enqueue exports for memories changed since the cursor.
            stmt = (
                select(MemoryMetadata.id)
                .outerjoin(
                    MemoryLogseqExport,
                    and_(
                        MemoryLogseqExport.organization_id == MemoryMetadata.organization_id,
                        MemoryLogseqExport.memory_id == MemoryMetadata.id,
                    ),
                )
                .where(
                    MemoryMetadata.organization_id == str(org_id),
                    MemoryMetadata.memory_type != "short_term",
                    MemoryMetadata.updated_at >= since,
                    or_(
                        MemoryLogseqExport.id.is_(None),
                        MemoryLogseqExport.updated_at < MemoryMetadata.updated_at,
                    ),
                )
                .order_by(MemoryMetadata.updated_at.desc())
                .limit(int(batch_size))
            )

            ids = (await tenant_session.execute(stmt)).scalars().all()

            for memory_id in ids:
                _enqueue_logseq_export(
                    org_id=str(org_id),
                    memory_id=str(memory_id),
                    initiator_user_id=service_user_id or None,
                    trace_id=f"nightly:{now.date().isoformat()}",
                )
                enqueued += 1

            await cfg_svc.update_last_nightly_export_at(organization_id=str(org_id), last_nightly_export_at=now)

    return {
        "ok": True,
        "orgs_processed": orgs_processed,
        "enqueued": enqueued,
        "batch_size": int(batch_size),
        "lookback_hours": int(lookback_hours),
        "ran_at": now.isoformat(),
    }


async def _cleanup_expired_snapshot_exports_async(*, batch_size: int) -> dict:
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        org_ids = (
            await session.execute(select(Organization.id).where(Organization.is_active.is_(True)))
        ).scalars().all()

    deleted_files = 0
    jobs_updated = 0
    orgs_processed = 0

    service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    for org_id in org_ids:
        orgs_processed += 1

        async with get_tenant_session(
            user_id=service_user_id,
            org_id=str(org_id),
            roles=service_roles,
            clearance_level=0,
            justification="cleanup_expired_snapshot_exports",
        ) as tenant_session:
            stmt = (
                select(ExportJob)
                .where(
                    ExportJob.organization_id == str(org_id),
                    ExportJob.job_type == "snapshot",
                    ExportJob.expires_at.is_not(None),
                    ExportJob.expires_at < now,
                    ExportJob.file_path.is_not(None),
                )
                .order_by(ExportJob.expires_at.asc())
                .limit(int(batch_size))
            )
            jobs = (await tenant_session.execute(stmt)).scalars().all()

            for job in jobs:
                file_path = getattr(job, "file_path", None)
                if not file_path:
                    continue

                # Best-effort delete: remove containing job dir if it looks like ours.
                p = Path(str(file_path))
                try:
                    job_dir = p.parent
                    base_dir = ExportJobService.snapshot_base_dir().resolve()
                    job_dir_resolved = job_dir.resolve()

                    if str(job_dir_resolved).startswith(str(base_dir)):
                        shutil.rmtree(job_dir, ignore_errors=True)
                    else:
                        if p.exists() and p.is_file():
                            p.unlink(missing_ok=True)
                except Exception:
                    logger.exception("Failed deleting export artifact", extra={"job_id": getattr(job, "id", None)})

                # Prevent repeated cleanup attempts.
                job.file_path = None
                job.file_bytes = None
                job.file_sha256 = None
                jobs_updated += 1
                deleted_files += 1

                try:
                    await AuditService(tenant_session).log_event(
                        event_type="export.snapshot_artifact_deleted",
                        actor_id=service_user_id or None,
                        actor_type="system",
                        organization_id=str(org_id),
                        resource_type="export_job",
                        resource_id=str(getattr(job, "id", "")),
                        success=True,
                        details={"expired_at": getattr(job, "expires_at", None).isoformat() if getattr(job, "expires_at", None) else None},
                    )
                except Exception:
                    logger.exception("Failed to write audit for export cleanup")

            await tenant_session.commit()

    return {
        "ok": True,
        "orgs_processed": orgs_processed,
        "jobs_updated": jobs_updated,
        "deleted_files": deleted_files,
        "batch_size": int(batch_size),
        "ran_at": now.isoformat(),
    }


@celery_app.task(bind=True)
def nightly_logseq_export_task(self, batch_size: int = 500, lookback_hours: int = 24):
    """Nightly batch export for changed memories.

    Enqueues per-memory Logseq export tasks (logseq_export_task).

    Note: in unit tests Celery defaults to memory:// broker; in that case this is a no-op.
    """

    if not _broker_enabled():
        return {
            "ok": True,
            "skipped": True,
            "reason": "broker_disabled",
        }

    try:
        return _run_async(_nightly_logseq_export_async(batch_size=batch_size, lookback_hours=lookback_hours))
    except Exception as e:
        logger.exception("Nightly logseq export task failed")
        raise e


@celery_app.task(bind=True)
def cleanup_expired_snapshot_exports_task(self, batch_size: int = 200):
    """Delete expired snapshot export artifacts from disk and clear file pointers."""

    if not _broker_enabled():
        return {
            "ok": True,
            "skipped": True,
            "reason": "broker_disabled",
        }

    try:
        return _run_async(_cleanup_expired_snapshot_exports_async(batch_size=batch_size))
    except Exception as e:
        logger.exception("Cleanup expired snapshot exports task failed")
        raise e
