"""Intelligence Digest Pipeline — P46.

Daily Celery beat task (06:00 UTC) that builds per-organization intelligence
digests and delivers them via the existing webhook infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import async_session_factory, get_tenant_session
from app.models.organization import Organization
from app.services.digest_service import DigestService


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


async def _run_digest_async(digest_type: str = "daily") -> dict[str, Any]:
    now = datetime.now(timezone.utc)

    async with async_session_factory() as session:
        org_ids = (
            await session.execute(
                select(Organization.id).where(Organization.is_active.is_(True))
            )
        ).scalars().all()

    service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    orgs_processed = 0
    orgs_failed = 0

    for org_id in org_ids:
        try:
            async with get_tenant_session(
                user_id=service_user_id,
                org_id=str(org_id),
                roles=service_roles,
                clearance_level=0,
                justification="daily_intelligence_digest",
            ) as tenant_session:
                svc = DigestService(tenant_session)
                report = await svc.build_digest(str(org_id), digest_type)
                await svc.save_report(report)

                # Deliver via webhook (best-effort — don't fail the pipeline if this errors)
                try:
                    from app.services.webhook_service import WebhookService
                    wh = WebhookService(tenant_session)
                    payload = {
                        "report_id": str(report.id),
                        "org_id": str(org_id),
                        "digest_type": digest_type,
                        "run_date": report.run_date.isoformat(),
                        "memories_total": report.memories_total,
                        "anomalies_count": report.anomalies_count,
                        "narrative": report.narrative,
                    }
                    await wh.emit_event(
                        organization_id=str(org_id),
                        event_type="digest.ready",
                        payload=payload,
                    )
                    report.delivered_at = datetime.now(timezone.utc)
                    await tenant_session.commit()
                except Exception:
                    logger.exception(
                        "Digest webhook delivery failed",
                        extra={"org_id": str(org_id)},
                    )

                orgs_processed += 1

        except Exception:
            logger.exception(
                "Digest pipeline failed for org", extra={"org_id": str(org_id)}
            )
            orgs_failed += 1

    return {
        "ok": True,
        "digest_type": digest_type,
        "orgs_processed": orgs_processed,
        "orgs_failed": orgs_failed,
        "ran_at": now.isoformat(),
    }


@celery_app.task(bind=True, name="app.tasks.digest_pipeline.digest_task")
def digest_task(self, digest_type: str = "daily"):
    """Daily intelligence digest — builds per-org digest reports at 06:00 UTC.

    Disabled (no-op) when Celery runs with the in-memory broker used during tests.
    """
    if not _broker_enabled():
        return {
            "ok": True,
            "skipped": True,
            "reason": "broker_disabled",
        }

    try:
        return _run_async(_run_digest_async, digest_type=digest_type)
    except Exception as e:
        logger.exception("Digest pipeline task failed")
        raise e
