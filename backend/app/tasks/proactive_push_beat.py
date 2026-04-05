"""Proactive push beat task (Feature 24.10).

Runs every 15 minutes and executes proactive intelligence push cycles
for all active organizations.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import async_session_factory, get_tenant_session
from app.models.organization import Organization
from app.services.proactive_push_service import ProactivePushService


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


async def _run_proactive_push_cycle(
    *,
    session_lookback_minutes: int = 180,
    event_lookback_minutes: int = 30,
    max_pushes_per_org: int = 25,
) -> dict[str, Any]:
    async with async_session_factory() as session:
        org_rows = (
            await session.execute(
                select(Organization.id, Organization.settings).where(Organization.is_active.is_(True))
            )
        ).all()

    service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", "") or "")
    service_roles = "system_admin" if service_user_id else ""

    orgs_processed = 0
    total_pushed = 0
    total_candidates = 0
    per_org: list[dict[str, Any]] = []

    for org_id, org_settings in org_rows:
        orgs_processed += 1

        async with get_tenant_session(
            user_id=service_user_id,
            org_id=str(org_id),
            roles=service_roles,
            clearance_level=0,
            justification="proactive_push_beat",
        ) as tenant_db:
            svc = ProactivePushService(tenant_db)
            threshold = svc.get_push_threshold(org_settings or {})
            outcome = await svc.run_cycle(
                organization_id=str(org_id),
                push_threshold=threshold,
                session_lookback_minutes=session_lookback_minutes,
                event_lookback_minutes=event_lookback_minutes,
                max_pushes=max_pushes_per_org,
            )
            per_org.append(outcome)
            total_pushed += int(outcome.get("pushed") or 0)
            total_candidates += int(outcome.get("candidates") or 0)

    return {
        "ok": True,
        "orgs_processed": orgs_processed,
        "candidates": total_candidates,
        "pushed": total_pushed,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "per_org": per_org,
    }


@celery_app.task(name="app.tasks.proactive_push_beat.proactive_push_beat_task")
def proactive_push_beat_task(
    *,
    session_lookback_minutes: int = 180,
    event_lookback_minutes: int = 30,
    max_pushes_per_org: int = 25,
) -> dict[str, Any]:
    """Scheduled proactive intelligence push cycle across all active orgs."""
    return _run_async(
        _run_proactive_push_cycle(
            session_lookback_minutes=session_lookback_minutes,
            event_lookback_minutes=event_lookback_minutes,
            max_pushes_per_org=max_pushes_per_org,
        )
    )
