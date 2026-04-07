"""Cognitive Schedule Runner — Phase 84.

Celery beat task that fires due CognitiveSchedule records every minute.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app

logger = get_task_logger(__name__)


@celery_app.task(name="cognitive_schedule_runner", bind=True, max_retries=3)
def run_cognitive_schedule_task(
    self,
    *,
    schedule_id: str,
    org_id: str,
) -> dict:
    """Execute a single cognitive schedule by invoking the cognitive gateway.

    Called either by the beat poller (fire_due_schedules) or via the
    manual /run endpoint.
    """
    try:
        result = asyncio.run(_exec_schedule(schedule_id=schedule_id, org_id=org_id))
        return result
    except Exception as exc:
        logger.error("cognitive schedule %s failed: %s", schedule_id, exc)
        raise self.retry(exc=exc, countdown=60)


@celery_app.task(name="cognitive_schedule_poller")
def fire_due_schedules() -> int:
    """Poll for due schedules and enqueue individual run tasks.

    Designed to run every 1 minute via Celery beat.
    Returns the number of schedules enqueued.
    """
    return asyncio.run(_poll_due_schedules())


async def _poll_due_schedules() -> int:
    from sqlalchemy import select
    from app.core.database import async_session_factory, set_tenant_context
    from app.models.cognitive_schedule import CognitiveSchedule
    from app.services.cognitive_schedule_service import compute_next_run

    now = datetime.now(timezone.utc)
    enqueued = 0

    async with async_session_factory() as session:
        result = await session.execute(
            select(CognitiveSchedule).where(
                CognitiveSchedule.is_active.is_(True),
                CognitiveSchedule.next_run_at <= now,
            )
        )
        schedules = result.scalars().all()
        for sched in schedules:
            run_cognitive_schedule_task.delay(
                schedule_id=sched.id, org_id=sched.organization_id
            )
            sched.last_run_at = now
            sched.next_run_at = compute_next_run(sched.cron_expression, after=now)
            enqueued += 1
        await session.commit()

    return enqueued


async def _exec_schedule(*, schedule_id: str, org_id: str) -> dict:
    """Run the cognitive verb from the schedule."""
    from sqlalchemy import select
    from app.core.database import async_session_factory
    from app.models.cognitive_schedule import CognitiveSchedule
    from app.services.cognitive_gateway_service import (
        CognitiveGatewayCapabilities,
        CognitiveGatewayService,
    )

    async with async_session_factory() as session:
        result = await session.execute(
            select(CognitiveSchedule).where(
                CognitiveSchedule.id == schedule_id,
                CognitiveSchedule.organization_id == org_id,
            )
        )
        sched = result.scalars().first()
        if sched is None:
            return {"status": "not_found", "schedule_id": schedule_id}

        svc = CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())
        verb = sched.cognitive_verb
        payload = dict(sched.payload or {})

        if verb == "plan":
            gate_result = await svc.plan(
                goal=payload.get("goal", "scheduled plan"),
                context=payload,
            )
            outcome = {"steps": gate_result.steps, "confidence": gate_result.confidence}
        elif verb in {"analyze", "summarize", "report", "monitor", "escalate", "acknowledge"}:
            gate_result = await svc.decide(
                content=payload.get("content", f"scheduled {verb}"),
                enrichment=payload,
            )
            outcome = {"decision": gate_result.decision, "confidence": gate_result.confidence}
        else:
            outcome = {"status": "unknown_verb", "verb": verb}

        return {"status": "ok", "schedule_id": schedule_id, "outcome": outcome}
