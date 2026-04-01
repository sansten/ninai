"""Celery tasks for prospective memory scanning and reminder firing (Phase 53)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import async_session_factory as AsyncSessionLocal
from app.models.prospective_reminder import ProspectiveReminder
from app.services.event_publishing_service import EventPublishingService

logger = logging.getLogger(__name__)

# Beat schedule key: every 5 minutes
PROSPECTIVE_SCAN_SCHEDULE = 60 * 5


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    time_limit=300,
    name="prospective_memory.scan_and_fire",
)
def prospective_memory_scan_task(self) -> dict[str, Any]:
    """
    Scan pending ProspectiveReminder rows and fire those whose trigger_at <= now().

    Runs every 5 minutes via Celery beat.
    """
    try:
        import asyncio

        result = asyncio.run(_scan_and_fire_async())
        return result
    except Exception as exc:
        logger.error("prospective_memory_scan_task failed: %s", exc)
        raise self.retry(exc=exc, countdown=60)


async def _scan_and_fire_async() -> dict[str, Any]:
    """Load overdue reminders, publish events, and mark fired."""
    now = datetime.now(timezone.utc)
    fired_count = 0
    skipped_count = 0

    async with AsyncSessionLocal() as session:
        stmt = select(ProspectiveReminder).where(
            and_(
                ProspectiveReminder.status == "pending",
                ProspectiveReminder.trigger_at <= now,
            )
        )
        result = await session.execute(stmt)
        reminders = list(result.scalars().all())

        for reminder in reminders:
            try:
                svc = EventPublishingService(session, reminder.org_id)
                await svc.publish_event(
                    event_type="prospective_reminder_fired",
                    resource_type="prospective_reminder",
                    resource_id=str(reminder.id),
                    payload={
                        "reminder_content": reminder.reminder_content,
                        "trigger_type": reminder.trigger_type,
                        "trigger_at": (
                            reminder.trigger_at.isoformat()
                            if reminder.trigger_at
                            else None
                        ),
                    },
                )
                reminder.status = "fired"
                reminder.fired_at = now
                fired_count += 1
            except Exception as exc:
                logger.error(
                    "Failed to fire reminder %s: %s", reminder.id, exc
                )
                skipped_count += 1

        await session.commit()

    logger.info(
        "prospective_memory_scan_task: fired=%d skipped=%d",
        fired_count,
        skipped_count,
    )
    return {"fired": fired_count, "skipped": skipped_count}
