"""Webhook delivery tasks."""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.services.webhook_service import WebhookService


logger = get_task_logger(__name__)


@celery_app.task(name="app.tasks.webhooks.dispatch_webhooks_task")
def dispatch_webhooks_task() -> int:
    """Dispatch due webhook deliveries."""

    async def _run() -> int:
        async with async_session_factory() as session:
            async with session.begin():
                svc = WebhookService(session)
                return await svc.dispatch_due_deliveries(limit=50)

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run())
