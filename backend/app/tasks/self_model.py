"""SelfModel Celery tasks."""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import get_tenant_session
from app.services.self_model_service import SelfModelService


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


@celery_app.task(name="app.tasks.self_model.self_model_recompute_task")
def self_model_recompute_task(*, org_id: str) -> dict:
    """Recompute SelfModel profile for an organization."""

    async def _run() -> dict:
        service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
        service_roles = "system_admin" if service_user_id else ""

        async with get_tenant_session(
            user_id=service_user_id,
            org_id=org_id,
            roles=service_roles,
            clearance_level=0,
            justification="self_model_recompute",
        ) as session:
            svc = SelfModelService(session)
            prof = await svc.recompute_profile(org_id=org_id)
            return {
                "ok": True,
                "org_id": org_id,
                "last_updated": (prof.last_updated).isoformat(),
            }

    try:
        return _run_async(_run())
    except Exception as e:
        logger.exception("self_model_recompute_task failed")
        return {"ok": False, "org_id": org_id, "error": str(e)}
