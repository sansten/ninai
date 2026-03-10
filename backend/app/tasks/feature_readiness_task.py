"""Feature readiness evaluation task — Phase 28.

Runs daily (03:00 UTC) and re-evaluates readiness for every organisation that
has at least one stored memory.  Results are persisted in
``feature_readiness_config`` so the API can serve them instantly.
"""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.services.feature_readiness_service import FeatureReadinessService

logger = get_task_logger(__name__)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

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


async def _evaluate_all_orgs() -> dict:
    """Fetch all org IDs with memories and re-evaluate readiness for each."""
    from sqlalchemy import select, distinct
    from app.models.memory import MemoryMetadata

    service = FeatureReadinessService()
    evaluated = 0
    errors = 0

    async with async_session_factory() as db:
        # Collect all org IDs that have at least one memory
        stmt = select(distinct(MemoryMetadata.organization_id))
        result = await db.execute(stmt)
        org_ids = [row[0] for row in result.fetchall()]

    for org_id in org_ids:
        try:
            async with async_session_factory() as db:
                await service.evaluate_all(org_id=org_id, db=db)
                evaluated += 1
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Feature readiness evaluation failed for org",
                extra={"org_id": org_id},
                exc_info=exc,
            )
            errors += 1

    return {"evaluated": evaluated, "errors": errors, "total_orgs": len(org_ids)}


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------

@celery_app.task(name="app.tasks.feature_readiness_task.evaluate_feature_readiness_task")
def evaluate_feature_readiness_task() -> dict:
    """Daily sweep: re-evaluate feature readiness for all tenants."""
    logger.info("Starting nightly feature readiness evaluation")
    result = _run_async(_evaluate_all_orgs())
    logger.info(
        "Feature readiness evaluation complete",
        extra=result,
    )
    return {"status": "success", **result}
