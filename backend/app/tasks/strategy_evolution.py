"""Strategy evolution pipeline task — Phase 50.

Nightly job to evolve strategy library from external action outcomes.
"""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.models.action_execution_record import ActionExecutionRecord
from app.services.continuous_learning_service import StrategyEvolutionService

logger = get_task_logger(__name__)


@celery_app.task(name="app.tasks.strategy_evolution.strategy_evolution_pipeline_task")
def strategy_evolution_pipeline_task(*, org_id: str | None = None, window_days: int = 30) -> dict:
    """Run strategy evolution for one org or all orgs with recent outcomes."""

    async def _run_for_org(target_org_id: str) -> dict:
        async with async_session_factory() as db:
            async with db.begin():
                svc = StrategyEvolutionService(db)
                result = await svc.run(org_id=target_org_id, window_days=window_days)
                return {
                    "org_id": target_org_id,
                    "promoted": result.promoted,
                    "pruned": result.pruned,
                    "drift_alerts": [a.__dict__ for a in result.drift_alerts],
                    "strategy_metrics": [m.__dict__ for m in result.strategy_metrics],
                    "playbook_metrics": [m.__dict__ for m in result.playbook_metrics],
                }

    async def _resolve_orgs() -> list[str]:
        if org_id:
            return [org_id]

        async with async_session_factory() as db:
            res = await db.execute(
                select(ActionExecutionRecord.organization_id).distinct()
            )
            org_ids = [str(x) for x in res.scalars().all() if x]
            return org_ids

    async def _run() -> dict:
        target_orgs = await _resolve_orgs()
        if not target_orgs:
            return {"status": "no-op", "processed_orgs": 0, "results": []}

        results = []
        for oid in target_orgs:
            try:
                results.append(await _run_for_org(oid))
            except Exception as exc:
                logger.exception("strategy evolution failed for org %s", oid)
                results.append({
                    "org_id": oid,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return {
            "status": "ok",
            "processed_orgs": len(target_orgs),
            "results": results,
        }

    return asyncio.run(_run())
