"""Prediction error consolidation pipeline — Phase 85.

Nightly task: scans PredictionErrorLog for unconsolidated high-divergence
events and bumps the importance scores of the associated retrieval chunks
in Qdrant so they surface more readily in future retrievals.

The effect: Ninai learns hardest from what surprised it most (prediction-error
theory of memory consolidation).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.models.prediction_error_log import PredictionErrorLog
from app.services.prediction_error_service import PredictionErrorService

logger = get_task_logger(__name__)

_IMPORTANCE_BOOST = 0.15   # added to existing importance score for surprised chunks
_MAX_BOOST = 0.95          # ceiling so scores don't saturate


@celery_app.task(
    name="app.tasks.prediction_error_pipeline.consolidate_prediction_errors_task"
)
def consolidate_prediction_errors_task(
    *,
    org_id: str | None = None,
    since_days: int = 7,
) -> dict:
    """Consolidate high-divergence prediction errors for one or all orgs."""

    async def _run_for_org(target_org_id: str, since: datetime) -> dict:
        async with async_session_factory() as db:
            async with db.begin():
                svc = PredictionErrorService()
                events = await svc.load_unconsolidated(
                    db, org_id=target_org_id, since=since
                )
                if not events:
                    return {"org_id": target_org_id, "events": 0, "chunks_boosted": 0}

                boosted = await _boost_chunks(target_org_id, events)
                log_ids = [str(e.id) for e in events]
                await svc.mark_consolidated(db, log_ids=log_ids)

                return {
                    "org_id": target_org_id,
                    "events": len(events),
                    "chunks_boosted": boosted,
                }

    async def _boost_chunks(org_id: str, events: list) -> int:
        """Attempt to bump importance on Qdrant payloads for surprised chunks."""
        boosted = 0
        try:
            from app.v2.retrieval.qdrant_client import QdrantRetriever
            retriever = QdrantRetriever()
        except Exception:
            return 0

        seen: set[str] = set()
        for event in events:
            for chunk_id in (event.chunk_ids or []):
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                try:
                    await retriever.boost_importance(
                        org_id=org_id,
                        chunk_id=chunk_id,
                        boost=_IMPORTANCE_BOOST,
                        ceiling=_MAX_BOOST,
                    )
                    boosted += 1
                except Exception as exc:
                    logger.debug("Chunk boost failed %s: %s", chunk_id, exc)
        return boosted

    async def _resolve_orgs() -> list[str]:
        if org_id:
            return [org_id]
        async with async_session_factory() as db:
            res = await db.execute(
                select(PredictionErrorLog.organization_id).distinct()
            )
            return [str(x) for x in res.scalars().all() if x]

    async def _run() -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=since_days)
        target_orgs = await _resolve_orgs()
        if not target_orgs:
            return {"status": "no-op", "processed_orgs": 0}

        results = []
        for oid in target_orgs:
            try:
                results.append(await _run_for_org(oid, since))
            except Exception as exc:
                logger.exception("prediction error consolidation failed for org %s", oid)
                results.append({"org_id": oid, "error": f"{type(exc).__name__}: {exc}"})

        total_events = sum(r.get("events", 0) for r in results)
        return {
            "status": "ok",
            "processed_orgs": len(target_orgs),
            "total_events_consolidated": total_events,
            "results": results,
        }

    return asyncio.run(_run())
