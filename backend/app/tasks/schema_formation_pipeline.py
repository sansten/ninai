"""Schema formation pipeline — Phase 87.

Nightly task: scans recent memories, clusters them by event type,
and crystallizes recurring patterns into SchemaFrame rows.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from celery.utils.log import get_task_logger
from sqlalchemy import select, text

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.services.schema_formation_service import SchemaFormationService

logger = get_task_logger(__name__)

_WINDOW_DAYS = 30
_MAX_MEMORIES_PER_ORG = 2000


@celery_app.task(name="app.tasks.schema_formation_pipeline.schema_formation_task")
def schema_formation_task(
    *,
    org_id: str | None = None,
    window_days: int = _WINDOW_DAYS,
) -> dict:
    """Induce schema frames from recent memory clusters for one or all orgs."""

    async def _load_memories(db, target_org_id: str, since: datetime) -> list[dict]:
        try:
            res = await db.execute(
                text(
                    "SELECT id::text, content AS text, 0.7 AS confidence "
                    "FROM memories "
                    "WHERE organization_id = :org_id "
                    "  AND created_at >= :since "
                    "  AND content IS NOT NULL "
                    "ORDER BY created_at DESC "
                    "LIMIT :lim"
                ),
                {"org_id": target_org_id, "since": since, "lim": _MAX_MEMORIES_PER_ORG},
            )
            return [dict(r._mapping) for r in res.fetchall()]
        except Exception as exc:
            logger.debug("Memory load failed for schema formation: %s", exc)
            return []

    async def _run_for_org(target_org_id: str, since: datetime) -> dict:
        async with async_session_factory() as db:
            async with db.begin():
                memories = await _load_memories(db, target_org_id, since)
                if not memories:
                    return {"org_id": target_org_id, "schemas_upserted": 0, "memories_scanned": 0}

                svc = SchemaFormationService()
                clusters: dict[str, list[dict]] = {}
                for mem in memories:
                    event_type = svc.extract_event_type(mem.get("text") or "")
                    if event_type:
                        clusters.setdefault(event_type, []).append(mem)

                schemas_upserted = 0
                for event_type, cluster in clusters.items():
                    candidate = svc.induce_schema(event_type, cluster)
                    if candidate:
                        await svc.upsert_schema(db, org_id=target_org_id, candidate=candidate)
                        schemas_upserted += 1

                return {
                    "org_id": target_org_id,
                    "memories_scanned": len(memories),
                    "event_clusters": len(clusters),
                    "schemas_upserted": schemas_upserted,
                }

    async def _resolve_orgs() -> list[str]:
        if org_id:
            return [org_id]
        try:
            async with async_session_factory() as db:
                res = await db.execute(
                    text("SELECT DISTINCT organization_id::text FROM memories WHERE organization_id IS NOT NULL")
                )
                return [str(r[0]) for r in res.fetchall() if r[0]]
        except Exception:
            return []

    async def _run() -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=window_days)
        target_orgs = await _resolve_orgs()
        if not target_orgs:
            return {"status": "no-op", "processed_orgs": 0}

        results = []
        for oid in target_orgs:
            try:
                results.append(await _run_for_org(oid, since))
            except Exception as exc:
                logger.exception("Schema formation failed for org %s", oid)
                results.append({"org_id": oid, "error": f"{type(exc).__name__}: {exc}"})

        return {
            "status": "ok",
            "processed_orgs": len(target_orgs),
            "results": results,
        }

    return asyncio.run(_run())
