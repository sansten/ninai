"""Memory Sleep Pipeline — Phase 40.

Nightly Celery beat task (02:00 UTC) that runs the offline memory sleep
cycle across all active tenants.

For each org it:
1. Selects long-term memories not recently sleep-processed (cursor-based).
2. Runs MemorySleepAgent on each memory's enrichment snapshot.
3. Tallies sleep_action counts and association links discovered.
4. Writes (or upserts) a SleepCycleReport row for the night.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from celery.utils.log import get_task_logger
from sqlalchemy import select

from app.agents.memory_sleep_agent import MemorySleepAgent
from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import async_session_factory, get_tenant_session
from app.models.memory import MemoryMetadata
from app.models.organization import Organization
from app.models.sleep_cycle_report import SleepCycleReport


logger = get_task_logger(__name__)

# How many memories to process per org per night (safety cap).
_DEFAULT_BATCH_SIZE = 500
# Memories not accessed within this window are eligible for sleep processing.
_SLEEP_LOOKBACK_DAYS = 1


def _broker_enabled() -> bool:
    broker = celery_app.conf.broker_url
    return bool(broker) and not str(broker).startswith("memory://")


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


def _build_agent_context(memory: MemoryMetadata) -> dict[str, Any]:
    """Assemble the AgentContext dict from a MemoryMetadata row."""
    enrichment = dict(memory.enrichment or {}) if hasattr(memory, "enrichment") else {}
    meta = dict(memory.metadata or {}) if hasattr(memory, "metadata") else {}

    # Merge metadata fields that carry enrichment signals from prior agents.
    for key in (
        "freshness_score", "is_stale", "reference_count",
        "consolidation_action", "merge_candidates", "redundancy_score",
        "episode_key", "episode_cohesion", "cross_episode_links",
        "similar_memories",
    ):
        if key not in enrichment and key in meta:
            enrichment[key] = meta[key]

    return {
        "memory": {
            "id": str(memory.id),
            "content": str(getattr(memory, "content", "") or ""),
            "created_at": (
                memory.created_at.isoformat()
                if getattr(memory, "created_at", None)
                else None
            ),
            "enrichment": enrichment,
        }
    }


async def _run_sleep_cycle_async(*, batch_size: int, lookback_days: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    today = now.date()
    cutoff = now - timedelta(days=lookback_days)

    async with async_session_factory() as session:
        org_ids = (
            await session.execute(
                select(Organization.id).where(Organization.is_active.is_(True))
            )
        ).scalars().all()

    agent = MemorySleepAgent()
    service_user_id = str(getattr(settings, "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    orgs_processed = 0
    total_scanned = 0

    for org_id in org_ids:
        orgs_processed += 1
        tally: dict[str, int] = {
            "retained": 0, "strengthened": 0, "merged": 0, "pruned": 0,
        }
        associations_created = 0
        per_memory_details: list[dict[str, Any]] = []

        try:
            async with get_tenant_session(
                user_id=service_user_id,
                org_id=str(org_id),
                roles=service_roles,
                clearance_level=0,
                justification="nightly_memory_sleep_cycle",
            ) as tenant_session:
                # Fetch long-term memories not updated recently (sleep-eligible).
                stmt = (
                    select(MemoryMetadata)
                    .where(
                        MemoryMetadata.organization_id == str(org_id),
                        MemoryMetadata.memory_type != "short_term",
                        MemoryMetadata.updated_at < cutoff,
                    )
                    .order_by(MemoryMetadata.updated_at.asc())
                    .limit(int(batch_size))
                )
                memories = (await tenant_session.execute(stmt)).scalars().all()

                for memory in memories:
                    total_scanned += 1
                    ctx = _build_agent_context(memory)
                    try:
                        result = await agent.run(str(memory.id), ctx)
                        action = result.outputs.get("sleep_action", "retain")
                        links = result.outputs.get("association_links") or []

                        action_key = action + "d" if action in ("retain", "strengthen", "merge", "prune") else "retained"
                        # map action → tally key
                        _action_map = {
                            "retain":    "retained",
                            "strengthen": "strengthened",
                            "merge":     "merged",
                            "prune":     "pruned",
                        }
                        tally_key = _action_map.get(action, "retained")
                        tally[tally_key] += 1
                        associations_created += len(links)

                        per_memory_details.append({
                            "memory_id": str(memory.id),
                            "sleep_action": action,
                            "strength_delta": result.outputs.get("strength_delta", 0.0),
                            "association_links": links,
                            "confidence": result.confidence,
                        })
                    except Exception:
                        logger.exception(
                            "MemorySleepAgent failed for memory",
                            extra={"memory_id": str(memory.id), "org_id": str(org_id)},
                        )
                        tally["retained"] += 1

                # Upsert SleepCycleReport for this org + date.
                existing_stmt = select(SleepCycleReport).where(
                    SleepCycleReport.organization_id == str(org_id),
                    SleepCycleReport.run_date == today,
                )
                existing = (await tenant_session.execute(existing_stmt)).scalar_one_or_none()

                if existing is not None:
                    existing.memories_scanned = len(memories)
                    existing.retained = tally["retained"]
                    existing.strengthened = tally["strengthened"]
                    existing.merged = tally["merged"]
                    existing.pruned = tally["pruned"]
                    existing.associations_created = associations_created
                    existing.report_payload = {"details": per_memory_details}
                    existing.completed_at = now
                else:
                    report = SleepCycleReport(
                        organization_id=str(org_id),
                        run_date=today,
                        memories_scanned=len(memories),
                        retained=tally["retained"],
                        strengthened=tally["strengthened"],
                        merged=tally["merged"],
                        pruned=tally["pruned"],
                        associations_created=associations_created,
                        report_payload={"details": per_memory_details},
                        completed_at=now,
                    )
                    tenant_session.add(report)

                await tenant_session.commit()

        except Exception:
            logger.exception(
                "Sleep cycle pipeline failed for org", extra={"org_id": str(org_id)}
            )

    return {
        "ok": True,
        "orgs_processed": orgs_processed,
        "total_scanned": total_scanned,
        "batch_size": int(batch_size),
        "lookback_days": int(lookback_days),
        "ran_at": now.isoformat(),
    }


@celery_app.task(bind=True)
def memory_sleep_task(self, batch_size: int = _DEFAULT_BATCH_SIZE, lookback_days: int = _SLEEP_LOOKBACK_DAYS):
    """Nightly memory sleep cycle — runs per-memory sleep action classification.

    Disabled (no-op) when Celery runs with the in-memory broker used during tests.
    """
    if not _broker_enabled():
        return {
            "ok": True,
            "skipped": True,
            "reason": "broker_disabled",
        }

    try:
        return _run_async(
            _run_sleep_cycle_async(batch_size=batch_size, lookback_days=lookback_days)
        )
    except Exception as e:
        logger.exception("Memory sleep pipeline task failed")
        raise e
