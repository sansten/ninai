"""Fact extraction and contradiction detection tasks (PR3)."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from uuid import uuid4

from celery.utils.log import get_task_logger
from sqlalchemy import and_, select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.models.contradiction import Contradiction, ContradictionSeverity
from app.models.memory import MemoryMetadata
from app.models.memory_fact import MemoryFact, MemoryFactStatus


logger = get_task_logger(__name__)

_ASYNC_LOOP: asyncio.AbstractEventLoop | None = None


def _run_async(coro):
    global _ASYNC_LOOP
    if _ASYNC_LOOP is None or _ASYNC_LOOP.is_closed():
        _ASYNC_LOOP = asyncio.new_event_loop()
    return _ASYNC_LOOP.run_until_complete(coro)


def _extract_fact_candidates(text: str) -> list[tuple[str, str, str, float]]:
    """Heuristic extraction for common fact statements.

    Patterns intentionally simple for deterministic extraction in OSS baseline.
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []

    candidates: list[tuple[str, str, str, float]] = []

    patterns = [
        (r"([A-Za-z0-9_\- ]+?)\s+phone\s+is\s+([A-Za-z0-9+\- ]{5,30})", "phone", 0.82),
        (r"([A-Za-z0-9_\- ]+?)\s+address\s+is\s+([A-Za-z0-9,\- ]{5,120})", "address", 0.78),
        (r"([A-Za-z0-9_\- ]+?)\s+plan\s+is\s+([A-Za-z0-9_\- ]{3,80})", "plan", 0.80),
        (r"([A-Za-z0-9_\- ]+?)\s+status\s+is\s+([A-Za-z0-9_\- ]{3,80})", "status", 0.72),
    ]

    for pattern, predicate, confidence in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            subject = match.group(1).strip().lower()
            obj = match.group(2).strip().lower()
            if subject and obj:
                candidates.append((subject, predicate, obj, confidence))

    return candidates


async def extract_facts_from_memory(*, org_id: str, memory_id: str, actor_user_id: str) -> dict:
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            memory = await db.get(MemoryMetadata, memory_id)
            if not memory or memory.organization_id != org_id:
                raise ValueError(f"Memory {memory_id} not found for organization {org_id}")

            source_text = " ".join([memory.title or "", memory.content_preview or ""])
            candidates = _extract_fact_candidates(source_text)

            created_ids: list[str] = []
            valid_from = memory.created_at or datetime.now(timezone.utc)

            for subject, predicate, obj, confidence in candidates:
                existing_stmt = select(MemoryFact).where(
                    and_(
                        MemoryFact.organization_id == org_id,
                        MemoryFact.subject == subject,
                        MemoryFact.predicate == predicate,
                        MemoryFact.object == obj,
                        MemoryFact.source_memory_id == memory_id,
                    )
                )
                existing = (await db.execute(existing_stmt)).scalar_one_or_none()
                if existing:
                    continue

                fact = MemoryFact(
                    id=str(uuid4()),
                    organization_id=org_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=confidence,
                    source_memory_id=memory_id,
                    valid_from=valid_from,
                    valid_to=None,
                    supersedes_fact_id=None,
                    status=MemoryFactStatus.ACTIVE,
                    contradiction_group_id=None,
                )
                db.add(fact)
                created_ids.append(fact.id)

            await db.flush()

            return {
                "memory_id": memory_id,
                "created_count": len(created_ids),
                "fact_ids": created_ids,
            }


async def detect_contradictions(*, org_id: str, fact_ids: list[str], actor_user_id: str) -> dict:
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            if not fact_ids:
                return {"created": 0, "pairs": []}

            facts_stmt = select(MemoryFact).where(
                and_(
                    MemoryFact.organization_id == org_id,
                    MemoryFact.id.in_(fact_ids),
                )
            )
            new_facts = list((await db.execute(facts_stmt)).scalars().all())

            created_pairs: list[tuple[str, str]] = []

            for fact in new_facts:
                other_stmt = select(MemoryFact).where(
                    and_(
                        MemoryFact.organization_id == org_id,
                        MemoryFact.subject == fact.subject,
                        MemoryFact.predicate == fact.predicate,
                        MemoryFact.id != fact.id,
                        MemoryFact.status == MemoryFactStatus.ACTIVE,
                    )
                )
                others = list((await db.execute(other_stmt)).scalars().all())

                for other in others:
                    if other.object.strip().lower() == fact.object.strip().lower():
                        continue

                    contradiction = Contradiction(
                        id=str(uuid4()),
                        organization_id=org_id,
                        fact_a=other.id,
                        fact_b=fact.id,
                        reason=(
                            f"Conflicting values for {fact.subject}.{fact.predicate}: "
                            f"'{other.object}' vs '{fact.object}'"
                        ),
                        severity=ContradictionSeverity.HIGH,
                        created_at=datetime.now(timezone.utc),
                        resolved_at=None,
                    )
                    db.add(contradiction)

                    group_id = str(uuid4())
                    fact.status = MemoryFactStatus.DISPUTED
                    other.status = MemoryFactStatus.DISPUTED
                    fact.contradiction_group_id = group_id
                    other.contradiction_group_id = group_id

                    created_pairs.append((other.id, fact.id))

            await db.flush()

            return {
                "created": len(created_pairs),
                "pairs": created_pairs,
            }


@celery_app.task(bind=True, max_retries=5, autoretry_for=(Exception,), dont_autoretry_for=(ValueError,), retry_backoff=True)
def fact_extractor_task(
    self,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    if storage != "long_term":
        return {"status": "skipped", "reason": "memory_not_long_term", "memory_id": memory_id}

    actor_user_id = initiator_user_id or "00000000-0000-0000-0000-000000000001"
    extraction = _run_async(extract_facts_from_memory(org_id=org_id, memory_id=memory_id, actor_user_id=actor_user_id))
    contradiction = _run_async(detect_contradictions(org_id=org_id, fact_ids=extraction["fact_ids"], actor_user_id=actor_user_id))

    return {
        "status": "ok",
        "org_id": org_id,
        "memory_id": memory_id,
        "trace_id": trace_id,
        "extraction": extraction,
        "contradictions": contradiction,
    }


def enqueue_fact_pipeline(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    broker = celery_app.conf.broker_url
    if not broker or str(broker).startswith("memory://"):
        return None

    try:
        return fact_extractor_task.si(
            org_id=org_id,
            memory_id=memory_id,
            initiator_user_id=initiator_user_id,
            trace_id=trace_id,
            storage=storage,
        ).apply_async(retry=False)
    except Exception as exc:
        logger.warning("fact pipeline enqueue failed for memory_id=%s: %s", memory_id, exc)
        return None
