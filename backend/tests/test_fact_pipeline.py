from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.contradiction import Contradiction
from app.models.memory import MemoryMetadata
from app.models.memory_fact import MemoryFact, MemoryFactStatus
from app.models.user import User
from app.services.fact_service import FactService
from app.tasks.fact_pipeline import detect_contradictions, extract_facts_from_memory


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _seed_memory(
    session: AsyncSession,
    *,
    org_id: str,
    user_id: str,
    title: str,
    content_preview: str,
    entities: dict | None = None,
) -> str:
    memory_id = str(uuid4())
    now = datetime.now(timezone.utc)
    await session.execute(
        insert(MemoryMetadata),
        {
            "id": memory_id,
            "organization_id": org_id,
            "owner_id": user_id,
            "scope": "personal",
            "scope_id": None,
            "memory_type": "long_term",
            "classification": "internal",
            "required_clearance": 0,
            "title": title,
            "content_preview": content_preview,
            "content_hash": _hash(content_preview),
            "tags": [],
            "entities": entities or {},
            "extra_metadata": {},
            "source_type": "manual",
            "source_id": None,
            "vector_id": f"vec-{memory_id}",
            "embedding_model": "test-model",
            "access_count": 0,
            "last_accessed_at": None,
            "retention_days": None,
            "expires_at": None,
            "legal_hold": False,
            "is_active": True,
            "is_promoted": False,
            "promoted_from_id": None,
            "created_at": now,
            "updated_at": now,
        },
    )
    return memory_id


@pytest.mark.asyncio
async def test_fact_extraction_creates_facts(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="john plan is premium",
        content_preview="john phone is 555-123-4567",
    )
    await db_session.commit()

    result = await extract_facts_from_memory(
        org_id=test_org_id,
        memory_id=memory_id,
        actor_user_id=test_user_id,
    )

    assert result["created_count"] >= 1

    facts = list(
        (
            await db_session.execute(
                select(MemoryFact).where(
                    MemoryFact.organization_id == test_org_id,
                    MemoryFact.source_memory_id == memory_id,
                )
            )
        ).scalars().all()
    )
    assert len(facts) >= 1
    assert all(f.status == MemoryFactStatus.ACTIVE for f in facts)


@pytest.mark.asyncio
async def test_fact_extraction_handles_speaker_state_and_origin_relation(
    db_session: AsyncSession,
    test_org_id: str,
    test_user_id: str,
):
    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="Conversation note",
        content_preview="Caroline: I'm single and I moved from Sweden four years ago.",
    )
    await db_session.commit()

    result = await extract_facts_from_memory(
        org_id=test_org_id,
        memory_id=memory_id,
        actor_user_id=test_user_id,
    )
    assert result["created_count"] >= 2

    facts = list(
        (
            await db_session.execute(
                select(MemoryFact).where(
                    MemoryFact.organization_id == test_org_id,
                    MemoryFact.source_memory_id == memory_id,
                )
            )
        ).scalars().all()
    )
    fact_tuples = {(f.subject, f.predicate, f.object) for f in facts}

    assert ("caroline", "relationship_status", "single") in fact_tuples
    assert any(subject == "caroline" and predicate == "moved_from" and obj.startswith("sweden") for subject, predicate, obj in fact_tuples)


@pytest.mark.asyncio
async def test_fact_extraction_does_not_use_entity_only_hint_for_first_person_state(
    db_session: AsyncSession,
    test_org_id: str,
    test_user_id: str,
):
    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="Ambiguous first person",
        content_preview="I'm single and moved from Sweden.",
        entities={"name": "Melanie"},
    )
    await db_session.commit()

    _ = await extract_facts_from_memory(
        org_id=test_org_id,
        memory_id=memory_id,
        actor_user_id=test_user_id,
    )

    facts = list(
        (
            await db_session.execute(
                select(MemoryFact).where(
                    MemoryFact.organization_id == test_org_id,
                    MemoryFact.source_memory_id == memory_id,
                )
            )
        ).scalars().all()
    )
    fact_tuples = {(f.subject, f.predicate, f.object) for f in facts}

    assert ("melanie", "relationship_status", "single") not in fact_tuples
    assert ("melanie", "moved_from", "sweden") not in fact_tuples


@pytest.mark.asyncio
async def test_contradiction_detection_flags_conflict(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    mem_a = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="john plan is basic",
        content_preview="account update",
    )
    mem_b = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="john plan is premium",
        content_preview="account update",
    )
    await db_session.commit()

    first = await extract_facts_from_memory(org_id=test_org_id, memory_id=mem_a, actor_user_id=test_user_id)
    second = await extract_facts_from_memory(org_id=test_org_id, memory_id=mem_b, actor_user_id=test_user_id)

    contradiction = await detect_contradictions(
        org_id=test_org_id,
        fact_ids=second["fact_ids"],
        actor_user_id=test_user_id,
    )

    assert contradiction["created"] >= 1

    contradictions = list(
        (
            await db_session.execute(
                select(Contradiction).where(Contradiction.organization_id == test_org_id)
            )
        ).scalars().all()
    )
    assert len(contradictions) >= 1

    facts = list(
        (
            await db_session.execute(
                select(MemoryFact).where(
                    MemoryFact.organization_id == test_org_id,
                    MemoryFact.id.in_(first["fact_ids"] + second["fact_ids"]),
                )
            )
        ).scalars().all()
    )
    assert any(f.status == MemoryFactStatus.DISPUTED for f in facts)


@pytest.mark.asyncio
async def test_fact_enrichment_active_preferred_over_superseded(db_session: AsyncSession, test_org_id: str):
    user_row = await db_session.execute(select(User.id).limit(1))
    any_owner_id = user_row.scalar_one_or_none()
    if any_owner_id is None:
        pytest.fail("Expected seeded test user for db_session fixture")

    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=any_owner_id,
        title="john plan is premium",
        content_preview="profile update",
    )
    now = datetime.now(timezone.utc)

    active_id = str(uuid4())
    superseded_id = str(uuid4())

    await db_session.execute(
        insert(MemoryFact),
        [
            {
                "id": superseded_id,
                "organization_id": test_org_id,
                "subject": "john",
                "predicate": "plan",
                "object": "basic",
                "confidence": 0.6,
                "source_memory_id": memory_id,
                "valid_from": now,
                "valid_to": now,
                "supersedes_fact_id": None,
                "status": "superseded",
                "contradiction_group_id": None,
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": active_id,
                "organization_id": test_org_id,
                "subject": "john",
                "predicate": "plan",
                "object": "premium",
                "confidence": 0.9,
                "source_memory_id": memory_id,
                "valid_from": now,
                "valid_to": None,
                "supersedes_fact_id": superseded_id,
                "status": "active",
                "contradiction_group_id": None,
                "created_at": now,
                "updated_at": now,
            },
        ],
    )
    await db_session.commit()

    service = FactService(db_session, test_org_id)
    enrichment = await service.get_enrichment_for_memories([memory_id])

    assert len(enrichment["facts_used"]) == 1
    assert enrichment["facts_used"][0]["id"] == active_id
    assert enrichment["facts_used"][0]["status"] in {"MemoryFactStatus.ACTIVE", "active"}
    assert enrichment["disputed_facts"] == []
