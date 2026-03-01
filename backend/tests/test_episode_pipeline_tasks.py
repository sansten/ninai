from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode, EpisodeStatus
from app.models.episode_event import EpisodeEvent
from app.models.memory import MemoryMetadata
from app.tasks.episode_pipeline import route_memory_to_episode, summarize_episode


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _seed_memory(
    session: AsyncSession,
    *,
    org_id: str,
    user_id: str,
    title: str,
    content_preview: str,
    tags: list[str] | None = None,
    entities: dict | None = None,
) -> str:
    memory_id = str(uuid4())
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
            "tags": tags or [],
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
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        },
    )
    return memory_id


@pytest.mark.asyncio
async def test_episode_router_creates_new_episode_when_no_match(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="New Issue: Router Offline",
        content_preview="Customer reports their home router is completely offline since this morning.",
        tags=["network", "router"],
        entities={"customer_id": "C-100"},
    )
    await db_session.commit()

    result = await route_memory_to_episode(
        org_id=test_org_id,
        memory_id=memory_id,
        actor_user_id=test_user_id,
    )

    assert result["attached_existing"] is False
    assert result["episode_id"]

    episode = await db_session.get(Episode, result["episode_id"])
    assert episode is not None
    assert episode.organization_id == test_org_id
    assert episode.status == EpisodeStatus.OPEN


@pytest.mark.asyncio
async def test_episode_router_attaches_to_existing_episode_when_similar(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    existing_episode_id = str(uuid4())
    now = datetime.now(timezone.utc)

    await db_session.execute(
        insert(Episode),
        {
            "id": existing_episode_id,
            "organization_id": test_org_id,
            "scope_type": "personal",
            "scope_id": None,
            "owner_user_id": test_user_id,
            "episode_type": "support_case",
            "status": "open",
            "title": "Customer Router Offline",
            "summary": "Customer reported router downtime and connectivity failures.",
            "started_at": now,
            "last_event_at": now,
            "resolved_at": None,
            "tags": ["network", "router"],
            "entities": {"customer_id": "C-100"},
            "created_at": now,
            "updated_at": now,
        },
    )

    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="Router still offline",
        content_preview="Follow-up: customer still has router downtime and network outage.",
        tags=["network", "router"],
        entities={"customer_id": "C-100"},
    )
    await db_session.commit()

    result = await route_memory_to_episode(
        org_id=test_org_id,
        memory_id=memory_id,
        actor_user_id=test_user_id,
    )

    assert result["attached_existing"] is True
    assert result["episode_id"] == existing_episode_id


@pytest.mark.asyncio
async def test_episode_summarizer_updates_summary_from_recent_events(db_session: AsyncSession, test_org_id: str, test_user_id: str):
    episode_id = str(uuid4())
    now = datetime.now(timezone.utc)

    await db_session.execute(
        insert(Episode),
        {
            "id": episode_id,
            "organization_id": test_org_id,
            "scope_type": "personal",
            "scope_id": None,
            "owner_user_id": test_user_id,
            "episode_type": "support_case",
            "status": "open",
            "title": "Intermittent Packet Loss",
            "summary": None,
            "started_at": now,
            "last_event_at": now,
            "resolved_at": None,
            "tags": ["network"],
            "entities": {"customer_id": "C-200"},
            "created_at": now,
            "updated_at": now,
        },
    )

    await db_session.execute(
        insert(EpisodeEvent),
        [
            {
                "id": str(uuid4()),
                "organization_id": test_org_id,
                "episode_id": episode_id,
                "memory_id": None,
                "event_type": "user_report",
                "event_ts": now,
                "actor_type": "user",
                "actor_id": test_user_id,
                "content": "Customer reports intermittent packet loss.",
                "payload": {},
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": str(uuid4()),
                "organization_id": test_org_id,
                "episode_id": episode_id,
                "memory_id": None,
                "event_type": "agent_action",
                "event_ts": now,
                "actor_type": "agent",
                "actor_id": test_user_id,
                "content": "Ran diagnostics and captured line metrics.",
                "payload": {},
                "created_at": now,
                "updated_at": now,
            },
        ],
    )
    await db_session.commit()

    result = await summarize_episode(
        org_id=test_org_id,
        episode_id=episode_id,
        actor_user_id=test_user_id,
    )

    assert result["episode_id"] == episode_id
    assert result["event_count"] == 2

    refreshed = await db_session.get(Episode, episode_id)
    assert refreshed is not None
    assert refreshed.summary is not None
    assert "Intermittent Packet Loss" in refreshed.summary
    assert "Customer reports intermittent packet loss" in refreshed.summary
