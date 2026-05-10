from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryMetadata
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.schemas.episode import EpisodeCreate, EpisodeEventCreate
from app.services.episode_service import EpisodeService


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
            "created_at": now,
            "updated_at": now,
        },
    )
    return memory_id


@pytest.mark.asyncio
async def test_episode_service_projects_case_episode_into_memory_episode(
    db_session: AsyncSession,
    test_org_id: str,
    test_user_id: str,
):
    service = EpisodeService(db_session, test_user_id, test_org_id)

    episode = await service.create_episode(
        EpisodeCreate(
            scope_type="personal",
            episode_type="research_thread",
            title="Model eval investigation",
            tags=["eval", "memory"],
            entities={"benchmark": "locomo"},
        )
    )
    await db_session.commit()

    projected = await db_session.get(MemoryEpisode, episode.id)
    assert projected is not None
    assert projected.status == "open"
    assert projected.title == "Model eval investigation"
    assert projected.extra_metadata["source_case_episode_id"] == episode.id
    assert projected.extra_metadata["projection_source"] == "case_episode_projection"
    assert projected.extra_metadata["tags"] == ["eval", "memory"]
    assert projected.extra_metadata["entities"]["benchmark"] == "locomo"


@pytest.mark.asyncio
async def test_episode_service_event_with_memory_creates_membership_and_resolve_closes_projection(
    db_session: AsyncSession,
    test_org_id: str,
    test_user_id: str,
):
    memory_id = await _seed_memory(
        db_session,
        org_id=test_org_id,
        user_id=test_user_id,
        title="Failure report",
        content_preview="The retrieval chain dropped a relevant memory during evaluation.",
        tags=["retrieval"],
        entities={"failure_mode": "coverage"},
    )
    await db_session.commit()

    service = EpisodeService(db_session, test_user_id, test_org_id)
    episode = await service.create_episode(
        EpisodeCreate(
            scope_type="personal",
            episode_type="support_case",
            title="Coverage regression",
        )
    )
    await service.create_episode_event(
        EpisodeEventCreate(
            episode_id=episode.id,
            memory_id=memory_id,
            event_type="user_report",
            actor_type="user",
            content="Attached the failure report to the investigation.",
        )
    )
    await service.resolve_episode(episode.id)
    await db_session.commit()

    projected = await db_session.get(MemoryEpisode, episode.id)
    assert projected is not None
    assert projected.status == "closed"
    assert projected.message_count == 1
    assert projected.extra_metadata["latest_memory_id"] == memory_id
    assert projected.extra_metadata["entities"]["failure_mode"] == "coverage"

    membership_stmt = select(MemoryEpisodeMembership).where(
        MemoryEpisodeMembership.organization_id == test_org_id,
        MemoryEpisodeMembership.episode_id == episode.id,
        MemoryEpisodeMembership.memory_id == memory_id,
    )
    membership = (await db_session.execute(membership_stmt)).scalar_one_or_none()
    assert membership is not None
    assert membership.position == 0
