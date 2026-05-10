from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.episode import EpisodeStatus
from app.services.episode_convergence_service import EpisodeConvergenceService


@pytest.mark.asyncio
async def test_sync_case_episode_projection_merges_case_and_memory_state(monkeypatch):
    session = AsyncMock()
    svc = EpisodeConvergenceService(session)

    projected = SimpleNamespace(
        id="ep-1",
        owner_id="user-1",
        scope="personal",
        scope_id=None,
        title=None,
        narrative_summary=None,
        boundary_start=None,
        boundary_end=None,
        boundary_reason=None,
        boundary_confidence=0.0,
        status="open",
        extra_metadata=None,
        message_count=0,
    )
    monkeypatch.setattr(svc, "_get_or_create_projection", AsyncMock(return_value=projected))
    monkeypatch.setattr(svc, "_ensure_membership", AsyncMock())
    monkeypatch.setattr(svc, "_membership_count", AsyncMock(return_value=1))

    started_at = datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc)
    event_ts = started_at + timedelta(minutes=15)
    memory_ts = started_at + timedelta(minutes=10)
    episode = SimpleNamespace(
        id="ep-1",
        organization_id="org-1",
        owner_user_id="user-1",
        scope_type="personal",
        scope_id=None,
        title="Coverage regression",
        summary="The recall floor regressed after a retrieval change.",
        started_at=started_at,
        last_event_at=event_ts,
        resolved_at=None,
        tags=["eval"],
        entities={"ticket": "T-100"},
        episode_type="support_case",
        status=EpisodeStatus.OPEN,
    )
    memory = SimpleNamespace(
        id="mem-1",
        organization_id="org-1",
        owner_id="user-1",
        created_at=memory_ts,
        tags=["retrieval"],
        entities={"failure_mode": "coverage"},
    )

    result = await svc.sync_case_episode_projection(
        episode=episode,
        memory=memory,
        actor_user_id="user-1",
        event_ts=event_ts,
        boundary_reason="episode_service_event",
    )

    assert result is projected
    assert projected.scope == "personal"
    assert projected.title == "Coverage regression"
    assert projected.narrative_summary == "The recall floor regressed after a retrieval change."
    assert projected.boundary_start == started_at
    assert projected.boundary_end == event_ts
    assert projected.status == "open"
    assert projected.message_count == 1
    assert projected.extra_metadata["projection_source"] == "case_episode_projection"
    assert projected.extra_metadata["source_case_episode_id"] == "ep-1"
    assert projected.extra_metadata["latest_memory_id"] == "mem-1"
    assert projected.extra_metadata["tags"] == ["eval", "retrieval"]
    assert projected.extra_metadata["entities"]["ticket"] == "T-100"
    assert projected.extra_metadata["entities"]["failure_mode"] == "coverage"
    svc._ensure_membership.assert_awaited_once()
    session.flush.assert_awaited()


@pytest.mark.asyncio
async def test_sync_case_episode_projection_maps_resolved_case_to_closed(monkeypatch):
    session = AsyncMock()
    svc = EpisodeConvergenceService(session)

    projected = SimpleNamespace(
        id="ep-2",
        owner_id="user-1",
        scope="personal",
        scope_id=None,
        title="Old title",
        narrative_summary="Old summary",
        boundary_start=datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc),
        boundary_end=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
        boundary_reason="episode_service_create",
        boundary_confidence=1.0,
        status="open",
        extra_metadata={"tags": ["ops"], "entities": {"ticket": "T-200"}},
        message_count=0,
    )
    monkeypatch.setattr(svc, "_get_or_create_projection", AsyncMock(return_value=projected))
    monkeypatch.setattr(svc, "_membership_count", AsyncMock(return_value=0))

    resolved_at = datetime(2026, 5, 1, 11, 0, tzinfo=timezone.utc)
    episode = SimpleNamespace(
        id="ep-2",
        organization_id="org-1",
        owner_user_id="user-1",
        scope_type="personal",
        scope_id=None,
        title="Resolved incident",
        summary="The case is closed.",
        started_at=datetime(2026, 5, 1, 9, 30, tzinfo=timezone.utc),
        last_event_at=resolved_at,
        resolved_at=resolved_at,
        tags=["ops"],
        entities={"ticket": "T-200"},
        episode_type="support_case",
        status=EpisodeStatus.RESOLVED,
    )

    result = await svc.sync_case_episode_projection(
        episode=episode,
        actor_user_id="user-1",
        event_ts=resolved_at,
        boundary_reason="episode_service_resolve",
    )

    assert result is projected
    assert projected.status == "closed"
    assert projected.boundary_end == resolved_at
    assert projected.extra_metadata["case_episode_status"] == "resolved"
