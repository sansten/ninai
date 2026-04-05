from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeMemory:
    id: str
    organization_id: str
    is_active: bool


@dataclass
class _FakeAgentRun:
    id: str
    organization_id: str
    agent_name: str


@dataclass
class _FakeContradiction:
    id: str
    organization_id: str
    severity: str
    reason: str
    created_at: datetime
    resolved_at: datetime | None


@dataclass
class _FakeMetaConflict:
    id: str
    organization_id: str
    conflict_type: str


@dataclass
class _FakeAuditEvent:
    id: str
    event_type: str
    timestamp: datetime
    details: dict[str, Any]


@dataclass
class _FakeFeedbackRow:
    id: str
    created_at: datetime


class _Scalars:
    def __init__(self, items: list[Any]):
        self._items = items

    def all(self):
        return self._items


class _ListResult:
    def __init__(self, items: list[Any]):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_feedback_memory_records_relevance_and_returns_created(monkeypatch):
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    memory = _FakeMemory(id="m1", organization_id="o1", is_active=True)

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM memory_metadata" in sql:
            return _ScalarOneOrNoneResult(memory)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    fake_row = _FakeFeedbackRow(id="fb-1", created_at=now)

    with patch("app.api.v1.endpoints.feedback.MemoryFeedbackService.create_feedback", AsyncMock(return_value=fake_row)), \
         patch("app.api.v1.endpoints.feedback.AuditService.log_event", AsyncMock(return_value=AsyncMock(id="ev-1", timestamp=now))), \
         patch("app.api.v1.endpoints.feedback.enqueue_feedback_learning") as enqueue_mock:
        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/v1/feedback/memory/m1",
                    headers=_auth_headers(),
                    json={"relevant": True, "comment": "exact context needed"},
                )

            assert resp.status_code == 201
            body = resp.json()
            assert body["feedback_type"] == "memory"
            assert body["memory_id"] == "m1"
            assert body["relevant"] is True
            assert enqueue_mock.call_count == 1
        finally:
            app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_feedback_decision_records_feedback_for_agent_run():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    run = _FakeAgentRun(id="d1", organization_id="o1", agent_name="CognitiveGateway")
    event = AsyncMock(id="ev-d1", timestamp=now)

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(run)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.feedback.AuditService.log_event", AsyncMock(return_value=event)):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/v1/feedback/decision/d1",
                    headers=_auth_headers(),
                    json={"correct": False, "actual_outcome": "hotfix worked"},
                )

        assert resp.status_code == 201
        body = resp.json()
        assert body["feedback_type"] == "decision"
        assert body["decision_id"] == "d1"
        assert body["correct"] is False
        assert body["target_agent"] == "CognitiveGateway"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_feedback_conflict_accepts_fact_conflict():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    conflict = _FakeContradiction(
        id="c1",
        organization_id="o1",
        severity="high",
        reason="conflicting facts",
        created_at=now,
        resolved_at=None,
    )
    event = AsyncMock(id="ev-c1", timestamp=now)

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM contradictions" in sql:
            return _ScalarOneOrNoneResult(conflict)
        if "FROM meta_conflict_registry" in sql:
            return _ScalarOneOrNoneResult(None)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.feedback.AuditService.log_event", AsyncMock(return_value=event)):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/v1/feedback/conflict/c1",
                    headers=_auth_headers(),
                    json={"false_positive": True, "comment": "not actually conflicting"},
                )

        assert resp.status_code == 201
        body = resp.json()
        assert body["feedback_type"] == "conflict"
        assert body["conflict_id"] == "c1"
        assert body["false_positive"] is True
        assert body["conflict_type"] == "fact_contradiction"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_feedback_anomaly_records_genuine_flag():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    run = _FakeAgentRun(id="a1", organization_id="o1", agent_name="AnomalyDetectionAgent")
    event = AsyncMock(id="ev-a1", timestamp=now)

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(run)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.feedback.AuditService.log_event", AsyncMock(return_value=event)):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/v1/feedback/anomaly/a1",
                    headers=_auth_headers(),
                    json={"genuine": True, "comment": "confirmed incident"},
                )

        assert resp.status_code == 201
        body = resp.json()
        assert body["feedback_type"] == "anomaly"
        assert body["anomaly_id"] == "a1"
        assert body["genuine"] is True
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_feedback_stats_aggregates_by_type_and_agent():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    events = [
        _FakeAuditEvent(
            id="e1",
            event_type="feedback.memory",
            timestamp=now,
            details={"relevant": True, "target_agent": "FeedbackIntegrationAgent"},
        ),
        _FakeAuditEvent(
            id="e2",
            event_type="feedback.decision",
            timestamp=now,
            details={"correct": False, "target_agent": "CognitiveGateway"},
        ),
        _FakeAuditEvent(
            id="e3",
            event_type="feedback.conflict",
            timestamp=now,
            details={"false_positive": True, "target_agent": "MetaSupervisor"},
        ),
        _FakeAuditEvent(
            id="e4",
            event_type="feedback.anomaly",
            timestamp=now,
            details={"genuine": True, "target_agent": "AnomalyDetectionAgent"},
        ),
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM audit_events" in sql:
            return _ListResult(events)
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/feedback/stats", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_feedback"] == 4
        assert body["by_type"]["memory"] == 1
        assert body["by_type"]["decision"] == 1
        assert body["by_type"]["conflict"] == 1
        assert body["by_type"]["anomaly"] == 1
        assert body["positive_by_type"]["memory"] == 1
        assert body["positive_by_type"]["decision"] == 0
        assert body["positive_by_type"]["conflict"] == 0
        assert body["positive_by_type"]["anomaly"] == 1
        assert body["per_agent"]["FeedbackIntegrationAgent"]["total"] == 1
    finally:
        app.dependency_overrides.clear()