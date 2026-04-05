from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

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
    title: str | None
    content_preview: str
    tags: list[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


@dataclass
class _FakeAuditEvent:
    id: str
    event_type: str
    timestamp: datetime
    resource_type: str | None
    resource_id: str | None
    success: bool
    details: dict[str, Any]
    changes: dict[str, Any] | None = None


@dataclass
class _FakeGoal:
    id: str
    title: str
    status: str
    priority: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


@dataclass
class _FakeContradiction:
    id: str
    severity: str
    reason: str
    created_at: datetime
    resolved_at: datetime | None


@dataclass
class _FakeMetaConflict:
    id: str
    conflict_type: str
    status: str
    resource_type: str
    resource_id: str
    created_at: datetime
    resolved_at: datetime | None


@dataclass
class _FakeEdge:
    id: str
    source_id: str
    target_id: str
    edge_type: str
    agent_name: str
    created_at: datetime
    edge_metadata: dict[str, Any]


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


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_topic_diff_returns_added_updated_invalidated_and_conflicts():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)

    memories = [
        _FakeMemory(
            id="m1",
            organization_id="o1",
            title="Customer churn spike",
            content_preview="Churn increased after billing issue",
            tags=["churn", "billing"],
            created_at=now,
            updated_at=now,
        )
    ]
    events = [
        _FakeAuditEvent(
            id="e1",
            event_type="memory.created",
            timestamp=now,
            resource_type="memory",
            resource_id="m1",
            success=True,
            details={"topic": "customer churn"},
        ),
        _FakeAuditEvent(
            id="e2",
            event_type="memory.updated",
            timestamp=now,
            resource_type="memory",
            resource_id="m1",
            success=True,
            details={"field": "confidence"},
        ),
        _FakeAuditEvent(
            id="e3",
            event_type="memory.deleted",
            timestamp=now,
            resource_type="memory",
            resource_id="m1",
            success=True,
            details={"reason": "invalidated"},
        ),
    ]
    contradictions = [
        _FakeContradiction(
            id="c1",
            severity="high",
            reason="conflicting churn metrics",
            created_at=now,
            resolved_at=None,
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM memory_metadata" in sql:
            return _ListResult(memories)
        if "FROM audit_events" in sql:
            return _ListResult(events)
        if "FROM contradictions" in sql:
            return _ListResult(contradictions)
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/cognitive/diff?topic=customer+churn", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["topic"] == "customer churn"
        assert body["changed"] is True
        assert body["payload"]["added_memories"] == ["m1"]
        assert body["payload"]["updated_memories"] == ["m1"]
        assert body["payload"]["invalidated_memories"] == ["m1"]
        assert body["payload"]["new_conflicts"][0]["id"] == "c1"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_memory_diff_returns_timeline_with_audit_and_provenance():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)

    events = [
        _FakeAuditEvent(
            id="e1",
            event_type="memory.updated",
            timestamp=now,
            resource_type="memory",
            resource_id="m1",
            success=True,
            details={"field": "title"},
        )
    ]
    edges = [
        _FakeEdge(
            id="p1",
            source_id="jira:INC-1",
            target_id="m1",
            edge_type="ingest",
            agent_name="IngestService",
            created_at=now,
            edge_metadata={"source": "jira"},
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM audit_events" in sql:
            return _ListResult(events)
        if "FROM provenance_edges" in sql:
            return _ListResult(edges)
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/cognitive/diff/memory/m1", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["memory_id"] == "m1"
        assert body["change_count"] == 2
        kinds = [item["kind"] for item in body["timeline"]]
        assert "audit" in kinds
        assert "provenance" in kinds
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_goal_diff_returns_goal_status_changes():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    goals = [
        _FakeGoal(
            id="g1",
            title="Reduce churn by 10%",
            status="in_progress",
            priority=2,
            created_at=now,
            updated_at=now,
            completed_at=None,
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM goals" in sql:
            return _ListResult(goals)
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/cognitive/diff/goals", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["changes"][0]["goal_id"] == "g1"
        assert body["changes"][0]["status"] == "in_progress"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_conflict_diff_returns_new_and_resolved_conflicts():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    contradictions = [
        _FakeContradiction(
            id="c-new",
            severity="high",
            reason="new conflict",
            created_at=now,
            resolved_at=None,
        ),
        _FakeContradiction(
            id="c-resolved",
            severity="low",
            reason="resolved conflict",
            created_at=now,
            resolved_at=now,
        ),
    ]
    meta = [
        _FakeMetaConflict(
            id="m-new",
            conflict_type="duplicate_candidate",
            status="open",
            resource_type="memory",
            resource_id="m1",
            created_at=now,
            resolved_at=None,
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM contradictions" in sql:
            return _ListResult(contradictions)
        if "FROM meta_conflict_registry" in sql:
            return _ListResult(meta)
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/cognitive/diff/conflicts", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["new_conflicts"]) == 2
        assert len(body["resolved_conflicts"]) == 1
        assert len(body["new_meta_conflicts"]) == 1
        assert body["new_meta_conflicts"][0]["id"] == "m-new"
    finally:
        app.dependency_overrides.clear()