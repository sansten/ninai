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
class _FakeCognitiveSession:
    id: str
    organization_id: str
    user_id: str
    agent_id: str | None
    status: str
    goal: str
    context_snapshot: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    trace_id: str | None
    goal_id: str | None = None


@dataclass
class _FakeIteration:
    id: str
    session_id: str
    iteration_num: int
    plan_json: dict[str, Any]
    execution_json: dict[str, Any]
    critique_json: dict[str, Any]
    evaluation: str
    started_at: datetime
    finished_at: datetime
    metrics: dict[str, Any]


@dataclass
class _FakeToolLog:
    id: str
    session_id: str
    iteration_id: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output_summary: dict[str, Any] | None = None
    status: str = "success"
    denial_reason: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


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
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_evaluation_report_generates_when_missing(monkeypatch):
    now = datetime(2026, 1, 24, tzinfo=timezone.utc)

    sess = _FakeCognitiveSession(
        id="s1",
        organization_id="o1",
        user_id="u1",
        agent_id=None,
        status="succeeded",
        goal="g",
        context_snapshot={},
        created_at=now,
        updated_at=now,
        trace_id=None,
    )

    iterations = [
        _FakeIteration(
            id="it1",
            session_id="s1",
            iteration_num=1,
            plan_json={},
            execution_json={},
            critique_json={},
            evaluation="pass",
            started_at=now,
            finished_at=now,
            metrics={"confidence": 0.8, "evidence_memory_ids": ["m1"]},
        )
    ]

    tool_logs = [_FakeToolLog(id="t1", session_id="s1", status="success")]

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.flush = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM cognitive_sessions" in sql:
            return _ScalarOneOrNoneResult(sess)
        if "FROM evaluation_reports" in sql:
            return _ScalarOneOrNoneResult(None)
        if "FROM cognitive_iterations" in sql:
            return _ListResult(iterations)
        if "FROM tool_call_logs" in sql:
            return _ListResult(tool_logs)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/cognitive/sessions/s1/report", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "s1"
    assert data["final_decision"] == "pass"
    assert data["report"]["iteration_count"] == 1
    assert data["report"]["evidence_memory_ids"] == ["m1"]


@pytest.mark.asyncio
async def test_get_session_forbidden_for_other_user():
    now = datetime(2026, 1, 24, tzinfo=timezone.utc)

    sess = _FakeCognitiveSession(
        id="s1",
        organization_id="o1",
        user_id="u_owner",
        agent_id=None,
        status="running",
        goal="g",
        context_snapshot={},
        created_at=now,
        updated_at=now,
        trace_id=None,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM cognitive_sessions" in sql:
            return _ScalarOneOrNoneResult(sess)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/cognitive/sessions/s1", headers=_auth_headers(user_id="u_other"))

    app.dependency_overrides.clear()

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_list_tool_calls_session_only_happy_path():
    now = datetime(2026, 1, 24, tzinfo=timezone.utc)

    sess = _FakeCognitiveSession(
        id="s1",
        organization_id="o1",
        user_id="u1",
        agent_id=None,
        status="running",
        goal="g",
        context_snapshot={},
        created_at=now,
        updated_at=now,
        trace_id=None,
    )

    tool_logs = [
        _FakeToolLog(
            id="t1",
            session_id="s1",
            iteration_id="it1",
            tool_name="memory.search",
            tool_input={"query": "q"},
            tool_output_summary={"count": 1},
            status="success",
            denial_reason=None,
            started_at=now,
            finished_at=now,
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM cognitive_sessions" in sql:
            return _ScalarOneOrNoneResult(sess)
        if "FROM tool_call_logs" in sql:
            return _ListResult(tool_logs)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/cognitive/sessions/s1/tool-calls", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == "t1"
    assert items[0]["session_id"] == "s1"
    assert items[0]["iteration_id"] == "it1"
    assert items[0]["tool_name"] == "memory.search"
    assert items[0]["status"] == "success"
