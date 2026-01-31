from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeAgentRun:
    id: str
    organization_id: str
    memory_id: str
    agent_name: str
    agent_version: str
    inputs_hash: str
    status: str
    confidence: float
    outputs: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    started_at: datetime
    finished_at: datetime
    trace_id: str | None
    provenance: list[dict[str, Any]]


@dataclass
class _FakeAgentRunEvent:
    id: str
    organization_id: str
    agent_run_id: str
    memory_id: str
    event_type: str
    step_index: int
    payload: dict[str, Any]
    summary_text: str
    created_at: datetime
    trace_id: str | None = None


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


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_list_agent_run_events_happy_path():
    now = datetime(2026, 1, 23, tzinfo=timezone.utc)

    run = _FakeAgentRun(
        id="ar1",
        organization_id="o1",
        memory_id="m1",
        agent_name="a",
        agent_version="1",
        inputs_hash="h" * 64,
        status="success",
        confidence=1.0,
        outputs={},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id=None,
        provenance=[],
    )

    ev = _FakeAgentRunEvent(
        id="e1",
        organization_id="o1",
        agent_run_id="ar1",
        memory_id="m1",
        event_type="tool_call",
        step_index=1,
        payload={"tool": "search"},
        summary_text="tool_call search",
        created_at=now,
        trace_id="t1",
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(run)
        if "FROM agent_run_events" in sql:
            return _ListResult([ev])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/agents/agent-runs/ar1/events", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == "e1"
    assert items[0]["event_type"] == "tool_call"
    assert items[0]["payload"]["tool"] == "search"
    assert items[0]["summary_text"] == "tool_call search"


@pytest.mark.asyncio
async def test_create_agent_run_event_happy_path(monkeypatch):
    now = datetime(2026, 1, 23, tzinfo=timezone.utc)

    run = _FakeAgentRun(
        id="ar1",
        organization_id="o1",
        memory_id="m1",
        agent_name="a",
        agent_version="1",
        inputs_hash="h" * 64,
        status="success",
        confidence=1.0,
        outputs={},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id=None,
        provenance=[],
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(run)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)
    session.add = Mock()
    session.commit = AsyncMock()

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    import app.api.v1.endpoints.agent_runs as agent_runs_endpoints

    monkeypatch.setattr(agent_runs_endpoints, "generate_uuid", lambda: "e_new")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/v1/agents/agent-runs/ar1/events",
            headers=_auth_headers(),
            json={"event_type": "plan_step", "step_index": 0, "payload": {"text": "do thing"}, "summary_text": "plan: do thing"},
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["id"] == "e_new"
    assert payload["agent_run_id"] == "ar1"
    assert payload["memory_id"] == "m1"
    assert payload["event_type"] == "plan_step"
    assert payload["payload"]["text"] == "do thing"
    assert payload["summary_text"] == "plan: do thing"
    assert session.commit.await_count == 1


@pytest.mark.asyncio
async def test_search_agent_run_events_happy_path():
    now = datetime(2026, 1, 23, tzinfo=timezone.utc)

    ev = _FakeAgentRunEvent(
        id="e1",
        organization_id="o1",
        agent_run_id="ar1",
        memory_id="m1",
        event_type="summary",
        step_index=2,
        payload={"text": "did thing"},
        summary_text="did thing",
        created_at=now,
        trace_id=None,
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_run_events" in sql:
            return _ListResult([ev])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/v1/agents/agent-run-events/search",
            headers=_auth_headers(),
            params={"q": "thing"},
        )

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == "e1"
    assert items[0]["summary_text"] == "did thing"


@pytest.mark.asyncio
async def test_list_agent_run_events_404_when_run_missing():
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(None)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/agents/agent-runs/ar_missing/events", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 404
