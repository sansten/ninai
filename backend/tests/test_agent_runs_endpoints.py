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
async def test_list_agent_runs_happy_path():
    now = datetime(2026, 1, 23, tzinfo=timezone.utc)

    run = _FakeAgentRun(
        id="ar1",
        organization_id="o1",
        memory_id="m1",
        agent_name="logseq_export",
        agent_version="1.0",
        inputs_hash="h" * 64,
        status="success",
        confidence=0.9,
        outputs={"ok": True},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id="t1",
        provenance=[],
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ListResult([run])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/agents/agent-runs", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == "ar1"
    assert items[0]["memory_id"] == "m1"
    assert items[0]["agent_name"] == "logseq_export"


@pytest.mark.asyncio
async def test_get_agent_run_404_when_missing():
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
        resp = await ac.get("/api/v1/agents/agent-runs/ar_missing", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_agent_run_happy_path():
    now = datetime(2026, 1, 23, tzinfo=timezone.utc)

    run = _FakeAgentRun(
        id="ar1",
        organization_id="o1",
        memory_id="m1",
        agent_name="logseq_export",
        agent_version="1.0",
        inputs_hash="h" * 64,
        status="success",
        confidence=0.9,
        outputs={"ok": True},
        warnings=["w1"],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id="t1",
        provenance=[{"source": "s1"}],
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            # detail lookup
            return _ScalarOneOrNoneResult(run)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/agents/agent-runs/ar1", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["id"] == "ar1"
    assert payload["inputs_hash"] == "h" * 64
    assert payload["outputs"] == {"ok": True}
    assert payload["warnings"] == ["w1"]
    assert payload["provenance"] == [{"source": "s1"}]
