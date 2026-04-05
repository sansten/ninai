from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _ScalarOneResult:
    value: object

    def scalar_one(self):
        return self.value


@dataclass
class _ScalarOneOrNoneResult:
    value: object

    def scalar_one_or_none(self):
        return self.value


@dataclass
class _OneResult:
    values: tuple

    def one_or_none(self):
        return self.values


@dataclass
class _AllResult:
    values: list[tuple]

    def all(self):
        return self.values


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_cognitive_prometheus_endpoint_available():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics/cognitive")

    assert resp.status_code == 200
    assert "text/plain" in (resp.headers.get("content-type") or "")


@pytest.mark.asyncio
async def test_cognitive_summary_endpoint_happy_path():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "count(*) AS count_1" in sql and "FROM memory_metadata" in sql and "created_at" not in sql:
            return _ScalarOneResult(100)
        if "FROM memory_metadata" in sql and "created_at" in sql:
            return _ScalarOneResult(36)
        if "FROM meta_conflict_registry" in sql:
            return _ScalarOneResult(3)
        if "max(audit_events.timestamp)" in sql:
            return _ScalarOneOrNoneResult(now - timedelta(seconds=90))
        if "FROM audit_events" in sql and "sum(CASE" in sql:
            return _OneResult((20, 18))
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics/cognitive/summary", headers=_auth_headers())
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total_memories"] == 100
    assert payload["active_conflicts"] == 3
    assert payload["memory_writes_per_second"] == 0.01
    assert payload["llm_success_rate"] == 0.9


@pytest.mark.asyncio
async def test_cognitive_agent_memory_event_metrics_endpoints():
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql and "GROUP BY agent_runs.agent_name" in sql:
            return _AllResult([("ConflictDetectionAgent", 10, 0.2, 1), ("MemoryDecayAgent", 5, 0.1, 0)])
        if "GROUP BY coalesce(memory_metadata.business_domain" in sql:
            return _AllResult([("engineering", 120), ("support", 80)])
        if "count(*) AS count_1" in sql and "FROM memory_metadata" in sql and "created_at" not in sql and "is_active" not in sql:
            return _ScalarOneResult(200)
        if "FROM memory_metadata" in sql and "created_at" in sql:
            return _ScalarOneResult(25)
        if "FROM memory_metadata" in sql and "is_active IS false" in sql:
            return _ScalarOneResult(20)
        if "FROM audit_events" in sql and "GROUP BY audit_events.event_type" in sql:
            return _AllResult([("memory.create", 30), ("agent.run", 12)])
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        headers = _auth_headers()
        agents_resp = await ac.get("/metrics/agents", headers=headers)
        memory_resp = await ac.get("/metrics/memory", headers=headers)
        events_resp = await ac.get("/metrics/events", headers=headers)
    app.dependency_overrides.clear()

    assert agents_resp.status_code == 200
    assert memory_resp.status_code == 200
    assert events_resp.status_code == 200

    agents_payload = agents_resp.json()
    assert agents_payload["total_agents"] == 2
    assert agents_payload["agents"][0]["agent_name"] == "ConflictDetectionAgent"

    memory_payload = memory_resp.json()
    assert memory_payload["total_memories"] == 200
    assert memory_payload["growth_last_24h"] == 25
    assert memory_payload["decay_rate"] == 0.1
    assert memory_payload["domain_distribution"]["engineering"] == 120

    events_payload = events_resp.json()
    assert events_payload["window"] == "1h"
    assert events_payload["events"][0]["event_type"] == "memory.create"


@pytest.mark.asyncio
async def test_metrics_endpoints_require_metrics_capability():
    transport = ASGITransport(app=app)
    headers = _auth_headers(roles=["member"])

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/metrics/cognitive/summary", headers=headers)

    assert resp.status_code == 403