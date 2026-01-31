from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import RedisClient
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeSimulationReport:
    id: str
    organization_id: str
    session_id: str | None
    memory_id: str | None
    report: dict[str, Any]
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


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_list_simulation_reports_happy_path(monkeypatch):
    now = datetime(2026, 1, 24, tzinfo=timezone.utc)

    rows = [
        _FakeSimulationReport(
            id="r1",
            organization_id="o1",
            session_id="s1",
            memory_id=None,
            report={"iteration_num": 1, "simulation": {"confidence": 0.9}},
            created_at=now,
        )
    ]

    # Bypass DB permission lookup by seeding the permission cache.
    monkeypatch.setattr(RedisClient, "get_json", AsyncMock(return_value=["simulation:read:reports"]))
    monkeypatch.setattr(RedisClient, "set_json", AsyncMock())

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM simulation_reports" in sql:
            return _ListResult(rows)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/simulation-reports", headers=_auth_headers())

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["id"] == "r1"
    assert data[0]["organization_id"] == "o1"
    assert data[0]["session_id"] == "s1"
    assert data[0]["report"]["iteration_num"] == 1
