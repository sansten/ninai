from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints import benchmarks
from app.core.database import get_db
from app.main import app


@dataclass
class _FakeUser:
    id: str
    is_admin: bool = True


@dataclass
class _FakeBenchmarkRun:
    id: str
    run_at: datetime
    mode: str
    strategy: str
    dataset: str
    vllm_model: str | None
    duration_seconds: float
    composite_score: float
    results: list[dict[str, Any]]


class _Scalars:
    def __init__(self, items: list[Any]):
        self._items = items

    def all(self):
        return self._items


class _ResultList:
    def __init__(self, items: list[Any]):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


class _ResultOne:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


@pytest.mark.asyncio
async def test_create_benchmark_run():
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()

    async def override_get_db():
        yield session

    async def override_admin_user():
        return _FakeUser(id="u1", is_admin=True)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[benchmarks.get_admin_user_simple] = override_admin_user

    payload = {
        "run_at": "2026-04-05T10:00:00+00:00",
        "mode": "unit",
        "strategy": "heuristic",
        "dataset": "kaggle",
        "vllm_model": None,
        "duration_seconds": 3.2,
        "composite_score": 0.76,
        "results": [{"benchmark": "goal", "accuracy": 0.8}],
    }

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/v1/admin/benchmarks", json=payload)

        assert resp.status_code == 201
        body = resp.json()
        assert body["strategy"] == "heuristic"
        assert body["dataset"] == "kaggle"
        assert body["composite_score"] == 0.76
        assert isinstance(body["id"], str)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_and_latest_benchmark_runs():
    run1 = _FakeBenchmarkRun(
        id="r1",
        run_at=datetime(2026, 4, 5, 10, 0, tzinfo=timezone.utc),
        mode="unit",
        strategy="heuristic",
        dataset="kaggle",
        vllm_model=None,
        duration_seconds=4.0,
        composite_score=0.72,
        results=[{"benchmark": "goal", "accuracy": 0.74}],
    )
    run2 = _FakeBenchmarkRun(
        id="r2",
        run_at=datetime(2026, 4, 5, 10, 5, tzinfo=timezone.utc),
        mode="unit",
        strategy="llm",
        dataset="kaggle",
        vllm_model="qwen2.5:7b",
        duration_seconds=5.5,
        composite_score=0.81,
        results=[{"benchmark": "goal", "accuracy": 0.84}],
    )

    execute_calls = 0
    session = AsyncMock(spec=AsyncSession)

    async def _execute(*args, **kwargs):
        nonlocal execute_calls
        execute_calls += 1
        if execute_calls == 1:
            return _ResultList([run2, run1])
        return _ResultOne(run2)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    async def override_admin_user():
        return _FakeUser(id="u1", is_admin=True)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[benchmarks.get_admin_user_simple] = override_admin_user

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            list_resp = await ac.get("/api/v1/admin/benchmarks?limit=10")
            latest_resp = await ac.get("/api/v1/admin/benchmarks/latest")

        assert list_resp.status_code == 200
        data = list_resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "r2"

        assert latest_resp.status_code == 200
        latest = latest_resp.json()
        assert latest["id"] == "r2"
        assert latest["strategy"] == "llm"
    finally:
        app.dependency_overrides.clear()

