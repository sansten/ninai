from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


@dataclass
class _FakeAuditEvent:
    id: str
    timestamp: datetime
    event_type: str
    actor_id: str | None
    organization_id: str | None
    resource_type: str | None
    resource_id: str | None
    success: bool
    details: dict
    request_id: str | None = None


class _Scalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _Result:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


@pytest.mark.asyncio
async def test_memory_stream_requires_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/memories/stream")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_memory_stream_emits_event_and_is_event_stream():
    now = datetime(2026, 1, 22, tzinfo=timezone.utc)

    event = _FakeAuditEvent(
        id="e1",
        timestamp=now,
        event_type="memory.create",
        actor_id="u1",
        organization_id="o1",
        resource_type="memory",
        resource_id="m1",
        success=True,
        details={"foo": "bar"},
        request_id="rid",
    )

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM audit_events" in sql:
            return _Result([event])
        return _Result([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    token = create_access_token(user_id="u1", org_id="o1", roles=["member"])
    headers = {"Authorization": f"Bearer {token}"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/memories/stream?max_events=1", headers=headers)

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: memory.create" in body
    assert "data:" in body
    assert "\"resource_id\":\"m1\"" in body
