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


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["member"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_session_message_stream_updates_context_and_returns_sse():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    sess = _FakeCognitiveSession(
        id="s1",
        organization_id="o1",
        user_id="u1",
        agent_id=None,
        status="running",
        goal="Investigate churn",
        context_snapshot={"turns": [], "decisions": []},
        created_at=now,
        updated_at=now,
        trace_id=None,
    )

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

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

    read_result = type("ReadResult", (), {"memories": [{"id": "m1"}], "total": 1})
    decide_result = type(
        "DecideResult",
        (),
        {
            "decision": "investigate",
            "confidence": 0.83,
            "tone": "cautionary",
            "action_recommended": "Assign analyst",
        },
    )

    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.cognitive_session_protocol.CognitiveGatewayService.read", AsyncMock(return_value=read_result)), \
             patch("app.api.v1.endpoints.cognitive_session_protocol.CognitiveGatewayService.decide", AsyncMock(return_value=decide_result)), \
             patch("app.api.v1.endpoints.cognitive_session_protocol.detect_goal", return_value="Investigate churn"), \
             patch("app.api.v1.endpoints.cognitive_session_protocol.RedisClient.xadd", AsyncMock(return_value="1-0")):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/api/v1/cognitive/sessions/s1/message",
                    headers=_auth_headers(),
                    json={"message": "Investigate churn by region"},
                )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "event: result" in body
        assert "investigate" in body
        assert "event: done" in body
        assert len(sess.context_snapshot.get("decisions", [])) == 1
        assert len(sess.context_snapshot.get("turns", [])) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_session_summary_and_collections_endpoints():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    sess = _FakeCognitiveSession(
        id="s2",
        organization_id="o1",
        user_id="u1",
        agent_id=None,
        status="running",
        goal="Reduce incidents",
        context_snapshot={
            "turns": [{"role": "user", "content": "hello"}],
            "decisions": [{"decision": "monitor"}],
            "surfaced_memories": [{"id": "m2"}],
            "identified_goals": [{"goal": "Reduce incidents", "source": "message"}],
        },
        created_at=now,
        updated_at=now,
        trace_id=None,
        goal_id="g-2",
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
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            summary = await ac.get("/api/v1/cognitive/sessions/s2/summary", headers=_auth_headers())
            goals = await ac.get("/api/v1/cognitive/sessions/s2/goals", headers=_auth_headers())
            memories = await ac.get("/api/v1/cognitive/sessions/s2/memories", headers=_auth_headers())
            decisions = await ac.get("/api/v1/cognitive/sessions/s2/decisions", headers=_auth_headers())

        assert summary.status_code == 200
        assert summary.json()["turn_count"] == 1
        assert summary.json()["decision_count"] == 1
        assert goals.status_code == 200
        assert len(goals.json()["goals"]) >= 1
        assert memories.status_code == 200
        assert memories.json()["memories"][0]["id"] == "m2"
        assert decisions.status_code == 200
        assert decisions.json()["decisions"][0]["decision"] == "monitor"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_patch_and_close_session():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    sess = _FakeCognitiveSession(
        id="s3",
        organization_id="o1",
        user_id="u1",
        agent_id=None,
        status="running",
        goal="Original",
        context_snapshot={},
        created_at=now,
        updated_at=now,
        trace_id=None,
    )

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

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
    try:
        with patch("app.api.v1.endpoints.cognitive_session_protocol.RedisClient.xadd", AsyncMock(return_value="1-0")):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                patch_resp = await ac.patch(
                    "/api/v1/cognitive/sessions/s3",
                    headers=_auth_headers(),
                    json={"goal": "Updated", "status": "failed", "context_snapshot": {"k": "v"}},
                )
                close_resp = await ac.delete("/api/v1/cognitive/sessions/s3", headers=_auth_headers())

        assert patch_resp.status_code == 200
        assert patch_resp.json()["goal"] == "Updated"
        assert patch_resp.json()["status"] == "failed"
        assert close_resp.status_code == 200
        assert close_resp.json()["closed"] is True
        assert close_resp.json()["status"] == "failed"
    finally:
        app.dependency_overrides.clear()
