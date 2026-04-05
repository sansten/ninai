from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.middleware.tenant_context import TenantContext, get_tenant_context


def _tenant(org_id: str = "org-1") -> TenantContext:
    return TenantContext(
        user_id="user-1",
        org_id=org_id,
        roles=["org_admin"],
        clearance_level=0,
    )


@pytest.mark.asyncio
async def test_sse_events_filters_by_event_type():
    async def override_tenant_context():
        return _tenant("org-1")

    app.dependency_overrides[get_tenant_context] = override_tenant_context

    batch = [
        (
            "events:org-1",
            [
                (
                    "1-0",
                    {
                        "event_type": "goal.progress",
                        "payload": json.dumps({"goal_id": "g-1", "progress": 30}),
                        "event_id": "evt-1",
                    },
                ),
                (
                    "2-0",
                    {
                        "event_type": "memory.created",
                        "payload": json.dumps({"memory_id": "m-1"}),
                        "event_id": "evt-2",
                    },
                ),
            ],
        )
    ]

    async def fake_xread(*args, **kwargs):
        return batch

    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.sse_stream.RedisClient.xread", side_effect=fake_xread):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/api/v1/sse/events?event_type=goal.progress&max_events=1")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "event: goal.progress" in resp.text
        assert "memory.created" not in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sse_goal_stream_filters_goal_id():
    async def override_tenant_context():
        return _tenant("org-42")

    app.dependency_overrides[get_tenant_context] = override_tenant_context

    batch = [
        (
            "events:org-42",
            [
                (
                    "1-0",
                    {
                        "event_type": "goal.progress",
                        "payload": json.dumps({"goal_id": "g-1", "progress": 10}),
                        "event_id": "evt-1",
                    },
                ),
                (
                    "2-0",
                    {
                        "event_type": "goal.progress",
                        "payload": json.dumps({"goal_id": "g-2", "progress": 75}),
                        "event_id": "evt-2",
                    },
                ),
            ],
        )
    ]

    async def fake_xread(*args, **kwargs):
        return batch

    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.sse_stream.RedisClient.xread", side_effect=fake_xread):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/api/v1/sse/goals/g-2?max_events=1")

        assert resp.status_code == 200
        assert "evt-2" in resp.text
        assert "evt-1" not in resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_sse_session_stream_matches_session_id_or_context_id():
    async def override_tenant_context():
        return _tenant("org-99")

    app.dependency_overrides[get_tenant_context] = override_tenant_context

    batch = [
        (
            "events:org-99",
            [
                (
                    "1-0",
                    {
                        "event_type": "thinking",
                        "payload": json.dumps({"context_id": "sess-keep", "step": "retrieving context"}),
                        "event_id": "evt-1",
                    },
                ),
                (
                    "2-0",
                    {
                        "event_type": "thinking",
                        "payload": json.dumps({"session_id": "sess-drop", "step": "forming plan"}),
                        "event_id": "evt-2",
                    },
                ),
            ],
        )
    ]

    async def fake_xread(*args, **kwargs):
        return batch

    transport = ASGITransport(app=app)
    try:
        with patch("app.api.v1.endpoints.sse_stream.RedisClient.xread", side_effect=fake_xread):
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/api/v1/sse/session/sess-keep?max_events=1")

        assert resp.status_code == 200
        assert "evt-1" in resp.text
        assert "evt-2" not in resp.text
    finally:
        app.dependency_overrides.clear()
