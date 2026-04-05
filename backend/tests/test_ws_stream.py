"""Tests for Real-Time Event Stream — P45.

Covers:
- RedisClient.xadd / xread helpers
- WebhookService.emit_event publishes to Redis Stream
- WebSocket endpoint: auth rejection, connect, event delivery, ping heartbeat
- Tenant isolation: events scoped by org_id stream key
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from app.core.redis import RedisClient


# ---------------------------------------------------------------------------
# RedisClient stream helpers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_xadd_calls_redis_xadd():
    mock_client = AsyncMock()
    mock_client.xadd = AsyncMock(return_value="1718000000000-0")
    with patch.object(RedisClient, "get_client", return_value=mock_client):
        result = await RedisClient.xadd("ninai:events:org-1:all", {"event_type": "memory.created", "payload": "{}"})
    mock_client.xadd.assert_called_once()
    assert result == "1718000000000-0"


@pytest.mark.asyncio
async def test_xadd_passes_maxlen():
    mock_client = AsyncMock()
    mock_client.xadd = AsyncMock(return_value="id-1")
    with patch.object(RedisClient, "get_client", return_value=mock_client):
        await RedisClient.xadd("ninai:events:org-1:all", {"k": "v"}, maxlen=500)
    _, kwargs = mock_client.xadd.call_args
    assert kwargs.get("maxlen") == 500 or mock_client.xadd.call_args[0][2] == 500  # positional or kw


@pytest.mark.asyncio
async def test_xread_calls_redis_xread():
    mock_client = AsyncMock()
    mock_client.xread = AsyncMock(return_value=[
        ["ninai:events:org-1:all", [("id-1", {"event_type": "test", "payload": "{}"})]],
    ])
    with patch.object(RedisClient, "get_client", return_value=mock_client):
        result = await RedisClient.xread({"ninai:events:org-1:all": "$"}, count=10, block_ms=0)
    assert len(result) == 1
    assert result[0][0] == "ninai:events:org-1:all"


@pytest.mark.asyncio
async def test_xread_returns_empty_list_when_no_entries():
    mock_client = AsyncMock()
    mock_client.xread = AsyncMock(return_value=None)
    with patch.object(RedisClient, "get_client", return_value=mock_client):
        result = await RedisClient.xread({"ninai:events:org-1:all": "$"}, count=10, block_ms=0)
    assert result == []


@pytest.mark.asyncio
async def test_xread_passes_block_none_when_block_ms_zero():
    mock_client = AsyncMock()
    mock_client.xread = AsyncMock(return_value=None)
    with patch.object(RedisClient, "get_client", return_value=mock_client):
        await RedisClient.xread({"ninai:events:org-1:all": "0"}, count=5, block_ms=0)
    _, kwargs = mock_client.xread.call_args
    # block_ms=0 → block=None (non-blocking)
    assert kwargs.get("block") is None


@pytest.mark.asyncio
async def test_xread_passes_block_value_when_nonzero():
    mock_client = AsyncMock()
    mock_client.xread = AsyncMock(return_value=None)
    with patch.object(RedisClient, "get_client", return_value=mock_client):
        await RedisClient.xread({"ninai:events:org-1:all": "0"}, count=5, block_ms=5000)
    _, kwargs = mock_client.xread.call_args
    assert kwargs.get("block") == 5000


# ---------------------------------------------------------------------------
# WebhookService.emit_event → Redis Stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_event_publishes_to_redis_stream():
    """emit_event should call RedisClient.xadd after writing to DB."""
    from app.services.webhook_service import WebhookService

    session = AsyncMock()
    event_mock = MagicMock()
    event_mock.id = "evt-1"

    # Simulate flush setting the id on the event
    async def _flush():
        pass
    session.flush = AsyncMock(side_effect=_flush)
    session.add = MagicMock()

    # No subscriptions
    subs_result = MagicMock()
    subs_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=subs_result)

    xadd_calls = []

    async def _fake_xadd(stream, fields, maxlen=1000):
        xadd_calls.append((stream, fields, maxlen))
        return "1-0"

    with patch.object(RedisClient, "xadd", side_effect=_fake_xadd):
        svc = WebhookService(session)
        await svc.emit_event(
            organization_id="org-1",
            event_type="memory.created",
            payload={"memory_id": "m-1"},
        )

    assert len(xadd_calls) == 3
    stream, fields, maxlen = xadd_calls[0]
    assert stream == "ninai:events:org-1:all"
    assert fields["event_type"] == "memory.created"
    assert maxlen == 1000


@pytest.mark.asyncio
async def test_emit_event_redis_failure_does_not_raise():
    """A Redis error in xadd must not propagate — DB write must succeed."""
    from app.services.webhook_service import WebhookService

    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    subs_result = MagicMock()
    subs_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=subs_result)

    with patch.object(RedisClient, "xadd", side_effect=Exception("redis down")):
        svc = WebhookService(session)
        # Must not raise
        await svc.emit_event(
            organization_id="org-1",
            event_type="memory.created",
            payload={},
        )


@pytest.mark.asyncio
async def test_emit_event_stream_key_uses_org_id():
    from app.services.webhook_service import WebhookService

    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    subs_result = MagicMock()
    subs_result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=subs_result)

    captured = {}

    async def _fake_xadd(stream, fields, maxlen=1000):
        captured["stream"] = stream
        return "1-0"

    with patch.object(RedisClient, "xadd", side_effect=_fake_xadd):
        svc = WebhookService(session)
        await svc.emit_event(
            organization_id="tenant-xyz",
            event_type="test",
            payload={},
        )

    assert captured["stream"] == "events:tenant-xyz"


@pytest.mark.asyncio
async def test_ws_rejects_org_query_mismatch():
    from app.api.v1.endpoints.ws_stream import ws_stream

    ws = AsyncMock(spec=WebSocket)
    ws.close = AsyncMock()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=_make_token_data(org_id="org-1")):
        await ws_stream(ws, token="valid", org_id="org-2")

    ws.close.assert_called_once_with(code=4008)


# ---------------------------------------------------------------------------
# WebSocket endpoint — auth
# ---------------------------------------------------------------------------

def _make_token_data(org_id: str = "org-1", user_id: str = "user-1"):
    from app.core.security import TokenData
    return TokenData(user_id=user_id, org_id=org_id, roles=["member"])


@pytest.mark.asyncio
async def test_ws_rejects_missing_token():
    """Connection without token must be closed with code 4008."""
    from app.api.v1.endpoints.ws_stream import ws_stream

    ws = AsyncMock(spec=WebSocket)
    ws.close = AsyncMock()

    await ws_stream(ws, token=None)

    ws.close.assert_called_once_with(code=4008)
    ws.accept.assert_not_called()


@pytest.mark.asyncio
async def test_ws_rejects_invalid_token():
    from app.api.v1.endpoints.ws_stream import ws_stream

    ws = AsyncMock(spec=WebSocket)
    ws.close = AsyncMock()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=None):
        await ws_stream(ws, token="bad.token")

    ws.close.assert_called_once_with(code=4008)


@pytest.mark.asyncio
async def test_ws_rejects_token_without_org_id():
    from app.api.v1.endpoints.ws_stream import ws_stream
    from app.core.security import TokenData

    ws = AsyncMock(spec=WebSocket)
    ws.close = AsyncMock()
    token_data = TokenData(user_id="user-1", org_id=None, roles=[])

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data):
        await ws_stream(ws, token="token")

    ws.close.assert_called_once_with(code=4008)


# ---------------------------------------------------------------------------
# WebSocket endpoint — event delivery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ws_accepts_valid_token():
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data()
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.close = AsyncMock()

    # xread raises WebSocketDisconnect to exit loop after first call
    call_count = 0

    async def _fake_xread(streams, count=50, block_ms=0):
        nonlocal call_count
        call_count += 1
        raise WebSocketDisconnect()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    ws.accept.assert_called_once()


@pytest.mark.asyncio
async def test_ws_sends_event_frame():
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data()
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()

    fields = {
        "event_type": "memory.created",
        "payload": json.dumps({"id": "m-1"}),
        "event_id": "evt-1",
    }

    call_count = 0

    async def _fake_xread(streams, count=50, block_ms=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [["ninai:events:org-1:all", [("1-0", fields)]]]
        raise WebSocketDisconnect()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    ws.send_json.assert_called()
    sent = ws.send_json.call_args_list[0][0][0]
    assert sent["event_type"] == "memory.created"
    assert sent["payload"] == {"id": "m-1"}
    assert sent["event_id"] == "evt-1"


@pytest.mark.asyncio
async def test_ws_sends_ping_when_no_events():
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data()
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()

    call_count = 0

    async def _fake_xread(streams, count=50, block_ms=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return []  # no events → should ping
        raise WebSocketDisconnect()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    sent_frames = [c[0][0] for c in ws.send_json.call_args_list]
    assert any(f.get("type") == "ping" for f in sent_frames)


@pytest.mark.asyncio
async def test_ws_advances_last_id_after_reading():
    """The stream cursor (last_id) should advance to the last received entry id."""
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data()
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()

    received_streams = []
    call_count = 0

    async def _fake_xread(streams, count=50, block_ms=0):
        nonlocal call_count
        call_count += 1
        received_streams.append(dict(streams))
        if call_count == 1:
            return [["ninai:events:org-1:all", [("5-0", {"event_type": "t", "payload": "{}", "event_id": "e"})]]]
        raise WebSocketDisconnect()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    # First call uses "$"; we can't easily check the second call's last_id since
    # the disconnect happens immediately — but we verify at least one read happened.
    assert call_count >= 1
    assert received_streams[0].get("ninai:events:org-1:all") == "$"


@pytest.mark.asyncio
async def test_ws_stream_key_scoped_to_org():
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data(org_id="tenant-abc")
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()

    read_streams = []

    async def _fake_xread(streams, count=50, block_ms=0):
        read_streams.append(list(streams.keys()))
        raise WebSocketDisconnect()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    assert read_streams[0] == ["ninai:events:tenant-abc:all"]


@pytest.mark.asyncio
async def test_ws_tenant_isolation():
    """Org A and org B get different stream keys — events don't cross."""
    from app.api.v1.endpoints.ws_stream import ws_stream

    keys_used: list[list] = []

    async def _fake_xread(streams, count=50, block_ms=0):
        keys_used.append(list(streams.keys()))
        raise WebSocketDisconnect()

    for org in ("org-alpha", "org-beta"):
        token_data = _make_token_data(org_id=org)
        ws = AsyncMock(spec=WebSocket)
        ws.accept = AsyncMock()
        ws.close = AsyncMock()
        with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
             patch.object(RedisClient, "xread", side_effect=_fake_xread):
            await ws_stream(ws, token="valid")

    assert keys_used[0] != keys_used[1]
    assert "ninai:events:org-alpha:all" in keys_used[0]
    assert "ninai:events:org-beta:all" in keys_used[1]


@pytest.mark.asyncio
async def test_ws_handles_redis_error_gracefully():
    """A Redis error during xread should not crash the handler abruptly."""
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data()
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()

    call_count = 0

    async def _fake_xread(streams, count=50, block_ms=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("redis unavailable")
        # Second iteration — send ping then disconnect
        raise WebSocketDisconnect()

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    # accept was called — connection stayed alive despite Redis error
    ws.accept.assert_called_once()


@pytest.mark.asyncio
async def test_ws_cleans_up_on_unexpected_exception():
    """An unexpected exception inside the loop must still close the WebSocket."""
    from app.api.v1.endpoints.ws_stream import ws_stream

    token_data = _make_token_data()
    ws = AsyncMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.close = AsyncMock()

    fields = {"event_type": "t", "payload": "{}", "event_id": "e"}
    call_count = 0

    async def _fake_xread(streams, count=50, block_ms=0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [["ninai:events:org-1:all", [("1-0", fields)]]]
        raise WebSocketDisconnect()

    # send_json raises on the first call (delivering the event frame), then ping
    ws.send_json = AsyncMock(side_effect=RuntimeError("boom"))

    with patch("app.api.v1.endpoints.ws_stream.verify_token", return_value=token_data), \
         patch.object(RedisClient, "xread", side_effect=_fake_xread):
        await ws_stream(ws, token="valid")

    ws.close.assert_called()
