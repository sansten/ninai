"""Real-Time Event Stream — P45.

WebSocket endpoint that streams per-tenant events from Redis Streams to
connected clients.  Auth is carried as a query-param JWT token (standard
WebSocket pattern — browsers cannot set headers on WS upgrades).

Route:
  GET /ws/stream?token=<jwt>
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.core.redis import RedisClient
from app.core.security import verify_token

logger = logging.getLogger(__name__)

router = APIRouter()

_PING_INTERVAL = 30        # seconds between heartbeat pings
_XREAD_BLOCK_MS = 5_000   # ms to block on XREAD; keeps the loop responsive
_XREAD_COUNT = 50          # max events per XREAD call


@router.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
    org_id: Optional[str] = None,
):
    """Stream org-scoped events over WebSocket.

    Clients connect with a valid JWT access token as ?token=<jwt>.
    Events are delivered as JSON frames:
      {"event_type": "...", "payload": {...}, "event_id": "..."}
    A heartbeat ping is sent every 30 s when no events arrive:
      {"type": "ping"}
    The connection is closed with code 4008 on auth failure.
    """
    # ---- Auth ----------------------------------------------------------------
    if not token:
        await websocket.close(code=4008)
        return

    token_data = verify_token(token)
    if token_data is None or not token_data.org_id:
        await websocket.close(code=4008)
        return

    token_org_id = str(token_data.org_id)
    if org_id and org_id != token_org_id:
        await websocket.close(code=4008)
        return

    org_id = token_org_id
    stream_key = f"ninai:events:{org_id}:all"

    await websocket.accept()
    logger.info("WS connect", extra={"org_id": org_id, "user_id": token_data.user_id})

    # Read only events that arrive after connect (last_id="$" means "new only").
    last_id = "$"

    try:
        while True:
            # Fetch new events from Redis Stream (blocks up to _XREAD_BLOCK_MS ms).
            try:
                entries = await RedisClient.xread(
                    {stream_key: last_id},
                    count=_XREAD_COUNT,
                    block_ms=_XREAD_BLOCK_MS,
                )
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("Redis xread error", extra={"org_id": org_id})
                entries = []

            if entries:
                # entries = [(stream_key, [(entry_id, fields), ...])]
                for _stream, messages in entries:
                    for entry_id, fields in messages:
                        last_id = entry_id
                        try:
                            payload_raw = fields.get("payload", "{}")
                            payload = (
                                json.loads(payload_raw)
                                if isinstance(payload_raw, str)
                                else payload_raw
                            )
                            frame = {
                                "event_type": fields.get("event_type", ""),
                                "payload": payload,
                                "event_id": fields.get("event_id", ""),
                                "stream_id": entry_id,
                            }
                            await websocket.send_json(frame)
                        except Exception:
                            logger.exception(
                                "Failed to send WS frame",
                                extra={"org_id": org_id, "entry_id": entry_id},
                            )
            else:
                # No events during the block window — send heartbeat ping.
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("WS disconnect", extra={"org_id": org_id})
    except Exception:
        logger.exception("WS stream error", extra={"org_id": org_id})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
