"""Cognitive Streaming API (Feature 11) SSE endpoints.

Provides org-scoped real-time streams over Server-Sent Events backed by
Redis Streams:
- GET /sse/events
- GET /sse/goals/{goal_id}
- GET /sse/session/{session_id}
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from app.core.redis import RedisClient
from app.middleware.tenant_context import TenantContext, get_tenant_context

router = APIRouter()


def _normalize_event_types(raw: list[str] | None) -> set[str] | None:
    if not raw:
        return None
    result: set[str] = set()
    for value in raw:
        if not value:
            continue
        for token in value.split(","):
            clean = token.strip()
            if clean:
                result.add(clean)
    return result or None


def _decode_payload(fields: dict) -> dict:
    payload_raw = fields.get("payload", "{}")
    if isinstance(payload_raw, str):
        try:
            parsed = json.loads(payload_raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    if isinstance(payload_raw, dict):
        return payload_raw
    return {}


async def _stream_events(
    *,
    request: Request,
    org_id: str,
    last_event_id: str | None,
    event_types: set[str] | None,
    max_events: int | None,
    block_ms: int,
    filter_fn: Callable[[str, dict], bool] | None = None,
):
    stream_key = f"ninai:events:{org_id}:all"
    cursor = last_event_id or "$"
    sent = 0

    while True:
        if await request.is_disconnected():
            return

        rows = await RedisClient.xread(
            {stream_key: cursor},
            count=50,
            block_ms=block_ms,
        )

        emitted = False
        for _stream, messages in rows:
            for entry_id, fields in messages:
                cursor = entry_id
                event_type = str(fields.get("event_type") or "")
                payload = _decode_payload(fields)

                if event_types and event_type not in event_types:
                    continue
                if filter_fn and not filter_fn(event_type, payload):
                    continue

                event_id = str(fields.get("event_id") or entry_id)
                frame = json.dumps(
                    {
                        "event_id": event_id,
                        "stream_id": entry_id,
                        "event_type": event_type,
                        "payload": payload,
                    },
                    separators=(",", ":"),
                    default=str,
                )
                yield f"id: {event_id}\nevent: {event_type or 'event'}\ndata: {frame}\n\n".encode("utf-8")

                emitted = True
                sent += 1
                if max_events is not None and sent >= max_events:
                    return

        if not emitted:
            yield b": ping\n\n"
            await asyncio.sleep(0.05)


@router.get("/events")
async def stream_events(
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    event_type: list[str] | None = Query(default=None),
    max_events: int | None = Query(default=None, ge=1, le=1000),
):
    event_types = _normalize_event_types(event_type)
    return StreamingResponse(
        _stream_events(
            request=request,
            org_id=tenant.org_id,
            last_event_id=last_event_id,
            event_types=event_types,
            max_events=max_events,
            block_ms=5000,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/goals/{goal_id}")
async def stream_goal_events(
    goal_id: str,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    max_events: int | None = Query(default=None, ge=1, le=1000),
):
    goal_event_types = {
        "goal.progress",
        "goal.updated",
        "goal.completed",
    }

    def _goal_filter(event_type: str, payload: dict) -> bool:
        if event_type not in goal_event_types:
            return False
        return str(payload.get("goal_id") or "") == goal_id

    return StreamingResponse(
        _stream_events(
            request=request,
            org_id=tenant.org_id,
            last_event_id=last_event_id,
            event_types=goal_event_types,
            max_events=max_events,
            block_ms=5000,
            filter_fn=_goal_filter,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.get("/session/{session_id}")
async def stream_session_events(
    session_id: str,
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    max_events: int | None = Query(default=None, ge=1, le=1000),
):
    def _session_filter(_event_type: str, payload: dict) -> bool:
        payload_session_id = str(payload.get("session_id") or payload.get("context_id") or "")
        return payload_session_id == session_id

    return StreamingResponse(
        _stream_events(
            request=request,
            org_id=tenant.org_id,
            last_event_id=last_event_id,
            event_types=None,
            max_events=max_events,
            block_ms=5000,
            filter_fn=_session_filter,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
