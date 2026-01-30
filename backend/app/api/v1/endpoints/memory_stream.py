"""Server-Sent Events stream for memory-related events.

This is a lightweight read-only event feed intended for UI realtime updates
and simple integrations. It currently streams a curated subset of org-scoped
`audit_events` (e.g. `memory.create`, `memory.update`).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.audit import AuditEvent


router = APIRouter()


@router.get("/stream")
async def stream_memory_events(
    request: Request,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    since: datetime | None = Query(default=None),
    poll_interval_seconds: float = Query(default=1.0, ge=0.2, le=10.0),
    max_events: int | None = Query(default=None, ge=1, le=1000),
):
    """Stream memory-related events as SSE.

    Notes:
    - Org-scoped via TenantContext and `organization_id` filtering.
    - Uses DB polling (no extra infra). Intended as a "lite" stream.
    - `max_events` is primarily for testing/debugging.
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    allowed_types = {
        "memory.create",
        "memory.update",
        "knowledge.reviewed",
    }

    start_ts = since or datetime.now(timezone.utc)

    async def _event_generator():
        sent = 0
        cursor_ts = start_ts
        cursor_id = last_event_id

        while True:
            if await request.is_disconnected():
                return

            stmt = (
                select(AuditEvent)
                .where(
                    AuditEvent.organization_id == tenant.org_id,
                    AuditEvent.event_type.in_(allowed_types),
                    AuditEvent.timestamp >= cursor_ts,
                )
                .order_by(AuditEvent.timestamp.asc())
                .limit(50)
            )

            res = await db.execute(stmt)
            rows = list(res.scalars().all())

            emitted_any = False
            for ev in rows:
                ev_id = str(ev.id)
                if cursor_id and ev_id == cursor_id:
                    continue

                payload = {
                    "id": ev_id,
                    "type": ev.event_type,
                    "timestamp": ev.timestamp.isoformat() if ev.timestamp else None,
                    "actor_id": ev.actor_id,
                    "resource_type": ev.resource_type,
                    "resource_id": ev.resource_id,
                    "success": ev.success,
                    "details": ev.details or {},
                    "request_id": ev.request_id,
                }

                data = json.dumps(payload, separators=(",", ":"), default=str)
                yield f"id: {ev_id}\nevent: {ev.event_type}\ndata: {data}\n\n".encode("utf-8")

                emitted_any = True
                sent += 1
                cursor_ts = ev.timestamp or cursor_ts
                cursor_id = ev_id

                if max_events is not None and sent >= max_events:
                    return

            if not emitted_any:
                # heartbeat keeps proxies from buffering forever
                yield b": ping\n\n"

            await asyncio.sleep(poll_interval_seconds)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
