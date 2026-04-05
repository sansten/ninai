"""GraphQL subscriptions backed by Redis streams."""

from __future__ import annotations

import json
from typing import AsyncGenerator
from typing import Any

import strawberry
from strawberry.scalars import JSON

from app.core.redis import RedisClient
from app.middleware.tenant_context import TenantContext


@strawberry.type
class MemoryEvent:
    id: str
    event_type: str
    payload: JSON


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def memory_events(
        self,
        info: strawberry.Info,
        org_id: str | None = None,
    ) -> AsyncGenerator[MemoryEvent, None]:
        ctx = info.context
        tenant = ctx["tenant"]
        if not isinstance(tenant, TenantContext):
            raise ValueError("Invalid tenant context")

        stream_org = org_id or tenant.org_id
        if stream_org != tenant.org_id:
            raise ValueError("org_id must match authenticated tenant")

        stream = f"events:{stream_org}"
        last_id = "$"

        while True:
            items = await RedisClient.xread({stream: last_id}, count=50, block_ms=15000)
            if not items:
                continue

            for _, entries in items:
                for entry_id, fields in entries:
                    last_id = entry_id
                    raw_payload = fields.get("payload") if isinstance(fields, dict) else None
                    payload: dict[str, Any]
                    if isinstance(raw_payload, str):
                        try:
                            payload = json.loads(raw_payload)
                            if not isinstance(payload, dict):
                                payload = {"raw": raw_payload}
                        except json.JSONDecodeError:
                            payload = {"raw": raw_payload}
                    elif isinstance(raw_payload, dict):
                        payload = raw_payload
                    else:
                        payload = {}

                    yield MemoryEvent(
                        id=str(fields.get("event_id") or entry_id),
                        event_type=str(fields.get("event_type") or "unknown"),
                        payload=payload,
                    )
