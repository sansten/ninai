"""Cognitive Gateway streaming endpoints.

SSE variants of decide / plan / read.
Prefixed at /cognitive/gateway/stream (mounted in router.py).

Each endpoint returns text/event-stream.  Callers iterate over the response
body and parse SSE frames:

  data: {"event": "progress", "payload": {"message": "...", "step": 1}}
  data: {"event": "result",   "payload": {...}}
  data: {"event": "done",     "payload": {...}}

  or on error:
  data: {"event": "error",    "payload": {"detail": "..."}}
  data: {"event": "done",     "payload": {}}

The generators in integrations/streaming.py provide the frame logic.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.cognitive_gateway_service import (
    CognitiveGatewayCapabilities,
    CognitiveGatewayService,
)
from integrations.streaming import stream_decide, stream_plan, stream_read

router = APIRouter()


def _get_gateway(tenant: TenantContext = Depends(require_org_admin())) -> CognitiveGatewayService:
    return CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/stream/decide
# ---------------------------------------------------------------------------

@router.post("/decide")
async def stream_gateway_decide(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> StreamingResponse:
    """Stream anomaly detection and decision as SSE.

    Request body: content (str, required), enrichment (dict), context_id (str)
    """
    content = str(payload.get("content") or "")
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content is required",
        )
    context_id = payload.get("context_id") or None

    async def _gen() -> AsyncGenerator[str, None]:
        async for frame in stream_decide(
            gateway,
            content,
            dict(payload.get("enrichment") or {}),
            context_id,
            tenant.org_id if context_id else None,
        ):
            yield frame

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/stream/plan
# ---------------------------------------------------------------------------

@router.post("/plan")
async def stream_gateway_plan(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> StreamingResponse:
    """Stream goal decomposition as SSE — one frame per step.

    Request body: goal (str, required), context (dict), context_id (str)
    """
    goal = str(payload.get("goal") or "")
    if not goal.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="goal is required",
        )
    context_id = payload.get("context_id") or None

    async def _gen() -> AsyncGenerator[str, None]:
        async for frame in stream_plan(
            gateway,
            goal,
            dict(payload.get("context") or {}),
            context_id,
            tenant.org_id if context_id else None,
        ):
            yield frame

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/stream/read
# ---------------------------------------------------------------------------

@router.post("/read")
async def stream_gateway_read(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> StreamingResponse:
    """Stream memory retrieval as SSE — one frame per memory result.

    Request body: query (str, required), memories (list), limit (int), context_id (str)
    """
    query = str(payload.get("query") or "")
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query is required",
        )
    context_id = payload.get("context_id") or None

    async def _gen() -> AsyncGenerator[str, None]:
        async for frame in stream_read(
            gateway,
            query,
            list(payload.get("memories") or []),
            int(payload.get("limit") or 10),
            context_id,
            tenant.org_id if context_id else None,
        ):
            yield frame

    return StreamingResponse(_gen(), media_type="text/event-stream")
