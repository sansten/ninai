"""Cognitive Gateway API — Phase 49.

Five-verb REST interface exposing Ninai's full intelligence stack.
Each endpoint maps to one verb: write / read / decide / plan / explain.

All endpoints are scoped by the tenant's authenticated org.
Capability gating is enforced per verb; denied verbs return 403.

Prefixed at /cognitive/gateway (mounted in router.py).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status

from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.cognitive_gateway_service import (
    CognitiveGatewayService,
    CognitiveGatewayCapabilities,
)

router = APIRouter()


def _get_gateway(tenant: TenantContext = Depends(require_org_admin())) -> CognitiveGatewayService:
    """Instantiate gateway with org-appropriate capabilities.

    In production this would be loaded from org feature flags.
    Default: full capabilities for all orgs.
    """
    return CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/write
# ---------------------------------------------------------------------------

@router.post("/write")
async def gateway_write(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Store and enrich a memory record.

    Request body:
      content  (str, required)
      title    (str, optional)
      tags     (list[str], optional)
      metadata (dict, optional)
    """
    content = str(payload.get("content") or "")
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content is required",
        )

    try:
        result = gateway.write(
            content=content,
            title=str(payload.get("title") or ""),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memory_id": result.memory_id,
        "enriched": result.enriched,
        "enrichment_summary": result.enrichment_summary,
        "tags": result.tags,
        "created_at": result.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/read
# ---------------------------------------------------------------------------

@router.post("/read")
async def gateway_read(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Retrieve and rank memories for a query.

    Request body:
      query    (str, required)
      memories (list[dict], optional — pass pre-fetched candidates)
      limit    (int, optional, default 10)
    """
    query = str(payload.get("query") or "")
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query is required",
        )

    try:
        result = gateway.read(
            query=query,
            memories=list(payload.get("memories") or []),
            limit=int(payload.get("limit") or 10),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memories": result.memories,
        "total": result.total,
        "query": result.query,
        "context_assembled": result.context_assembled,
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/decide
# ---------------------------------------------------------------------------

@router.post("/decide")
async def gateway_decide(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Run the enrichment pipeline and return a decision with confidence.

    Request body:
      content    (str, required)
      enrichment (dict, optional — pass existing enrichment to build on)
    """
    content = str(payload.get("content") or "")
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content is required",
        )

    try:
        result = gateway.decide(
            content=content,
            enrichment=dict(payload.get("enrichment") or {}),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "tone": result.tone,
        "action_recommended": result.action_recommended,
        "enrichment": result.enrichment,
        "agents_run": result.agents_run,
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/plan
# ---------------------------------------------------------------------------

@router.post("/plan")
async def gateway_plan(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Decompose a goal into ordered, actionable steps.

    Request body:
      goal    (str, required)
      context (dict, optional)
    """
    goal = str(payload.get("goal") or "")
    if not goal.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="goal is required",
        )

    try:
        result = gateway.plan(
            goal=goal,
            context=dict(payload.get("context") or {}),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "goal": result.goal,
        "steps": result.steps,
        "step_count": result.step_count,
        "blocking_step": result.blocking_step,
        "confidence": result.confidence,
    }


# ---------------------------------------------------------------------------
# GET /cognitive/gateway/explain/{memory_id}
# ---------------------------------------------------------------------------

@router.get("/explain/{memory_id}")
async def gateway_explain(
    memory_id: str = Path(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Return audit trail and explainability summary for a memory.

    In production this fetches AgentDecisionTrail rows for the memory.
    In heuristic mode returns the explainability structure with empty decisions.
    """
    try:
        result = gateway.explain(memory_id=memory_id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memory_id": result.memory_id,
        "decisions": result.decisions,
        "agents": result.agents,
        "confidence": result.confidence,
        "explainability_summary": result.explainability_summary,
    }
