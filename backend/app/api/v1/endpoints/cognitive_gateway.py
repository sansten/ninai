"""Cognitive Gateway API — Phase 49.

Five-verb REST interface exposing Ninai's full intelligence stack.
Each endpoint maps to one verb: write / read / decide / plan / explain.

All endpoints are scoped by the tenant's authenticated org.
Capability gating is enforced per verb; denied verbs return 403.

Prefixed at /cognitive/gateway (mounted in router.py).

context_id (optional, any string): pass the same context_id across multiple
calls to chain them — prior decisions, steps, and enrichment accumulate in
Redis (1h TTL) and inform each subsequent call.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.core.requester_context import RequesterContext
from app.middleware.tenant_context import TenantContext, get_requester_context, require_org_admin
from app.services.usage_service import UsageService
from app.services.cognitive_gateway_service import (
    CognitiveGatewayCapabilities,
    CognitiveGatewayService,
    GatewayContextSession,
    load_gateway_context,
    save_gateway_context,
)

router = APIRouter()


def _get_gateway(tenant: TenantContext = Depends(require_org_admin())) -> CognitiveGatewayService:
    """Instantiate gateway with org-appropriate capabilities.

    In production this would be loaded from org feature flags.
    Default: full capabilities for all orgs.
    """
    return CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())


async def _context_working_set_summary(context_id: str | None, org_id: str) -> dict[str, Any] | None:
    if not context_id:
        return None
    ctx = await load_gateway_context(context_id, org_id)
    if ctx is None:
        return None
    return dict(ctx.working_set_summary or {})


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/write
# ---------------------------------------------------------------------------

@router.post("/write")
async def gateway_write(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    requester: RequesterContext = Depends(get_requester_context),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Store and enrich a memory record.

    Request body:
      content    (str, required)
      title      (str, optional)
      tags       (list[str], optional)
      metadata   (dict, optional)
      context_id (str, optional) — chain with prior gateway calls
    """
    content = str(payload.get("content") or "")
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content is required",
        )
    context_id = payload.get("context_id") or None

    try:
        result = await gateway.write(
            content=content,
            title=str(payload.get("title") or ""),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            context_id=context_id,
            org_id=tenant.org_id if context_id else None,
            requester=requester,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memory_id": result.memory_id,
        "enriched": result.enriched,
        "enrichment_summary": result.enrichment_summary,
        "tags": result.tags,
        "created_at": result.created_at.isoformat(),
        "context_id": context_id,
        "working_set_summary": await _context_working_set_summary(context_id, tenant.org_id),
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/read
# ---------------------------------------------------------------------------

@router.post("/read")
async def gateway_read(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    requester: RequesterContext = Depends(get_requester_context),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Retrieve and rank memories for a query.

    Request body:
      query      (str, required)
      memories   (list[dict], optional — pass pre-fetched candidates)
      limit      (int, optional, default 10)
      context_id (str, optional) — merge prior memories from chain
    """
    query = str(payload.get("query") or "")
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="query is required",
        )
    context_id = payload.get("context_id") or None

    filter_tags = payload.get("filter_tags") or None
    if filter_tags and not isinstance(filter_tags, list):
        filter_tags = None

    try:
        result = await gateway.read(
            query=query,
            memories=list(payload.get("memories") or []),
            limit=int(payload.get("limit") or 10),
            filter_tags=filter_tags,
            context_id=context_id,
            org_id=tenant.org_id if context_id else None,
            requester=requester,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memories": result.memories,
        "total": result.total,
        "query": result.query,
        "context_assembled": result.context_assembled,
        "retrieval_confidence": result.retrieval_confidence,
        "corrected_by": result.corrected_by,
        "reasoning_steps": result.reasoning_steps,
        "compression_ratio": result.compression_ratio,
        "information_density": result.information_density,
        "context_id": context_id,
        "working_set_summary": await _context_working_set_summary(context_id, tenant.org_id),
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/decide
# ---------------------------------------------------------------------------

@router.post("/decide")
async def gateway_decide(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    requester: RequesterContext = Depends(get_requester_context),
    db: AsyncSession = Depends(get_db),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Run the enrichment pipeline and return a decision with confidence.

    Request body:
      content    (str, required)
      enrichment (dict, optional — pass existing enrichment to build on)
      context_id (str, optional) — merge accumulated enrichment from chain
    """
    content = str(payload.get("content") or "")
    if not content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content is required",
        )
    context_id = payload.get("context_id") or None
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    try:
        result = await gateway.decide(
            content=content,
            enrichment=dict(payload.get("enrichment") or {}),
            context_id=context_id,
            org_id=tenant.org_id if context_id else None,
            requester=requester,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    usage = UsageService(db, tenant.org_id)
    await usage.increment(metric="cognitive_gateway_calls", value=1)
    await db.commit()

    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "tone": result.tone,
        "action_recommended": result.action_recommended,
        "enrichment": result.enrichment,
        "agents_run": result.agents_run,
        "debate_transcript": result.debate_transcript,
        "fingerprint_alerts": result.fingerprint_alerts,
        "context_id": context_id,
        "working_set_summary": await _context_working_set_summary(context_id, tenant.org_id),
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/plan
# ---------------------------------------------------------------------------

@router.post("/plan")
async def gateway_plan(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    requester: RequesterContext = Depends(get_requester_context),
    db: AsyncSession = Depends(get_db),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Decompose a goal into ordered, actionable steps.

    Request body:
      goal       (str, required)
      context    (dict, optional)
      context_id (str, optional) — carry forward prior decision into context
    """
    goal = str(payload.get("goal") or "")
    if not goal.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="goal is required",
        )
    context_id = payload.get("context_id") or None
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    try:
        result = await gateway.plan(
            goal=goal,
            context=dict(payload.get("context") or {}),
            context_id=context_id,
            org_id=tenant.org_id if context_id else None,
            requester=requester,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    usage = UsageService(db, tenant.org_id)
    await usage.increment(metric="cognitive_gateway_calls", value=1)
    await db.commit()

    return {
        "goal": result.goal,
        "steps": result.steps,
        "step_count": result.step_count,
        "blocking_step": result.blocking_step,
        "confidence": result.confidence,
        "context_id": context_id,
        "working_set_summary": await _context_working_set_summary(context_id, tenant.org_id),
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
        result = await gateway.explain(memory_id=memory_id)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memory_id": result.memory_id,
        "decisions": result.decisions,
        "agents": result.agents,
        "confidence": result.confidence,
        "explainability_summary": result.explainability_summary,
    }


# ---------------------------------------------------------------------------
# GET /cognitive/gateway/context/{context_id}
# ---------------------------------------------------------------------------

@router.get("/context/{context_id}")
async def get_gateway_context(
    context_id: str = Path(...),
    tenant: TenantContext = Depends(require_org_admin()),
) -> dict[str, Any]:
    """Inspect the accumulated state of a gateway context chain.

    Returns the context session including prior decision, steps, memories,
    and enrichment accumulated across calls sharing this context_id.
    Returns 404 if the context has expired (> 1h since last use) or never existed.
    """
    import dataclasses

    ctx = await load_gateway_context(context_id, tenant.org_id)
    if ctx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Context not found or expired.",
        )
    return dataclasses.asdict(ctx)


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/context  (create a new context_id)
# ---------------------------------------------------------------------------

@router.post("/context")
async def create_gateway_context(
    tenant: TenantContext = Depends(require_org_admin()),
) -> dict[str, Any]:
    """Create a new gateway context_id for chaining multiple verb calls.

    Returns a fresh context_id that callers pass to write/read/decide/plan/explain
    to accumulate state across calls.
    """
    context_id = str(uuid.uuid4())
    ctx = GatewayContextSession(context_id=context_id, org_id=tenant.org_id)
    await save_gateway_context(ctx)
    return {
        "context_id": context_id,
        "org_id": tenant.org_id,
        "ttl_seconds": 3600,
        "working_set_summary": ctx.working_set_summary,
    }


# ---------------------------------------------------------------------------
# GET /cognitive/gateway/state
# ---------------------------------------------------------------------------

@router.get("/state")
async def get_cognitive_state(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Return the current SystemCognitionState for this org.

    Shows what the Cognitive OS is currently attending to, cognitive load,
    unresolved anomaly count, and the time of the last heartbeat.
    Updated every 5 minutes by the autonomous heartbeat task.
    """
    from app.services.system_cognition_state import SystemCognitionStateService

    svc = SystemCognitionStateService(db)
    state = await svc.get(tenant.org_id)
    if state is None:
        return {
            "status": "no_heartbeat_data",
            "message": "No heartbeat has run yet for this org.",
        }
    return state.to_dict()


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/answer
# ---------------------------------------------------------------------------

@router.post("/answer")
async def gateway_answer(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Generate an answer to a question using pre-fetched memory context.

    LLM inference runs server-side against the in-cluster Ollama instance.

    Request body:
      question  (str, required)
      memories  (list[dict], required — each with a "content" field)
      model     (str, optional — defaults to OLLAMA_MODEL_AGENTS)
      num_ctx   (int, optional, default 32768)
    """
    question = str(payload.get("question") or "")
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question is required",
        )
    memories = list(payload.get("memories") or [])
    prompt_override = payload.get("prompt_override") or None
    if not memories and not prompt_override:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="memories or prompt_override is required",
        )

    try:
        result = await gateway.answer(
            question=question,
            memories=memories,
            model=payload.get("model") or None,
            num_ctx=int(payload.get("num_ctx") or 32768),
            prompt_override=prompt_override,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "answer": result.answer,
        "model": result.model,
        "context_turns": result.context_turns,
        "used_llm": result.used_llm,
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/judge
# ---------------------------------------------------------------------------

@router.post("/judge")
async def gateway_judge(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    gateway: CognitiveGatewayService = Depends(_get_gateway),
) -> dict[str, Any]:
    """Semantic equivalence check between a generated answer and a gold answer.

    Used by benchmark runners to score answers server-side.

    Request body:
      question   (str, required)
      gold       (str, required)
      generated  (str, required)
      model      (str, optional)
    """
    question = str(payload.get("question") or "")
    gold = str(payload.get("gold") or "")
    generated = str(payload.get("generated") or "")
    if not question.strip() or not gold.strip() or not generated.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="question, gold, and generated are all required",
        )

    try:
        result = await gateway.judge(
            question=question,
            gold=gold,
            generated=generated,
            model=payload.get("model") or None,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "equivalent": result.equivalent,
        "raw": result.raw,
        "model": result.model,
    }
