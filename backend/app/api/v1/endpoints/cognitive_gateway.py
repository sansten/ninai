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

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.query_intelligence_agent import run_llm_only_query_intelligence
from app.core.database import get_db, set_tenant_context
from app.core.requester_context import RequesterContext
from app.middleware.tenant_context import TenantContext, get_requester_context, require_org_admin
from app.schemas.memory import MemoryResponse, MemorySearchRequest, SearchHnmsMode
from app.services.cognitive_evidence_service import CognitiveEvidenceService
from app.services.cognitive_ingestion_service import CognitiveIngestionService
from app.services.cognitive_read_planner import CognitiveReadPlanner
from app.services.grounded_answer_service import GroundedAnswerService
from app.services.usage_service import UsageService
from app.services.embedding_service import EmbeddingService
from app.services.cognitive_gateway_service import (
    CognitiveGatewayCapabilities,
    CognitiveGatewayService,
    GatewayContextSession,
    load_gateway_context,
    save_gateway_context,
)
from app.services.memory_service import MemoryService

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


def _coerce_hnms_mode(value: Any) -> SearchHnmsMode | None:
    if value is None or value == "":
        return None
    if isinstance(value, SearchHnmsMode):
        return value
    try:
        return SearchHnmsMode(str(value).strip().lower())
    except ValueError:
        return None


def _merge_str_lists(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
    return merged


def _expand_query_with_intelligence(query: str, intelligence: dict[str, Any]) -> str:
    extras = list(intelligence.get("extracted_entities") or [])
    intent = str(intelligence.get("query_intent") or "retrieve").strip().lower()

    if intent == "find_timeline":
        extras.extend(["timeline", "date", "sequence"])
    elif intent == "find_person":
        extras.extend(["person", "name"])
    elif intent == "explain":
        extras.extend(["cause", "reason"])
    elif intent == "compare":
        extras.extend(["difference", "contrast"])
    elif intent == "analyze":
        extras.extend(["pattern", "trend"])

    merged = _merge_str_lists(extras)
    if not merged:
        return query
    return (query + " " + " ".join(merged[:6])).strip()


def _memory_to_gateway_candidate(memory: Any) -> dict[str, Any]:
    return MemoryResponse.model_validate(memory).model_dump(mode="python")


def _looks_structured_prompt_override(prompt_override: Any) -> bool:
    text = str(prompt_override or "")
    if not text.strip():
        return False
    lowered = text.lower()
    line_count = text.count("\n") + 1
    if len(text) > 3500 or line_count > 60:
        return True
    return (
        "conversation:" in lowered
        and "question:" in lowered
        and "answer:" in lowered
    )


def _lightweight_evidence_package(question: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    seen_facts: set[tuple[str, str, str]] = set()
    memory_scores: list[float] = []

    for memory in memories:
        try:
            memory_scores.append(float(memory.get("score") or 0.0))
        except (TypeError, ValueError):
            memory_scores.append(0.0)

        extra = dict(memory.get("extra_metadata") or {})
        fact_support = dict(extra.get("fact_support") or {})
        support_candidates = []
        if fact_support:
            support_candidates.append(fact_support)
        support_candidates.extend(list(extra.get("fact_supporting_facts") or []))

        for item in support_candidates:
            subject = str((item or {}).get("subject") or "").strip()
            predicate = str((item or {}).get("predicate") or "").strip()
            obj = str((item or {}).get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            key = (subject.lower(), predicate.lower(), obj.lower())
            if key in seen_facts:
                continue
            seen_facts.add(key)
            facts.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "status": str((item or {}).get("status") or "active"),
                    "confidence": float((item or {}).get("confidence") or 0.8),
                }
            )

    avg_memory_score = sum(memory_scores) / len(memory_scores) if memory_scores else 0.0
    return {
        "query": question,
        "memory_hits": list(memories or []),
        "facts": facts,
        "contradictions": [],
        "query_intelligence": {"extracted_entities": []},
        "evidence_quality": {
            "memory_count": len(memories or []),
            "fact_count": len(facts),
            "avg_memory_score": round(avg_memory_score, 4),
            "avg_semantic_quality": 0.0,
            "avg_feedback_signal": 0.0,
        },
    }


def _apply_query_intelligence_filters(
    memories: list[dict[str, Any]],
    intelligence: dict[str, Any],
) -> list[dict[str, Any]]:
    filters = dict(intelligence.get("dynamic_filters") or {})
    if not filters:
        return memories

    filtered = list(memories)

    min_credibility = filters.get("min_credibility")
    try:
        min_credibility_f = float(min_credibility) if min_credibility is not None else None
    except (TypeError, ValueError):
        min_credibility_f = None
    if min_credibility_f is not None:
        def _credibility(mem: dict[str, Any]) -> float:
            raw = (
                mem.get("credibility_score")
                or (mem.get("enrichment") or {}).get("credibility_score")
                or 0.0
            )
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        filtered = [
            mem
            for mem in filtered
            if _credibility(mem) >= min_credibility_f
        ]

    excluded_uncertainty = {
        str(v).strip().lower()
        for v in (filters.get("exclude_uncertainty_levels") or [])
        if str(v).strip()
    }
    if excluded_uncertainty:
        kept: list[dict[str, Any]] = []
        for mem in filtered:
            uncertainty = str(
                mem.get("uncertainty_level")
                or (mem.get("enrichment") or {}).get("uncertainty_level")
                or (mem.get("extra_metadata") or {}).get("uncertainty_level")
                or ""
            ).strip().lower()
            if uncertainty not in excluded_uncertainty:
                kept.append(mem)
        filtered = kept

    if filters.get("has_temporal_data"):
        def _score(mem: dict[str, Any]) -> float:
            try:
                return float(mem.get("score") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        filtered.sort(
            key=lambda mem: (
                not bool(
                    mem.get("occurred_at")
                    or (mem.get("extra_metadata") or {}).get("event_time")
                    or (mem.get("enrichment") or {}).get("temporal_anchor")
                ),
                -_score(mem),
            )
        )

    return filtered


async def _compose_gateway_candidates(
    *,
    query: str,
    payload: dict[str, Any],
    tenant: TenantContext,
    db: AsyncSession,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    try:
        intelligence = await run_llm_only_query_intelligence(query, {})
    except Exception:
        # Keep retrieval available even if LLM intelligence is temporarily unavailable.
        intelligence = {
            "query_intent": "retrieve",
            "extracted_entities": [],
            "dynamic_filters": {},
            "suggested_agents": [],
            "confidence": 0.0,
            "rationale": "llm_unavailable",
        }
    expanded_query = _expand_query_with_intelligence(query, intelligence)
    filter_tags = _merge_str_lists(
        payload.get("filter_tags"),
        (intelligence.get("dynamic_filters") or {}).get("tags"),
    )
    limit = max(1, min(int(payload.get("limit") or 10), 100))

    search_request = MemorySearchRequest(
        query=expanded_query,
        scope=payload.get("scope"),
        team_id=payload.get("team_id"),
        limit=min(max(limit * 3, 10), 100),
        hybrid=bool(payload.get("hybrid", True)),
        use_graph=bool(payload.get("use_graph")) or bool(intelligence.get("extracted_entities")),
        hnms_mode=_coerce_hnms_mode(payload.get("hnms_mode")),
        tags=filter_tags or None,
    )

    memory_service = MemoryService(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        clearance_level=tenant.clearance_level,
    )
    query_embedding = await EmbeddingService.embed(expanded_query)
    results = await memory_service.search_memories(query_embedding=query_embedding, request=search_request)
    candidates = [_memory_to_gateway_candidate(memory) for memory in results]
    candidates = _apply_query_intelligence_filters(candidates, intelligence)
    return candidates, intelligence, expanded_query


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/write
# ---------------------------------------------------------------------------

@router.post("/write")
async def gateway_write(
    request: Request,
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    requester: RequesterContext = Depends(get_requester_context),
    db: AsyncSession = Depends(get_db),
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
    request_id = getattr(request.state, "request_id", None)

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    try:
        if hasattr(gateway, "_check"):
            gateway._check("write")

        ingestion_service = CognitiveIngestionService(
            session=db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            clearance_level=tenant.clearance_level,
            roles_string=tenant.roles_string,
        )
        memory_create = CognitiveIngestionService.build_gateway_memory_create(
            content=content,
            title=str(payload.get("title") or ""),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            payload=payload,
            requester=requester,
            context_id=context_id,
        )
        ingestion = await ingestion_service.ingest_memory(
            data=memory_create,
            request_id=request_id,
            requester=requester,
            storage="long_term",
        )
        result = await gateway.write(
            content=content,
            title=str(payload.get("title") or ""),
            tags=list(payload.get("tags") or []),
            metadata=dict(payload.get("metadata") or {}),
            memory_id=ingestion.memory.id,
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
        "storage": ingestion.storage,
        "memory": MemoryResponse.model_validate(ingestion.memory).model_dump(mode="python"),
    }


# ---------------------------------------------------------------------------
# POST /cognitive/gateway/read
# ---------------------------------------------------------------------------

@router.post("/read")
async def gateway_read(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
    requester: RequesterContext = Depends(get_requester_context),
    db: AsyncSession = Depends(get_db),
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
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    filter_tags = payload.get("filter_tags") or None
    if filter_tags and not isinstance(filter_tags, list):
        filter_tags = None

    try:
        planner = CognitiveReadPlanner(
            db,
            user_id=tenant.user_id,
            org_id=tenant.org_id,
            clearance_level=tenant.clearance_level,
            gateway=gateway,
        )
        planned = await planner.plan_and_read(
            query=query,
            limit=int(payload.get("limit") or 10),
            filter_tags=filter_tags,
            context_id=context_id,
            scope=payload.get("scope"),
            team_id=payload.get("team_id"),
            hybrid=bool(payload.get("hybrid", True)),
            use_graph=bool(payload.get("use_graph")) if "use_graph" in payload else None,
            hnms_mode=_coerce_hnms_mode(payload.get("hnms_mode")),
            supplied_memories=list(payload.get("memories") or []),
            requester=requester,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "memories": planned.memories,
        "total": planned.total,
        "query": planned.query,
        "context_assembled": planned.context_assembled,
        "retrieval_confidence": planned.retrieval_confidence,
        "reasoning_steps": planned.reasoning_steps,
        "compression_ratio": planned.compression_ratio,
        "information_density": planned.information_density,
        "context_id": context_id,
        "working_set_summary": await _context_working_set_summary(context_id, tenant.org_id),
        "retrieval_strategy": planned.retrieval_strategy,
        "target_memory_level": planned.target_memory_level,
        "expanded_query": planned.expanded_query if planned.expanded_query != query else None,
        "query_intelligence": planned.query_intelligence,
        "evidence_package": planned.evidence_package,
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
    requester: RequesterContext = Depends(get_requester_context),
    db: AsyncSession = Depends(get_db),
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
    context_id = payload.get("context_id") or None
    needs_db_context = not (prompt_override and memories)
    grounded_from_supplied_memories = bool(memories) and _looks_structured_prompt_override(prompt_override)

    _timeout_raw = payload.get("timeout_seconds") or payload.get("timeout")
    _timeout_val: float | None = None
    if _timeout_raw is not None:
        try:
            _timeout_val = float(_timeout_raw)
        except (TypeError, ValueError):
            _timeout_val = None

    _keep_alive_raw = payload.get("keep_alive")
    _keep_alive_val: int | None = None
    if _keep_alive_raw is not None:
        try:
            _keep_alive_val = int(_keep_alive_raw)
        except (TypeError, ValueError):
            _keep_alive_val = None

    try:
        if needs_db_context:
            await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

        evidence_package = dict(payload.get("evidence_package") or {})
        planned = None

        if grounded_from_supplied_memories and not evidence_package:
            evidence_package = _lightweight_evidence_package(question, memories)

        if not prompt_override and not memories:
            planner = CognitiveReadPlanner(
                db,
                user_id=tenant.user_id,
                org_id=tenant.org_id,
                clearance_level=tenant.clearance_level,
                gateway=gateway,
            )
            planned = await planner.plan_and_read(
                query=question,
                limit=int(payload.get("limit") or 10),
                context_id=context_id,
                scope=payload.get("scope"),
                team_id=payload.get("team_id"),
                filter_tags=payload.get("filter_tags") if isinstance(payload.get("filter_tags"), list) else None,
                hybrid=bool(payload.get("hybrid", True)),
                use_graph=bool(payload.get("use_graph")) if "use_graph" in payload else None,
                hnms_mode=_coerce_hnms_mode(payload.get("hnms_mode")),
                requester=requester,
            )
            memories = list(planned.memories or [])
            evidence_package = dict(planned.evidence_package or {})
        if grounded_from_supplied_memories or not prompt_override:
            if not evidence_package:
                evidence_service = CognitiveEvidenceService(db, org_id=tenant.org_id)
                evidence_package = await evidence_service.build_package(
                    query=question,
                    memories=memories,
                )

            grounded = await GroundedAnswerService(gateway).answer(
                question=question,
                evidence_package=evidence_package,
                memories=memories,
                model=payload.get("model") or None,
                num_ctx=int(payload.get("num_ctx") or 32768),
                timeout_seconds=_timeout_val,
                keep_alive=_keep_alive_val,
            )
            return {
                "answer": grounded.answer,
                "model": grounded.model,
                "context_turns": grounded.context_turns,
                "used_llm": grounded.used_llm,
                "answer_source": grounded.answer_source,
                "llm_error": grounded.llm_error,
                "llm_failure_mode": getattr(grounded, "llm_failure_mode", None),
                "llm_endpoint": getattr(grounded, "llm_endpoint", None),
                "grounded": grounded.grounded,
                "confidence": grounded.confidence,
                "support": grounded.support,
                "contradictions": grounded.contradictions,
                "uncertainty_reason": grounded.uncertainty_reason,
                "evidence_package": evidence_package,
                "query_intelligence": planned.query_intelligence if planned else None,
                "retrieval_strategy": planned.retrieval_strategy if planned else None,
                "target_memory_level": planned.target_memory_level if planned else None,
                "context_id": context_id,
            }

        result = await gateway.answer(
            question=question,
            memories=memories,
            model=payload.get("model") or None,
            num_ctx=int(payload.get("num_ctx") or 32768),
            prompt_override=prompt_override,
            timeout_seconds=_timeout_val,
            keep_alive=_keep_alive_val,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "answer": result.answer,
        "model": result.model,
        "context_turns": result.context_turns,
        "used_llm": result.used_llm,
        "answer_source": result.answer_source,
        "llm_error": result.llm_error,
        "llm_failure_mode": getattr(result, "llm_failure_mode", None),
        "llm_endpoint": getattr(result, "llm_endpoint", None),
        "context_id": context_id,
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
