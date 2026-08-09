"""
V2 FastAPI Router

Mounted at /v2 when NINAI_ENGINE_VERSION=v2.
All endpoints require the same JWT auth used by v1.

Endpoints:
  POST /v2/interact            — single cognitive turn (3-phase pipeline)
  GET  /v2/enrichment/status   — async-extract pending count for a tenant (cheap poll)
  POST /v2/graph/inspect       — inspect graph subgraph by entity ids
  GET  /v2/context/wiki        — structured world briefing from the knowledge graph
  GET  /v2/health              — component health check
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException

from app.v2.api.schemas import (
    V2ContextWikiResponse,
    V2EnrichmentStatusResponse,
    V2GraphInspectRequest,
    V2GraphInspectResponse,
    V2GraphNode,
    V2HealthResponse,
    V2InteractRequest,
    V2InteractResponse,
    V2WikiEvent,
    V2WikiPerson,
    V2WikiTopic,
)
from app.v2.memory.dnc_router import _count_enrich_pending
from app.v2.pipeline.factory import get_v2_loop

logger = logging.getLogger(__name__)

v2_router = APIRouter(prefix="/v2", tags=["v2-engine"])


# Benchmark harnesses (LongMemEval/LoCoMo) legitimately create thousands of
# synthetic per-question tenants under a single demo account and pass
# tenant_id explicitly to route each question into its own isolated
# namespace — see evaluation/_runner/longmemeval_run.py. Only bench-mode
# deployments trust a caller-supplied tenant override outright; everywhere
# else it must match the caller's own org unless they're a superuser.
_BENCH_MODE = os.environ.get("NINAI_BENCH_MODE", "0").lower() in ("1", "true", "yes")


def _own_org(current_user) -> str:
    # current_user is a User model or dict depending on the auth dependency
    if hasattr(current_user, "organization_id"):
        return str(current_user.organization_id or "unknown")
    if isinstance(current_user, dict):
        return str(
            current_user.get("org_id")
            or current_user.get("organization_id")
            or current_user.get("sub", "unknown")
        )
    return "unknown"


def _resolve_tenant(request_tenant: str | None, current_user) -> str:
    """Resolve tenant_id from request, enforcing it matches the caller's own org.

    A caller-supplied tenant_id that differs from the authenticated user's org
    is only honored for bench-mode deployments (synthetic per-question
    tenants) or superusers — otherwise a regular user could read/write any
    other organization's memory graph simply by naming its tenant_id.
    """
    own_org = _own_org(current_user)
    if not request_tenant:
        return own_org
    if request_tenant == own_org:
        return request_tenant
    if _BENCH_MODE or getattr(current_user, "is_superuser", False):
        return request_tenant
    raise HTTPException(
        status_code=403,
        detail="tenant_id does not match the authenticated organization",
    )


# ---------------------------------------------------------------------------
# Auth dependency — re-use v1 get_tenant_context (same JWT, lighter weight)
# ---------------------------------------------------------------------------
try:
    from app.api.v1.endpoints.auth import get_current_user as _auth_dep
    _AUTH = Depends(_auth_dep)
except ImportError:
    # Fail CLOSED, not open: a bare `_AUTH = None` here used to make every v2
    # endpoint's `current_user=_AUTH` default to None with no dependency at
    # all — i.e. fully unauthenticated — if this import ever broke (circular
    # import, refactor, missing module). Refuse every request instead.
    logger.critical(
        "v2 auth dependency failed to import; all /v2 endpoints will return "
        "503 until this is fixed."
    )

    async def _auth_unavailable():
        raise HTTPException(status_code=503, detail="v2 authentication is unavailable")

    _AUTH = Depends(_auth_unavailable)


# ---------------------------------------------------------------------------
# POST /v2/interact
# ---------------------------------------------------------------------------

@v2_router.post("/interact", response_model=V2InteractResponse)
async def v2_interact(
    req: V2InteractRequest,
    current_user=_AUTH,  # type: ignore[assignment]
) -> V2InteractResponse:
    """
    Run the full three-phase v2 cognitive loop for one user turn.

    Phase 1: dual-path retrieval (FalkorDB subgraph + Qdrant dense)
    Phase 2: Graph-RAG inference via configured LLM backend
    Phase 3: graph write-back + decay + pruning
    """
    tenant_id = _resolve_tenant(req.tenant_id, current_user or {})
    loop = get_v2_loop()

    result = await loop.run(
        tenant_id=tenant_id,
        session_id=req.session_id,
        user_input=req.user_input,
        disable_write=req.disable_write,
        ingest_only=req.ingest_only,
        prev_utterance_id=req.prev_utterance_id,
        model_hint=req.model_hint,
        raw_context=req.raw_context,
    )

    return V2InteractResponse(
        response=result.response,
        session_id=req.session_id,
        user_utterance_id=result.user_utterance_id,
        assistant_utterance_id=result.assistant_utterance_id,
        cited_node_ids=result.cited_node_ids,
        extracted_entities=result.extracted_entities,
        graph_nodes_retrieved=result.graph_nodes_retrieved,
        qdrant_chunks_retrieved=result.qdrant_chunks_retrieved,
        graph_writes=result.graph_writes,
        decay_stats=result.decay_stats,
        enrichment_pending=result.enrichment_pending,
        pending_enrichments=result.pending_enrichments,
        latency_ms=result.latency_ms,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# GET /v2/enrichment/status
# ---------------------------------------------------------------------------

@v2_router.get("/enrichment/status", response_model=V2EnrichmentStatusResponse)
async def v2_enrichment_status(
    tenant: str | None = None,
    current_user=_AUTH,  # type: ignore[assignment]
) -> V2EnrichmentStatusResponse:
    """How many async entity extractions are still in flight for the tenant.

    Cheap counterpart to the `enrichment_pending` field on /v2/interact: a client can
    poll this after ingest and wait for `pending == 0` before issuing a graph-dependent
    query, without paying for a full cognitive turn. Always 0 in the default inline
    (synchronous) extraction path.
    """
    tenant_id = _resolve_tenant(tenant, current_user or {})
    pending = await _count_enrich_pending(tenant_id)
    return V2EnrichmentStatusResponse(
        tenant_id=tenant_id,
        pending=pending,
        enrichment_pending=pending > 0,
    )


# ---------------------------------------------------------------------------
# POST /v2/graph/inspect
# ---------------------------------------------------------------------------

@v2_router.post("/graph/inspect", response_model=V2GraphInspectResponse)
async def v2_graph_inspect(
    req: V2GraphInspectRequest,
    current_user=_AUTH,  # type: ignore[assignment]
) -> V2GraphInspectResponse:
    """Return the FalkorDB subgraph around the given entity seed ids."""
    tenant_id = _resolve_tenant(req.tenant_id, current_user or {})
    loop = get_v2_loop()
    graph_client = loop._router._graph  # type: ignore[attr-defined]

    nodes_raw = await graph_client.fetch_subgraph(
        tenant_id=tenant_id,
        seed_ids=req.entity_ids,
        hops=req.hops,
        limit=req.limit,
    )

    nodes = [
        V2GraphNode(
            id=str(n.get("id", "")),
            label=str(n.get("label", "Node")),
            content=str(n.get("content") or n.get("text") or n.get("name") or ""),
            weight=float(n.get("weight") or 0.0),
            created_at=int(n.get("created_at") or 0),
        )
        for n in nodes_raw
    ]
    return V2GraphInspectResponse(
        nodes=nodes,
        seed_count=len(req.entity_ids),
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# GET /v2/context/wiki
# ---------------------------------------------------------------------------

def _build_wiki_text(
    people: list[V2WikiPerson],
    events: list[V2WikiEvent],
    topics: list[V2WikiTopic],
) -> str:
    lines: list[str] = []
    if people:
        lines.append("## People")
        for p in people:
            lines.append(f"**{p.name}**: {p.profile}")
        lines.append("")
    if events:
        lines.append("## Recent Events")
        for e in events:
            prefix = f"[{e.date}]" if e.date else ""
            subj = f"{e.subject}: " if e.subject else ""
            lines.append(f"{prefix} {subj}{e.summary}".strip())
        lines.append("")
    if topics:
        lines.append("## Key Topics")
        for t in topics:
            lines.append(f"**{t.name}** ({t.entity_type}): {t.summary}")
    return "\n".join(lines)


@v2_router.get("/context/wiki", response_model=V2ContextWikiResponse)
async def v2_context_wiki(
    tenant: str | None = None,
    limit_people: int = 10,
    limit_events: int = 15,
    limit_topics: int = 20,
    current_user=_AUTH,  # type: ignore[assignment]
) -> V2ContextWikiResponse:
    """Return a structured world briefing assembled from the tenant's knowledge graph.

    Assembles person profiles, recent temporal events, and top-weight entities into
    a pre-formatted context page ('LLM wiki') that an agent can load before starting
    a task — grounding it in what the system knows without a full retrieval pass.
    """
    tenant_id = _resolve_tenant(tenant, current_user or {})
    loop = get_v2_loop()
    graph_client = loop._router._graph  # type: ignore[attr-defined]

    people_raw, events_raw, topics_raw = await __import__("asyncio").gather(
        graph_client.fetch_all_profiles(tenant_id, limit=limit_people),
        graph_client.fetch_recent_temporal_events(tenant_id, limit=limit_events),
        graph_client.fetch_top_entities(tenant_id, limit=limit_topics),
    )

    people = [
        V2WikiPerson(
            name=str(p.get("subject") or p.get("name") or ""),
            profile=str(p.get("content") or ""),
        )
        for p in people_raw
        if p.get("subject") or p.get("name")
    ]

    events = [
        V2WikiEvent(
            date=str(e.get("canonical_date") or ""),
            subject=str(e.get("subject") or ""),
            summary=str(e.get("content") or e.get("name") or ""),
        )
        for e in events_raw
    ]

    topics = [
        V2WikiTopic(
            name=str(t.get("name") or ""),
            summary=str(t.get("content") or "")[:200],
            entity_type=str(t.get("entity_type") or ""),
            weight=float(t.get("weight") or 0.0),
        )
        for t in topics_raw
        if t.get("name")
    ]

    wiki_text = _build_wiki_text(people, events, topics)
    return V2ContextWikiResponse(
        tenant_id=tenant_id,
        people=people,
        recent_events=events,
        topics=topics,
        wiki_text=wiki_text,
    )


# ---------------------------------------------------------------------------
# GET /v2/health
# ---------------------------------------------------------------------------

@v2_router.get("/health", response_model=V2HealthResponse)
async def v2_health() -> V2HealthResponse:
    """Verify v2 component connectivity."""
    loop = get_v2_loop()
    graph_ok = loop._router._graph.is_available()  # type: ignore[attr-defined]
    llm_ok = False
    try:
        llm_ok = await loop._engine.is_available()  # type: ignore[attr-defined]
    except Exception:
        pass

    msg_parts = []
    if not graph_ok:
        msg_parts.append("FalkorDB unreachable")
    if not llm_ok:
        msg_parts.append("LLM backend unreachable")

    return V2HealthResponse(
        graph_available=graph_ok,
        llm_available=llm_ok,
        message="; ".join(msg_parts) if msg_parts else "all systems operational",
    )
