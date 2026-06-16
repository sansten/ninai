"""
V2 FastAPI Router

Mounted at /v2 when NINAI_ENGINE_VERSION=v2.
All endpoints require the same JWT auth used by v1.

Endpoints:
  POST /v2/interact            — single cognitive turn (3-phase pipeline)
  GET  /v2/enrichment/status   — async-extract pending count for a tenant (cheap poll)
  POST /v2/graph/inspect       — inspect graph subgraph by entity ids
  GET  /v2/health              — component health check
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.v2.api.schemas import (
    V2EnrichmentStatusResponse,
    V2GraphInspectRequest,
    V2GraphInspectResponse,
    V2GraphNode,
    V2HealthResponse,
    V2InteractRequest,
    V2InteractResponse,
)
from app.v2.memory.dnc_router import _count_enrich_pending
from app.v2.pipeline.factory import get_v2_loop

logger = logging.getLogger(__name__)

v2_router = APIRouter(prefix="/v2", tags=["v2-engine"])


def _resolve_tenant(request_tenant: str | None, current_user) -> str:
    """Resolve tenant_id from request or fall back to the user's org."""
    if request_tenant:
        return request_tenant
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


# ---------------------------------------------------------------------------
# Auth dependency — re-use v1 get_tenant_context (same JWT, lighter weight)
# ---------------------------------------------------------------------------
try:
    from app.api.v1.endpoints.auth import get_current_user as _auth_dep
    _AUTH = Depends(_auth_dep)
except ImportError:
    _AUTH = None  # type: ignore[assignment]


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
