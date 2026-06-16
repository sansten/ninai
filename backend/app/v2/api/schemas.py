"""V2 API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class V2InteractRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(..., min_length=1, max_length=128)
    tenant_id: str | None = Field(
        default=None,
        description="Override tenant; defaults to the authenticated org's tenant_id",
    )
    disable_write: bool = Field(
        default=False,
        description="When true, skip Phase 3 write-back so evaluation does not mutate memory.",
    )
    ingest_only: bool = Field(
        default=False,
        description="When true, skip LLM inference (Phase 2) and only run write-back (Phase 3). "
                    "Populates both Qdrant and FalkorDB without paying LLM latency per turn.",
    )
    prev_utterance_id: str | None = Field(
        default=None,
        description="Explicit previous utterance id for turn chaining; "
                    "inferred from session if omitted",
    )
    model_hint: str | None = Field(
        default=None,
        description="Override the inference model for this request only "
                    "(e.g. 'qwen2.5:7b' for fast SLM, 'qwen2.5:32b' for deep reasoning). "
                    "Falls back to the server-configured VLLM_MODEL when omitted.",
    )
    raw_context: str | None = Field(
        default=None,
        description="When provided and NINAI_FULL_CONV_BYPASS=1, skip Phase 1 retrieval and "
                    "use this text directly as the context for inference.",
    )


class V2InteractResponse(BaseModel):
    response: str
    session_id: str
    user_utterance_id: str
    assistant_utterance_id: str
    cited_node_ids: list[str]
    extracted_entities: list[dict[str, Any]]
    graph_nodes_retrieved: int
    qdrant_chunks_retrieved: int
    graph_writes: int
    decay_stats: dict[str, int]
    # Async-extract progress hint (NINAI_ASYNC_EXTRACT). pending_enrichments > 0 /
    # enrichment_pending = True means the entity graph for this tenant is still being
    # back-filled, so graph-derived context in this response may be incomplete.
    enrichment_pending: bool = False
    pending_enrichments: int = 0
    latency_ms: int
    error: str


class V2EnrichmentStatusResponse(BaseModel):
    """Cheap poll for async-extract progress (NINAI_ASYNC_EXTRACT). Lets a client wait
    for the entity graph to finish back-filling before issuing a graph-dependent query,
    without paying for a full /v2/interact call."""
    tenant_id: str
    pending: int = 0
    enrichment_pending: bool = False


class V2GraphInspectRequest(BaseModel):
    entity_ids: list[str] = Field(
        ..., min_length=1, description="Seed entity ids for subgraph retrieval"
    )
    hops: int = Field(default=2, ge=1, le=4)
    limit: int = Field(default=30, ge=1, le=200)
    tenant_id: str | None = None


class V2GraphNode(BaseModel):
    id: str
    label: str
    content: str
    weight: float
    created_at: int


class V2GraphInspectResponse(BaseModel):
    nodes: list[V2GraphNode]
    seed_count: int
    tenant_id: str


class V2HealthResponse(BaseModel):
    engine_version: str = "v2"
    graph_available: bool
    llm_available: bool
    message: str
