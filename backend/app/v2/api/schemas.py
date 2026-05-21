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
    prev_utterance_id: str | None = Field(
        default=None,
        description="Explicit previous utterance id for turn chaining; "
                    "inferred from session if omitted",
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
    latency_ms: int
    error: str


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
    ollama_available: bool
    message: str
