"""Agent Run Schemas
=================

Response schemas for agent run observability and trajectory/procedural memory MVP.

These are intentionally thin wrappers around the persisted AgentRun model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from app.schemas.base import BaseSchema


class AgentRunSummaryResponse(BaseSchema):
    """Lightweight agent run record suitable for lists."""

    id: str
    organization_id: str
    memory_id: str

    agent_name: str
    agent_version: str

    status: str = Field(..., description="success|retry|failed|skipped")
    confidence: float = Field(..., ge=0.0, le=1.0)

    started_at: datetime
    finished_at: datetime

    trace_id: Optional[str] = None


class AgentRunDetailResponse(AgentRunSummaryResponse):
    """Full agent run record."""

    inputs_hash: str

    outputs: dict[str, Any] = Field(default_factory=dict)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CheckpointSnapshot(BaseSchema):
    """Single checkpoint snapshot for replay."""

    id: str
    step_index: int
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    retrieval_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ReplayResponse(BaseSchema):
    """Full replay data for time-travel debugging."""

    agent_run_id: str
    checkpoints: list[CheckpointSnapshot] = Field(default_factory=list)


class RetrievalExplanation(BaseSchema):
    """Detailed explanation of why specific memories were retrieved."""

    step_index: int
    input_query: str
    input_filters: dict[str, Any] = Field(default_factory=dict)
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieved_scores: list[float] = Field(default_factory=list)
    retrieval_filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_cutoff: Optional[float] = Field(default=None)
    model_state: dict[str, Any] = Field(default_factory=dict)
    step_output_keys: list[str] = Field(default_factory=list)


class ReproduceRequest(BaseSchema):
    """Request to reproduce a specific step."""

    step_index: int


class ReproduceResponse(BaseSchema):
    """Result of step reproduction."""

    step_index: int
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    retrieval_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
