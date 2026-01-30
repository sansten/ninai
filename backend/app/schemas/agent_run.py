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
