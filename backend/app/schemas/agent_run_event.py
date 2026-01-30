"""Agent run event schemas.

Events are the step-level building blocks of trajectories (procedural memory).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from app.schemas.base import BaseSchema


class AgentRunEventCreateRequest(BaseSchema):
    event_type: str = Field(..., min_length=1, max_length=50)
    step_index: int = Field(default=0, ge=0)
    payload: dict[str, Any] = Field(default_factory=dict)
    summary_text: str = Field(default="", max_length=20000)
    created_at: Optional[datetime] = Field(
        default=None,
        description="Optional timestamp override (UTC). If omitted, server sets now().",
    )
    trace_id: Optional[str] = Field(default=None, max_length=100)


class AgentRunEventResponse(BaseSchema):
    id: str
    organization_id: str
    agent_run_id: str
    memory_id: str

    event_type: str
    step_index: int
    payload: dict[str, Any]
    summary_text: str
    created_at: datetime

    trace_id: Optional[str] = None
