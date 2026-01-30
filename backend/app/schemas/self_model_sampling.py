"""SelfModel sampling API schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class ToolOutcomeSampleIn(BaseSchema):
    tool_name: str = Field(..., min_length=1, max_length=255)
    success: bool
    duration_ms: float | None = Field(None, ge=0)
    session_id: str | None = None
    memory_id: str | None = None
    notes: str | None = Field(None, max_length=2000)
    extra: dict[str, Any] = Field(default_factory=dict)


class ToolOutcomeSampleOut(BaseSchema):
    id: str
    organization_id: str
    event_type: str
    tool_name: str
    created_at: datetime


class ToolReliabilityResponse(BaseSchema):
    tool_name: str
    stats: dict[str, Any] = Field(default_factory=dict)
