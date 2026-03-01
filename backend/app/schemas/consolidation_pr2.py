"""Schemas for PR-2 consolidation sleep-cycle APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConsolidationStartRequest(BaseModel):
    session_type: str = Field(default="triggered", pattern="^(nightly|weekly|triggered)$")


class ConsolidationSessionResponse(BaseModel):
    id: str
    session_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    status: str
    operations: Dict[str, Any] = Field(default_factory=dict)
    memory_quality_before: Optional[float] = None
    memory_quality_after: Optional[float] = None


class ConsolidationSessionsResponse(BaseModel):
    items: List[ConsolidationSessionResponse]
    total: int


class MemoryArcResponse(BaseModel):
    memory_id: str
    measurements: List[Dict[str, Any]]
    trend: str
    trajectory_type: str
    prediction_next_access: Optional[datetime] = None
    last_computed_at: datetime


class PinMemoryResponse(BaseModel):
    memory_id: str
    pinned: bool
    message: str
