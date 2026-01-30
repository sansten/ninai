from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ResourceType = Literal["memory", "cognitive_session"]
SupervisionType = Literal["review", "arbitration", "calibration_update"]
MetaRunStatus = Literal["accepted", "modified", "rejected", "contested", "escalated"]
ConflictType = Literal["classification", "topic", "entity", "promotion", "tool", "belief"]
ConflictStatus = Literal["open", "resolved", "ignored"]
ResolvedBy = Literal["meta_auto", "human_admin"]


class MetaAgentRunOut(BaseModel):
    id: str
    organization_id: str
    resource_type: ResourceType
    resource_id: str
    supervision_type: SupervisionType
    status: MetaRunStatus
    final_confidence: float | None = None
    risk_score: float | None = None
    reasoning_summary: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class MetaConflictOut(BaseModel):
    id: str
    organization_id: str
    resource_type: ResourceType
    resource_id: str
    conflict_type: ConflictType
    candidates: dict[str, Any] = Field(default_factory=dict)
    resolution: dict[str, Any] = Field(default_factory=dict)
    resolved_by: ResolvedBy | None = None
    status: ConflictStatus
    resolved_at: datetime | None = None
    created_at: datetime


class CalibrationProfileOut(BaseModel):
    organization_id: str
    promotion_threshold: float
    conflict_escalation_threshold: float
    drift_threshold: float
    signal_weights: dict[str, float] = Field(default_factory=dict)
    learning_rate: float
    updated_at: datetime


class CalibrationProfileUpdateIn(BaseModel):
    signal_weights: dict[str, float] | None = None
    learning_rate: float | None = None
    promotion_threshold: float | None = None
    conflict_escalation_threshold: float | None = None
    drift_threshold: float | None = None


