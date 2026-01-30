"""Schemas for SelfModel (profiles + planner summary)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class SelfModelProfileResponse(BaseSchema):
    organization_id: str
    domain_confidence: dict[str, float] = Field(default_factory=dict)
    tool_reliability: dict[str, Any] = Field(default_factory=dict)
    agent_accuracy: dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime


class SelfModelPlannerSummary(BaseSchema):
    unreliable_tools: list[str] = Field(default_factory=list)
    low_confidence_domains: list[str] = Field(default_factory=list)
    recommended_evidence_multiplier: int = Field(ge=1, le=5, default=1)


class SelfModelBundleResponse(BaseSchema):
    profile: SelfModelProfileResponse
    planner_summary: SelfModelPlannerSummary
