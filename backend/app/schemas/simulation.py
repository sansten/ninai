"""SimulationAgent schemas (Pydantic v2).

Deterministic core output schema for counterfactual evaluation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.base import BaseSchema


RiskFactorType = Literal[
    "insufficient_evidence",
    "tool_unreliable",
    "policy_risk",
    "contradiction",
    "scope_mismatch",
]


class SimulationPlanRisk(BaseSchema):
    success_probability: float = Field(ge=0.0, le=1.0)
    policy_violation_probability: float = Field(ge=0.0, le=1.0)
    data_leak_probability: float = Field(ge=0.0, le=1.0)
    tool_failure_probability: float = Field(ge=0.0, le=1.0)


class SimulationRiskFactor(BaseSchema):
    type: RiskFactorType
    description: str
    affected_steps: list[str] = Field(default_factory=list)
    mitigation: str


class SimulationAddStep(BaseSchema):
    step_id: str
    action: str
    tool: str | None = None
    tool_input_hint: dict[str, Any] = Field(default_factory=dict)
    expected_output: str
    success_criteria: list[str] = Field(default_factory=list)


class SimulationModifyStep(BaseSchema):
    step_id: str
    patch: dict[str, Any] = Field(default_factory=dict)


class SimulationRecommendedPlanPatch(BaseSchema):
    remove_steps: list[str] = Field(default_factory=list)
    add_steps: list[SimulationAddStep] = Field(default_factory=list)
    modify_steps: list[SimulationModifyStep] = Field(default_factory=list)


class SimulationOutput(BaseSchema):
    plan_risk: SimulationPlanRisk
    risk_factors: list[SimulationRiskFactor] = Field(default_factory=list)
    recommended_plan_patch: SimulationRecommendedPlanPatch = Field(default_factory=SimulationRecommendedPlanPatch)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_memory_ids: list[str] = Field(default_factory=list)


class SimulationReportResponse(BaseSchema):
    id: str
    organization_id: str
    session_id: str | None = None
    memory_id: str | None = None
    report: dict[str, Any]
    created_at: datetime
