"""Schemas for Cognitive Loop (Planner/Executor/Critic + session records)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.base import BaseSchema


# -----------------
# Agent I/O schemas
# -----------------


class PlannerStep(BaseSchema):
    step_id: str
    action: str
    tool: str | None = None
    tool_input_hint: dict[str, Any] = Field(default_factory=dict)
    expected_output: str
    success_criteria: list[str]
    risk_notes: list[str] = Field(default_factory=list)


class PlannerOutput(BaseSchema):
    objective: str
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    steps: list[PlannerStep]
    stop_conditions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


ExecutorStepStatus = Literal["success", "denied", "failed", "skipped"]


class ExecutorStepResult(BaseSchema):
    step_id: str
    status: ExecutorStepStatus
    tool_name: str | None = None
    tool_call_id: str | None = None
    summary: str
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ExecutorOutput(BaseSchema):
    step_results: list[ExecutorStepResult]
    overall_status: Literal["success", "partial", "failed"]
    errors: list[str] = Field(default_factory=list)


CriticEvaluation = Literal["pass", "fail", "retry", "needs_evidence"]
CriticIssueType = Literal[
    "missing_evidence",
    "policy_violation",
    "low_confidence",
    "contradiction",
    "tool_failure",
]


class CriticIssue(BaseSchema):
    type: CriticIssueType
    description: str
    affected_steps: list[str] = Field(default_factory=list)
    recommended_fix: str


class CriticOutput(BaseSchema):
    evaluation: CriticEvaluation
    strengths: list[str] = Field(default_factory=list)
    issues: list[CriticIssue] = Field(default_factory=list)
    followup_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class EvaluationQualityMetrics(BaseSchema):
    avg_confidence: float = Field(ge=0.0, le=1.0)
    policy_denials: int = Field(ge=0)
    tool_failures: int = Field(ge=0)


class EvaluationReportPayload(BaseSchema):
    final_decision: Literal["pass", "fail", "needs_evidence", "contested"]
    reason: str
    goal_id: str | None = None
    evidence_memory_ids: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    iteration_count: int = Field(ge=1)
    quality_metrics: EvaluationQualityMetrics


# -----------------
# Session schemas
# -----------------


class CognitiveSessionCreateRequest(BaseSchema):
    goal: str
    goal_id: str | None = None
    agent_id: str | None = None
    mode: Literal["interactive", "batch"] = "interactive"
    context_snapshot: dict[str, Any] = Field(default_factory=dict)


class CognitiveSessionResponse(BaseSchema):
    id: str
    organization_id: str
    user_id: str
    agent_id: str | None = None
    status: Literal["running", "succeeded", "failed", "aborted"]
    goal: str
    goal_id: str | None = None
    context_snapshot: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    trace_id: str | None = None


class CognitiveIterationResponse(BaseSchema):
    id: str
    session_id: str
    iteration_num: int
    plan_json: dict[str, Any]
    execution_json: dict[str, Any]
    critique_json: dict[str, Any]
    evaluation: CriticEvaluation
    started_at: datetime
    finished_at: datetime
    metrics: dict[str, Any]


class ToolCallLogResponse(BaseSchema):
    id: str
    session_id: str
    iteration_id: str
    tool_name: str
    tool_input: dict[str, Any]
    tool_output_summary: dict[str, Any]
    status: Literal["success", "denied", "failed"]
    denial_reason: str | None = None
    started_at: datetime
    finished_at: datetime


class EvaluationReportResponse(BaseSchema):
    id: str
    session_id: str
    report: dict[str, Any]
    final_decision: Literal["pass", "fail", "needs_evidence", "contested"]
    created_at: datetime
