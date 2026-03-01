"""Evaluation API schemas (PR6: Eval Harness + Drift Detection)."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# Eval Suite schemas

class EvalSuiteCreate(BaseModel):
    """Schema for creating an eval suite."""
    name: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    queries: list[dict[str, Any]] = Field(default_factory=list)
    expected: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class EvalSuiteUpdate(BaseModel):
    """Schema for updating an eval suite."""
    name: str | None = None
    description: str | None = None
    queries: list[dict[str, Any]] | None = None
    expected: dict[str, Any] | None = None
    is_active: bool | None = None


class EvalSuiteResponse(BaseModel):
    """Schema for eval suite response."""
    id: str
    organization_id: str
    name: str
    description: str | None
    queries: list[dict[str, Any]]
    expected: dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Eval Run schemas

class EvalRunTrigger(BaseModel):
    """Schema for manually triggering an eval run."""
    suite_id: str
    config: dict[str, Any] | None = Field(
        default=None,
        description="Optional config overrides (k_values, thresholds, etc.)"
    )


class EvalRunResponse(BaseModel):
    """Schema for eval run response."""
    id: str
    organization_id: str
    suite_id: str
    started_at: datetime
    finished_at: datetime | None
    config: dict[str, Any]
    metrics: dict[str, Any]
    status: str
    error_message: str | None
    created_at: datetime

    class Config:
        from_attributes = True


# Drift Report schemas

class DriftReportTrigger(BaseModel):
    """Schema for triggering drift computation."""
    baseline_run_id: str
    current_run_id: str


class DriftReportResponse(BaseModel):
    """Schema for drift report response."""
    id: str
    organization_id: str
    baseline_run_id: str
    current_run_id: str
    delta: dict[str, Any]
    severity: str
    flagged_issues: list[str]
    created_at: datetime

    class Config:
        from_attributes = True
