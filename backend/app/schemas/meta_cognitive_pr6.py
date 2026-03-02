"""
PR-6: Meta-Cognitive Planning Schemas
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class StrategyRequest(BaseModel):
    """Request for selecting a cognitive strategy."""

    query: str = Field(..., min_length=1)
    time_budget_seconds: int = Field(30, ge=5, le=3600)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)


class ComplexityResponse(BaseModel):
    """Complexity estimation response."""

    query: str
    complexity_score: float


class StrategyResponse(BaseModel):
    """Selected strategy and resource allocation."""

    query: str
    complexity_estimated: float
    strategy_selected: str
    retrieval_budget: int
    reasoning_depth: int
    verification_required: bool
    confidence_threshold: float
    time_budget_seconds: int
    expected_answer_quality: float
    created_at: str


class ConfidenceCalibrationResponse(BaseModel):
    """Confidence calibration status response."""

    confidence_calibration: float
    surprise_frequency: float
    well_calibrated: bool


class EpistemicStateResponse(BaseModel):
    """Current epistemic state response."""

    timestamp: str
    known_domains: List[str]
    uncertain_domains: List[str]
    unknown_domains: List[str]
    confidence_calibration: Optional[float] = None
    surprise_frequency: Optional[float] = None
