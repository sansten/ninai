"""
PR-5: Temporal Reasoning Schemas

Pydantic models for validating temporal reasoning requests and responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Request Schemas
# ============================================================================

class TemporalFactRequest(BaseModel):
    """Request to tag a fact with temporal validity."""
    fact_id: str = Field(..., description="Unique identifier for the fact")
    onset_timestamp: datetime = Field(..., description="When the fact becomes true")
    offset_timestamp: Optional[datetime] = Field(None, description="When the fact stops being true")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in this interval")
    change_type: str = Field("stable", description="onset | offset | stable | transient")


class TemporalSequenceRequest(BaseModel):
    """Request to detect recurring event sequences."""
    entities: List[str] = Field(..., description="Ordered list of entity IDs in sequence")
    min_gap_seconds: int = Field(60, ge=0, description="Minimum gap between events")
    max_gap_seconds: int = Field(300, ge=60, description="Maximum gap between events")
    min_strength: float = Field(0.7, ge=0.0, le=1.0, description="Minimum pattern strength")


class MeasurementPoint(BaseModel):
    """Single measurement point in a trajectory."""
    timestamp: datetime
    value: float


class TrajectoryRequest(BaseModel):
    """Request to analyze trajectory of a quantity over time."""
    entity_id: str = Field(..., description="Entity being tracked")
    quantity: str = Field(..., description="Metric name (e.g., sentiment_score)")
    measurements: List[Dict[str, Any]] = Field(..., description="List of {timestamp, value} measurements")


class ForecastRequest(BaseModel):
    """Request to forecast future trajectory values."""
    entity_id: str = Field(..., description="Entity to forecast")
    horizon_steps: int = Field(7, ge=1, le=52, description="How many periods ahead")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Desired confidence")


class TemporalQueryRequest(BaseModel):
    """Request for temporal SQL-like queries."""
    query_type: str = Field(..., description="facts_valid_at_time | trajectory_crosses_threshold | etc")
    timestamp: Optional[datetime] = None
    entity_id: Optional[str] = None
    threshold: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class ActionTimingRequest(BaseModel):
    """Request to determine optimal action timing."""
    entity_id: str = Field(..., description="Entity to analyze")
    threshold: float = Field(..., description="Critical threshold value")
    measurements: List[Dict[str, Any]] = Field(..., description="Current measurements")
    lookahead_hours: int = Field(24, ge=0, le=2400, description="How far to look ahead")


# ============================================================================
# Response Schemas
# ============================================================================

class TemporalFactResponse(BaseModel):
    """Response from tagging a temporal fact."""
    fact_id: str
    temporal_id: str
    valid_from: datetime
    valid_to: Optional[datetime]
    confidence: float
    change_type: str
    created_at: datetime


class TemporalSequenceResponse(BaseModel):
    """Response from detecting a sequence."""
    sequence_id: str
    entities: List[str]
    pattern_type: str  # escalation | resolution | oscillation | trend
    pattern_strength: float
    temporal_gaps: List[int]
    predicted_next_event: Optional[str]
    observation_count: int
    confidence: float


class InflectionPoint(BaseModel):
    """Identified inflection point in trajectory."""
    timestamp: datetime
    value: float
    trend_change: str
    severity: float
    statistical_significance: float


class TrajectoryResponse(BaseModel):
    """Response from trajectory analysis."""
    trajectory_id: str
    entity_id: str
    quantity: str
    trend_direction: str  # increasing | decreasing | stable | cyclic
    trend_strength: float
    measurement_count: int
    first_measurement: datetime
    last_measurement: datetime
    inflection_points: List[InflectionPoint]
    forecast_generated: bool


class ForecastPoint(BaseModel):
    """Single forecast point with confidence interval."""
    period: int
    forecast_value: float
    lower_bound: float
    upper_bound: float
    confidence: float


class ForecastResponse(BaseModel):
    """Response from trajectory forecasting."""
    entity_id: str
    horizon_steps: int
    confidence_level: float
    forecasts: List[ForecastPoint]
    model_quality: float
    adequate_historical_data: bool


class InflectionPointResponse(BaseModel):
    """Response from inflection point detection."""
    entity_id: str
    inflections_found: int
    inflection_points: List[InflectionPoint]
    sensitivity_threshold: float
    analysis_quality: float


class ActionTimingResponse(BaseModel):
    """Response from optimal action timing analysis."""
    entity_id: str
    action_recommended: bool
    recommended_timestamp: Optional[datetime]
    confidence: float
    reasoning: str
    trend_summary: str
    forecast_quality: float


class TemporalQueryResponse(BaseModel):
    """Response from temporal queries."""
    query_type: str
    result_count: int
    results: List[Dict[str, Any]]
    execution_time_ms: float


# ============================================================================
# Wrapper Schemas
# ============================================================================

class TemporalFactBatchRequest(BaseModel):
    """Batch request for multiple facts."""
    facts: List[TemporalFactRequest]


class TemporalSequenceBatchRequest(BaseModel):
    """Batch request for multiple sequences."""
    sequences: List[TemporalSequenceRequest]


class TrajectoryBatchRequest(BaseModel):
    """Batch request for multiple trajectories."""
    trajectories: List[TrajectoryRequest]


class TemporalSearchOptions(BaseModel):
    """Options for temporal searches."""
    include_confidence: bool = True
    include_source_data: bool = False
    limit_results: Optional[int] = None
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
