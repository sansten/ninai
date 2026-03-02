"""
PR-8: Emotional & Affective Memory API Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AffectiveMemoryRequest(BaseModel):
    """Request to analyze emotional content of a memory."""

    memory_content: str = Field(..., description="Memory text content to analyze")
    user_ids: Optional[List[str]] = Field(None, description="User IDs involved in this memory")


class AffectiveAnalysisResponse(BaseModel):
    """Response with emotional analysis."""

    valence: float = Field(
        ..., description="Emotional tone: -1.0 (negative) to +1.0 (positive)"
    )
    arousal: float = Field(..., description="Emotional intensity: 0 (calm) to 1 (intense)")
    emotional_tags: List[str] = Field(..., description="Categorical emotional labels")
    significance: float = Field(..., description="How emotionally important is this? 0-1")
    confidence_in_measurement: float = Field(..., description="Confidence in analysis: 0-1")


class EmotionalMeasurement(BaseModel):
    """Single point in time for emotional state."""

    timestamp: str = Field(..., description="ISO format timestamp")
    valence: float
    arousal: float
    tags: Optional[List[str]] = None


class EmotionalTrajectoryResponse(BaseModel):
    """Trajectory of emotional state over time."""

    user_id: str
    measurements: List[EmotionalMeasurement]
    trend: str = Field(..., description="improving | stable | deteriorating")
    current_state: Dict = Field(..., description="Latest emotional snapshot")
    escalation_risk: float = Field(..., description="Probability of escalation: 0-1")
    de_escalation_strategies: List[str] = Field(
        ..., description="Recommended de-escalation approaches"
    )
    is_at_risk: bool = Field(..., description="Is user at immediate escalation risk?")


class EscalationRiskResponse(BaseModel):
    """Escalation risk assessment."""

    user_id: str
    escalation_risk: float = Field(..., description="Risk score: 0-1")
    should_escalate_to_human: bool = Field(
        ..., description="Should human agent take over?"
    )
    recommended_strategy: Optional[str] = Field(
        None, description="Suggested de-escalation strategy"
    )


class EmpatheticResponseRequest(BaseModel):
    """Request to adjust tone empathetically."""

    response_content: str = Field(..., description="Original response to adjust")
    user_valence: float = Field(..., description="User's emotional valence: -1 to +1")
    user_arousal: float = Field(..., description="User's emotional arousal: 0-1")
    user_emotional_tags: Optional[List[str]] = Field(
        None, description="Emotional tags for user state"
    )


class EmpatheticResponseResponse(BaseModel):
    """Adjusted response with empathetic tone."""

    original_response: str
    adjusted_response: str
    tone_adjustments: List[str] = Field(
        ..., description="What adjustments were made?"
    )


class InteractionOutcomeRequest(BaseModel):
    """Record outcome of an emotional interaction."""

    interaction_content: str
    initial_valence: float
    initial_arousal: float
    final_valence: Optional[float] = None
    final_arousal: Optional[float] = None
    agent_response_tone: Optional[str] = None
    de_escalation_applied: Optional[str] = None
    outcome: Optional[str] = Field(
        None, description="success | partial_success | failure | escalated"
    )


class InteractionOutcomeResponse(BaseModel):
    """Confirmation of recorded interaction."""

    event_id: str
    user_id: str
    was_escalation: bool
    was_de_escalated: bool
    outcome: Optional[str]
    created_at: datetime
