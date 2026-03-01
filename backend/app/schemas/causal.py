"""
Causal Reasoning API Schemas (PR-1)
===================================

Pydantic models for causal endpoint requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CausalEdgeResponse(BaseModel):
    """Response model for a causal edge."""

    id: str
    organization_id: str
    cause_entity_id: str
    cause_entity_type: str
    effect_entity_id: str
    effect_entity_type: str
    mechanism: str
    strength: float = Field(ge=0.0, le=1.0)
    latency_hours: Optional[int] = None
    evidence_memory_ids: List[str] = Field(default_factory=list)
    discovered_at: datetime
    validation_count: int = 0
    invalidation_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class DiscoverCausalEdgesRequest(BaseModel):
    """Request to discover causal edges from an episode."""

    episode_id: str
    auto_save: bool = Field(default=True, description="Save discovered edges to database")


class DiscoverCausalEdgesResponse(BaseModel):
    """Response with discovered edges."""

    episode_id: str
    edges_discovered: int
    edges: List[CausalEdgeResponse] = Field(default_factory=list)


class PredictionRequest(BaseModel):
    """Request for causal prediction."""

    cause_ids: List[str] = Field(..., description="Starting cause fact IDs")
    target_id: str = Field(..., description="Target fact ID to predict")
    max_depth: int = Field(default=3, ge=1, le=10)


class PredictionResponse(BaseModel):
    """Causal prediction result."""

    outcome_id: str
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    causal_chain: List[str] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    """Request for causal explanation."""

    effect_id: str = Field(..., description="Effect/outcome to explain")


class ExplainResponse(BaseModel):
    """Causal explanation result."""

    effect_id: str
    root_causes: List[Dict[str, Any]] = Field(default_factory=list)


class CounterfactualRequest(BaseModel):
    """Counterfactual query request."""

    condition: str = Field(..., description="If condition X, what about Y?")
    query: Dict[str, Any] = Field(default_factory=dict)


class CounterfactualResponse(BaseModel):
    """Counterfactual scenario result."""

    scenario_id: str
    condition: str
    status: str


class DoCalculusRequest(BaseModel):
    """Do-calculus request for causal effects."""

    treatment: str = Field(..., description="Intervention variable")
    outcome_var: str = Field(..., description="Outcome to measure")


class DoCalculusResponse(BaseModel):
    """Causal effect estimation."""

    treatment: str
    outcome: str
    causal_effect: float = Field(ge=0.0, le=1.0)
    num_paths: int = 0


class ValidateEdgeRequest(BaseModel):
    """Request to validate a causal edge."""

    edge_id: str
    success: bool = Field(..., description="Did the causal prediction hold?")
    strength_adjustment: float = Field(
        default=0.1, ge=0.0, le=1.0, description="How much to adjust strength"
    )


class ValidateEdgeResponse(BaseModel):
    """Updated edge after validation."""

    id: str
    strength: float = Field(ge=0.0, le=1.0)
    validation_count: int
    invalidation_count: int
    last_validated_at: datetime

    class Config:
        from_attributes = True
