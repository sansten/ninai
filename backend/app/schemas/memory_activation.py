"""Memory Activation Scoring Pydantic Schemas

This module contains Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.memory import MemoryResponse


class ActivationComponentsSchema(BaseModel):
    """Breakdown of all 8 activation components for a single memory result."""

    rel: float = Field(..., ge=0.0, le=1.0, description="Relevance (vector similarity)")
    rec: float = Field(..., ge=0.0, le=1.0, description="Recency (exp decay)")
    freq: float = Field(..., ge=0.0, le=1.0, description="Frequency (access count)")
    imp: float = Field(..., ge=0.0, le=1.0, description="Importance (base + feedback)")
    conf: float = Field(..., ge=0.0, le=1.0, description="Confidence (adjusted for contradiction)")
    ctx: float = Field(..., ge=0.0, le=1.0, description="Context gate (scope/episode/goal)")
    prov: float = Field(..., ge=0.0, le=1.0, description="Provenance (evidence links)")
    risk: float = Field(..., ge=0.0, le=1.0, description="Risk (classification)")
    nbr: Optional[float] = Field(None, ge=0.0, le=1.0, description="Neighbor boost (co-activation)")

    class Config:
        frozen = True


class GatingInfoSchema(BaseModel):
    """Information about access gating decision."""

    allowed: bool = Field(..., description="Whether memory passed access control")
    reason: Optional[str] = Field(None, description="Reason if denied (e.g., 'Policy X blocked')")

    class Config:
        frozen = True


class RetrievalResultSchema(BaseModel):
    """Single result in a retrieval explanation log."""

    memory_id: str = Field(..., description="UUID of retrieved memory")
    activation: float = Field(..., ge=0.0, le=1.0, description="Final activation score")
    components: ActivationComponentsSchema = Field(..., description="Breakdown of all components")
    gating: GatingInfoSchema = Field(..., description="Access control decision")
    rank: int = Field(..., ge=1, description="Rank in result set (1-based)")

    class Config:
        frozen = True


class MemoryRetrievalExplanationSchema(BaseModel):
    """Response model for retrieval explanation log entry."""

    id: str = Field(..., description="UUID of explanation log entry")
    organization_id: str = Field(..., description="Organization UUID")
    user_id: str = Field(..., description="User UUID who made the query")
    query_hash: str = Field(..., description="Hash of the query (for grouping)")
    retrieved_at: datetime = Field(..., description="When retrieval happened")
    top_k: int = Field(..., description="Number of results requested")
    results: List[RetrievalResultSchema] = Field(..., description="Scored results with explanations")

    class Config:
        from_attributes = True


class MemoryActivationStateSchema(BaseModel):
    """Response model for memory activation state."""

    id: str = Field(..., description="UUID of activation state record")
    organization_id: str = Field(..., description="Organization UUID")
    memory_id: str = Field(..., description="Memory UUID")
    base_importance: float = Field(..., ge=0.0, le=1.0, description="Importance weight")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in memory")
    contradicted: bool = Field(..., description="Whether memory is contradicted")
    risk_factor: float = Field(..., ge=0.0, le=1.0, description="Risk classification")
    access_count: int = Field(..., ge=0, description="Number of times accessed")
    last_accessed_at: Optional[datetime] = Field(None, description="Most recent access")
    created_at: datetime = Field(..., description="When record created")
    updated_at: datetime = Field(..., description="When record last updated")

    class Config:
        from_attributes = True


class CoactivationEdgeSchema(BaseModel):
    """Response model for co-activation edge."""

    id: str = Field(..., description="UUID of edge record")
    organization_id: str = Field(..., description="Organization UUID")
    memory_id_a: str = Field(..., description="First memory UUID")
    memory_id_b: str = Field(..., description="Second memory UUID")
    coactivation_count: int = Field(..., ge=0, description="Times co-activated together")
    edge_weight: float = Field(..., ge=0.0, le=1.0, description="Weight (1 - exp(-λ*count))")
    last_coactivated_at: Optional[datetime] = Field(None, description="Most recent co-activation")
    created_at: datetime = Field(..., description="When edge created")

    class Config:
        from_attributes = True


class CoactivatedNeighborSchema(BaseModel):
    """Neighbor in co-activation graph."""

    memory_id: str = Field(..., description="Neighbor memory UUID")
    edge_weight: float = Field(..., ge=0.0, le=1.0, description="Co-activation edge weight")
    coactivation_count: int = Field(..., ge=0, description="Times co-activated")

    class Config:
        frozen = True


class CoactivatedNeighborDetailSchema(BaseModel):
    """Neighbor in co-activation graph with memory metadata."""

    edge_weight: float = Field(..., ge=0.0, le=1.0, description="Co-activation edge weight")
    coactivation_count: int = Field(..., ge=0, description="Times co-activated")
    memory: MemoryResponse = Field(..., description="Neighbor memory metadata")

    class Config:
        frozen = True


class RelationTypeEnum(str, Enum):
    """Types of causal relationships."""

    CAUSES = "causes"
    LEADS_TO = "leads_to"
    BLOCKS = "blocks"
    RESOLVES = "resolves"
    CORRELATES = "correlates"


class HypothesisStatusEnum(str, Enum):
    """Lifecycle states for causal hypotheses."""

    PROPOSED = "proposed"
    ACTIVE = "active"
    CONTESTED = "contested"
    REJECTED = "rejected"


class CausalHypothesisSchema(BaseModel):
    """Response model for causal hypothesis."""

    id: str = Field(..., description="UUID of hypothesis")
    organization_id: str = Field(..., description="Organization UUID")
    episode_id: Optional[str] = Field(None, description="Optional episode UUID")
    from_event_id: Optional[str] = Field(None, description="Optional from-event UUID")
    to_event_id: Optional[str] = Field(None, description="Optional to-event UUID")
    relation: RelationTypeEnum = Field(..., description="Type of causal relationship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in hypothesis")
    evidence_memory_ids: Optional[List[str]] = Field(None, description="Supporting memory UUIDs")
    status: HypothesisStatusEnum = Field(..., description="Hypothesis lifecycle state")
    created_at: datetime = Field(..., description="When hypothesis created")
    updated_at: datetime = Field(..., description="When hypothesis last updated")

    class Config:
        from_attributes = True


# ==================== Request Models ====================


class CreateCausalHypothesisRequest(BaseModel):
    """Request to create a causal hypothesis."""

    episode_id: Optional[str] = Field(None, description="Optional episode UUID")
    from_event_id: Optional[str] = Field(None, description="Optional from-event UUID")
    to_event_id: Optional[str] = Field(None, description="Optional to-event UUID")
    relation: RelationTypeEnum = Field(..., description="Type of relationship")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Initial confidence")
    evidence_memory_ids: Optional[List[str]] = Field(None, description="Supporting memory UUIDs")


class UpdateCausalHypothesisRequest(BaseModel):
    """Request to update a causal hypothesis."""

    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    evidence_memory_ids: Optional[List[str]] = Field(None)
    status: Optional[HypothesisStatusEnum] = Field(None)
