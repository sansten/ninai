"""Memory Activation Scoring Database Models

This module contains SQLAlchemy ORM models for the memory activation scoring system.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Index,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship

from app.models.base import Base


class MemoryActivationState(Base):
    """Stores evolving activation metrics for each memory.
    
    Tracks:
    - base_importance: User/system-provided importance weight (0-1)
    - confidence: How confident we are in this memory (0-1)
    - contradicted: Whether this memory is marked as contradicted by evidence
    - risk_factor: Risk classification (0-1)
    - access_count: Number of times accessed
    - last_accessed_at: Most recent access timestamp
    """

    __tablename__ = "memory_activation_state"
    __table_args__ = (
        Index("ix_memory_activation_state_org_memory", "organization_id", "memory_id", unique=True),
        Index("ix_memory_activation_state_org_accessed", "organization_id", "last_accessed_at"),
    )

    id = Column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    organization_id = Column(PG_UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    memory_id = Column(PG_UUID(as_uuid=False), ForeignKey("memory_metadata.id", ondelete="CASCADE"), nullable=False)
    
    base_importance = Column(Float, nullable=False, default=0.5)
    confidence = Column(Float, nullable=False, default=0.8)
    contradicted = Column(Boolean, nullable=False, default=False)
    risk_factor = Column(Float, nullable=False, default=0.0)
    
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class MemoryCoactivationEdge(Base):
    """Lightweight co-activation graph (stats-based).
    
    Stores pairs of memories that are frequently retrieved together.
    Relationships are bidirectional but stored as directed (memory_id_a, memory_id_b).
    
    Edge weight formula: 1 - exp(-λ * coactivation_count)
    """

    __tablename__ = "memory_coactivation_edges"
    __table_args__ = (
        Index("ix_memory_coactivation_edges_org_a_b", "organization_id", "memory_id_a", "memory_id_b", unique=True),
        Index("ix_memory_coactivation_edges_org_a", "organization_id", "memory_id_a"),
        Index("ix_memory_coactivation_edges_org_b", "organization_id", "memory_id_b"),
    )

    id = Column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    organization_id = Column(PG_UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    memory_id_a = Column(PG_UUID(as_uuid=False), ForeignKey("memory_metadata.id", ondelete="CASCADE"), nullable=False)
    memory_id_b = Column(PG_UUID(as_uuid=False), ForeignKey("memory_metadata.id", ondelete="CASCADE"), nullable=False)
    
    coactivation_count = Column(Integer, nullable=False, default=0)
    edge_weight = Column(Float, nullable=False, default=0.0)
    last_coactivated_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class MemoryRetrievalExplanation(Base):
    """Stores auditable scoring breakdown for each retrieval (append-only log).
    
    Records:
    - Which user made the query
    - Query hash (to group similar queries)
    - Results array with per-memory scores and component breakdown
    - All 8 activation components: Rel, Rec, Freq, Imp, Conf, Ctx, Prov, Risk
    - Gating info (allowed/denied and reason)
    
    Partitioned by month optional (not enforced in schema for simplicity).
    """

    __tablename__ = "memory_retrieval_explanations"
    __table_args__ = (
        Index("ix_memory_retrieval_explanations_org_user", "organization_id", "user_id"),
        Index("ix_memory_retrieval_explanations_org_query", "organization_id", "query_hash"),
        Index("ix_memory_retrieval_explanations_org_timestamp", "organization_id", "retrieved_at"),
    )

    id = Column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    organization_id = Column(PG_UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(PG_UUID(as_uuid=False), nullable=False)
    
    query_hash = Column(String(64), nullable=False)
    retrieved_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    top_k = Column(Integer, nullable=False, default=10)
    
    # JSONB array of results
    # Each result object contains:
    # {
    #   "memory_id": str,
    #   "activation": float,
    #   "components": {
    #     "rel": float, "rec": float, "freq": float, "imp": float,
    #     "conf": float, "ctx": float, "prov": float, "risk": float, "nbr": float
    #   },
    #   "gating": {"allowed": bool, "reason": str or null},
    #   "rank": int
    # }
    results = Column(JSONB, nullable=False, default=list)
    
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class CausalHypothesis(Base):
    """Causal hypothesis tracking (causality as hypotheses, not truth).
    
    Stores proposed cause-effect relationships with:
    - relation: Type of relationship (causes, leads_to, blocks, resolves, correlates)
    - confidence: Confidence score (0-1)
    - evidence_memory_ids: List of supporting memory IDs
    - status: Lifecycle (proposed, active, contested, rejected)
    
    Links to episodes and events but doesn't require them (nullable).
    """

    __tablename__ = "causal_hypotheses"
    __table_args__ = (
        Index("ix_causal_hypotheses_org_episode", "organization_id", "episode_id"),
        Index("ix_causal_hypotheses_org_status", "organization_id", "status"),
        Index("ix_causal_hypotheses_org_created", "organization_id", "created_at"),
    )

    id = Column(PG_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    organization_id = Column(PG_UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    
    episode_id = Column(PG_UUID(as_uuid=False), nullable=True)
    from_event_id = Column(PG_UUID(as_uuid=False), nullable=True)
    to_event_id = Column(PG_UUID(as_uuid=False), nullable=True)
    
    # Relation type
    relation = Column(String(32), nullable=False)  # causes, leads_to, blocks, resolves, correlates
    
    confidence = Column(Float, nullable=False, default=0.5)
    
    # Array of UUID strings (evidence memory IDs)
    evidence_memory_ids = Column(ARRAY(PG_UUID(as_uuid=False)), nullable=True, default=list)
    
    # Status lifecycle
    status = Column(String(32), nullable=False, default="proposed")  # proposed, active, contested, rejected
    
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
