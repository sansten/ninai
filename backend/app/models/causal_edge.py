"""
Causal Edge Model (PR-1: Causal Memory Graph)
==============================================

Represents a directed causal relationship between two entities or facts.
Enables interventional reasoning: "What would happen if we changed X?"
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, Integer, String, Text, UUID, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CausalEdge(Base):
    """
    A causal edge: cause_entity → effect_entity
    
    Represents: "If we change cause_entity, it affects effect_entity"
    
    Strength scores reflect confidence in the causal relationship:
    - 0.0-0.3: Weak/speculative causation
    - 0.3-0.6: Moderate causation with some contradictions
    - 0.6-0.9: Strong causation, well-validated
    - 0.9-1.0: Very strong, nearly deterministic
    
    Mechanism types:
    - "direct": immediate cause-effect (A directly causes B)
    - "temporal": temporal precedence (A happens, then B happens)
    - "correlative": high correlation but mechanism unclear
    - "hypothetical": LLM-suggested causation, not yet validated
    """

    __tablename__ = "causal_edges"

    # Primary identifiers
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    organization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    # Entity references (can be MemoryFact ID, MemorySemanticNode ID, or free-text entity identifier)
    cause_entity_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    cause_entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="fact"
    )  # "fact" | "semantic_node" | "entity"
    effect_entity_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    effect_entity_type: Mapped[str] = mapped_column(String(50), nullable=False, default="fact")

    # Causal relationship details
    mechanism: Mapped[str] = mapped_column(
        String(50), nullable=False, default="correlative"
    )  # "direct" | "temporal" | "correlative" | "hypothetical"
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    # Confidence in this edge (0-1): how sure are we this causation exists?

    latency_hours: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Typical delay between cause and effect (hours). None = immediate or unknown.

    # Evidence tracking
    evidence_memory_ids: Mapped[List[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    # Memory IDs that support this causal edge
    counterfactual_evidence_ids: Mapped[List[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    # Memory IDs where "when we didn't do cause_entity, effect_entity didn't happen"

    # Validation counts
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_validated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    validation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # How many times has this edge been confirmed by new evidence?
    invalidation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # How many times has this edge been contradicted?

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Indexes for efficient queries
    __table_args__ = (
        Index(
            "idx_causal_edges_org_from",
            "organization_id",
            "cause_entity_id",
            "effect_entity_id",
        ),
        Index("idx_causal_edges_org_to", "organization_id", "effect_entity_id"),
        Index("idx_causal_edges_strength", "organization_id", "strength"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "cause_entity_id": self.cause_entity_id,
            "cause_entity_type": self.cause_entity_type,
            "effect_entity_id": self.effect_entity_id,
            "effect_entity_type": self.effect_entity_type,
            "mechanism": self.mechanism,
            "strength": self.strength,
            "latency_hours": self.latency_hours,
            "evidence_memory_ids": self.evidence_memory_ids or [],
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
            "validation_count": self.validation_count,
            "invalidation_count": self.invalidation_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<CausalEdge "
            f"{self.cause_entity_id} "
            f"--[{self.mechanism}:{self.strength:.2f}]--> "
            f"{self.effect_entity_id}>"
        )


class CounterfactualScenario(Base):
    """
    A hypothetical 'what-if' scenario for planning and learning.
    
    Used for:
    1. Planning: "What if we increase price by 10%?" → predict revenue impact
    2. Learning: "We didn't send reminder email → customer didn't renew" (validates causation)
    3. Explanation: "Service outage caused 500 customer complaints" (uses causal chain)
    """

    __tablename__ = "counterfactual_scenarios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    organization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    # Scenario definition
    scenario_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "prediction" | "causal_explanation" | "planning"
    condition: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # "if memory_id X = Y" or "if X increases by Z%"
    predicted_outcomes: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict
    )  # {fact_id_or_entity: probability}

    # Resolution (when scenario actually happens)
    actual_outcome: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    # When the counterfactual was actually observed, record the outcome
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # (1 - |predicted - actual|) / magnitude: how accurate was our prediction?

    # Lifecycle
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_counterfactual_org", "organization_id"),
        Index("idx_counterfactual_resolved", "organization_id", "resolved_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<CounterfactualScenario "
            f"{self.scenario_type} "
            f"condition: {self.condition[:50]} "
            f"accuracy: {self.accuracy or 'unresolved'}>"
        )
