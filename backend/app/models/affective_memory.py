"""
PR-8: Emotional & Affective Memory Models

Models for tracking emotional valence, intensity, and trajectory of memories and interactions.
Enables empathetic, emotion-aware responses and interaction management.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Float, Index, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class EmotionalTag(str, Enum):
    """Categorical tags for emotional content."""

    FRUSTRATION = "frustration"
    JOY = "joy"
    FEAR = "fear"
    SATISFACTION = "satisfaction"
    ANXIETY = "anxiety"
    SURPRISE = "surprise"
    CONFUSION = "confusion"
    RELIEF = "relief"
    ANGER = "anger"
    DISAPPOINTMENT = "disappointment"


class EmotionalTrend(str, Enum):
    """Trajectory direction of emotional state."""

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


class DeEscalationStrategy(str, Enum):
    """Strategies for managing escalating emotional situations."""

    EMPATHY = "empathy"
    SLOW_DOWN = "slow_down"
    INVOLVE_EXPERT = "involve_expert"
    BREAK = "break"
    VALIDATE = "validate"
    CLARIFY = "clarify"


class AffectiveMemory(Base, UUIDMixin, TimestampMixin):
    """
    Emotional valence and significance of a memory.
    
    Tracks:
    - Valence: -1.0 (very negative) to +1.0 (very positive)
    - Arousal: 0 (calm) to 1 (intense/excited)
    - Emotional tags: categorical labels
    - Significance: how emotionally important is this memory?
    """

    __tablename__ = "affective_memories"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        # FK to memory records - made optional to handle archived/deleted memories
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
    )

    valence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Emotional tone: -1.0 (negative) to +1.0 (positive)",
    )

    arousal: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Emotional intensity: 0 (calm) to 1 (intense)",
    )

    emotional_tags: Mapped[List[str]] = mapped_column(
        JSON,
        default=[],
        comment="Categorical emotional labels (frustration, joy, fear, etc.)",
    )

    significance: Mapped[float] = mapped_column(
        Float,
        default=0.5,
        comment="How emotionally important is this memory? 0-1",
    )

    associated_user_ids: Mapped[List[str]] = mapped_column(
        JSON,
        default=[],
        comment="Which users were involved in this emotional moment?",
    )

    measured_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)

    confidence_in_measurement: Mapped[float] = mapped_column(
        Float,
        default=0.8,
        comment="How confident are we in this analysis? 0-1",
    )

    __table_args__ = (
        Index("idx_affective_org_memory", "organization_id", "memory_id"),
        Index("idx_affective_org_time", "organization_id", "measured_at"),
        Index("idx_affective_valence", "organization_id", "valence"),
    )


class EmotionalTrajectory(Base, UUIDMixin, TimestampMixin):
    """
    Trajectory of a user's emotional state over time.
    
    Tracks:
    - Time-series measurements of valence and arousal
    - Trend direction and strength
    - Escalation risk and de-escalation strategies
    - Current emotional state snapshot
    """

    __tablename__ = "emotional_trajectories"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    measurements: Mapped[List[dict]] = mapped_column(
        JSON,
        default=[],
        comment="Time-series: [{timestamp, valence, arousal, tags}]",
    )

    trend: Mapped[str] = mapped_column(
        String(50),
        default=EmotionalTrend.STABLE.value,
        comment="Direction: improving | stable | deteriorating",
    )

    current_state: Mapped[dict] = mapped_column(
        JSON,
        default={},
        comment="Latest emotional state: {valence, arousal, tags, timestamp}",
    )

    escalation_risk: Mapped[float] = mapped_column(
        Float,
        default=0.1,
        comment="Probability of escalation (anger, give-up, complaint): 0-1",
    )

    de_escalation_strategies: Mapped[List[str]] = mapped_column(
        JSON,
        default=[],
        comment="Recommended strategies: empathy, slow_down, expert, break, validate",
    )

    is_at_risk: Mapped[bool] = mapped_column(Boolean, default=False)

    last_measured_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)

    __table_args__ = (
        Index("idx_emotional_org_user", "organization_id", "user_id"),
        Index("idx_emotional_risk", "organization_id", "is_at_risk"),
        Index("idx_emotional_trend", "organization_id", "trend"),
    )


class EmotionalInteractionEvent(Base, UUIDMixin, TimestampMixin):
    """
    Record of a specific interaction with emotional context.
    
    Used for training emotion detection and understanding patterns in escalation.
    """

    __tablename__ = "emotional_interaction_events"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    interaction_content: Mapped[str] = mapped_column(String, nullable=False)

    initial_valence: Mapped[float] = mapped_column(Float, nullable=False)
    initial_arousal: Mapped[float] = mapped_column(Float, nullable=False)

    final_valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_arousal: Mapped[float | None] = mapped_column(Float, nullable=True)

    agent_response_tone: Mapped[str] = mapped_column(String(255), nullable=True)
    de_escalation_applied: Mapped[str] = mapped_column(String(50), nullable=True)

    was_escalation: Mapped[bool] = mapped_column(Boolean, default=False)
    was_de_escalated: Mapped[bool] = mapped_column(Boolean, default=False)

    outcome: Mapped[str] = mapped_column(
        String(50),
        nullable=True,
        comment="success | partial_success | failure | escalated",
    )

    __table_args__ = (
        Index("idx_interaction_org_user", "organization_id", "user_id"),
        Index("idx_interaction_escalation", "organization_id", "was_escalation"),
    )
