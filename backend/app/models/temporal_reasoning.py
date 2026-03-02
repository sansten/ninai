"""
PR-5: Temporal Reasoning Engine Models

Temporal fact validity intervals, event sequences, and trajectory analysis.
Enables trend analysis, forecasting, and time-based planning.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from sqlalchemy import Index, String, Float, Integer, DateTime, ForeignKey, JSON, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class TemporalChangetype(str, Enum):
    """Types of temporal changes."""
    ONSET = "onset"  # Fact starts being true
    OFFSET = "offset"  # Fact stops being true
    STABLE = "stable"  # Fact remains true
    TRANSIENT = "transient"  # Fact briefly true then false


class SequenceType(str, Enum):
    """Types of temporal sequences."""
    EVENT_SEQUENCE = "event_sequence"
    TRAJECTORY = "trajectory"
    CYCLE = "cycle"


class PatternType(str, Enum):
    """Pattern types in sequences."""
    ESCALATION = "escalation"  # Increasing severity/frequency
    RESOLUTION = "resolution"  # Decreasing to stable point
    OSCILLATION = "oscillation"  # Regular cycles
    TREND = "trend"  # Consistent directional change
    ANOMALOUS = "anomalous"  # Unexpected deviation


class TrendDirection(str, Enum):
    """Direction of trend in trajectory."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLIC = "cyclic"


class TemporalFact(Base, UUIDMixin, TimestampMixin):
    """
    Fact with temporal validity interval.
    
    Example:
    - fact_id: memory fact about "product X costs $50"
    - valid_from: 2026-01-01
    - valid_to: 2026-03-15 (price changed)
    - change_type: offset (price no longer $50)
    """
    __tablename__ = "temporal_facts"
    
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    fact_id: Mapped[str] = mapped_column(String(255), nullable=False)
    valid_from: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    valid_to: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)
    confidence_at_time: Mapped[float] = mapped_column(Float, default=0.8)
    change_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    __table_args__ = (
        Index("idx_temporal_facts_org_fact", "organization_id", "fact_id"),
        Index("idx_temporal_facts_valid_range", "valid_from", "valid_to"),
    )


class TemporalSequence(Base, UUIDMixin, TimestampMixin):
    """
    Ordered sequence of events/facts for pattern detection and prediction.
    
    Example:
    - entities: [issue_reported_id, support_assigned_id, user_satisfied_id]
    - temporal_gaps: [300, 7200]  # 5 min, then 2 hours
    - pattern_type: resolution
    - pattern_strength: 0.87  # This sequence is reliable
    """
    __tablename__ = "temporal_sequences"
    
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    sequence_type: Mapped[str] = mapped_column(String(50), default="event_sequence")
    entities: Mapped[List[str]] = mapped_column(JSON, default=[])
    temporal_gaps: Mapped[List[int]] = mapped_column(JSON, default=[])
    pattern_type: Mapped[str] = mapped_column(String(50), nullable=False)
    pattern_strength: Mapped[float] = mapped_column(Float, default=0.5)
    predicted_next_event: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, default=1)
    
    __table_args__ = (
        Index("idx_temporal_sequences_org_type", "organization_id", "pattern_type"),
        Index("idx_temporal_sequences_strength", "organization_id", "pattern_strength"),
    )


class TemporalTrajectory(Base, UUIDMixin, TimestampMixin):
    """
    How a quantity changes over time.
    
    Example:
    - entity_id: user_123
    - quantity: sentiment_score (or memory_strength, task_completion_rate)
    - measurements: [{timestamp, value}, ...]
    - trend_direction: decreasing
    - seasonality: {period: 604800, amplitude: 0.15}  # weekly cycle, 15% variance
    """
    __tablename__ = "temporal_trajectories"
    
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    entity_id: Mapped[str] = mapped_column(String(255), nullable=False)
    quantity: Mapped[str] = mapped_column(String(100), nullable=False)
    measurements: Mapped[List[Dict]] = mapped_column(JSON, default=[])
    trend_direction: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    trend_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_future: Mapped[Optional[List[Dict]]] = mapped_column(JSON, nullable=True)
    inflection_points: Mapped[List[Dict]] = mapped_column(JSON, default=[])
    seasonality: Mapped[Optional[Dict]] = mapped_column(JSON, nullable=True)
    
    __table_args__ = (
        Index("idx_temporal_trajectories_org_entity", "organization_id", "entity_id"),
        Index("idx_temporal_trajectories_quantity", "organization_id", "quantity"),
    )
