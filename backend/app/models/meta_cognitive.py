"""
PR-6: Meta-Cognitive Planning Models

Executive-function metadata that lets the agent reason about how to reason.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List

from sqlalchemy import Float, Index, Integer, String, TIMESTAMP, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class StrategySelected(str, Enum):
    """Reasoning strategy selected by the meta-cognitive controller."""

    HEURISTIC = "heuristic"
    DELIBERATIVE = "deliberative"
    MIXED = "mixed"
    ESCALATE = "escalate"


class CognitiveStrategy(Base, UUIDMixin, TimestampMixin):
    """
    Metadata about how the agent allocates cognitive resources for a query.
    """

    __tablename__ = "cognitive_strategies"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    query_id: Mapped[str] = mapped_column(String(255), nullable=False)
    complexity_estimated: Mapped[float] = mapped_column(Float, default=0.5)
    strategy_selected: Mapped[str] = mapped_column(
        String(50), default=StrategySelected.MIXED.value
    )
    retrieval_budget: Mapped[int] = mapped_column(Integer, default=20)
    reasoning_depth: Mapped[int] = mapped_column(Integer, default=2)
    verification_required: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.7)
    time_budget_seconds: Mapped[int] = mapped_column(Integer, default=30)
    expected_answer_quality: Mapped[float] = mapped_column(Float, default=0.7)
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    actual_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    strategy_effectiveness: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("idx_cognitive_strategies_org_query", "organization_id", "query_id"),
        Index("idx_cognitive_strategies_org_strategy", "organization_id", "strategy_selected"),
    )


class EpistemicState(Base, UUIDMixin):
    """
    Snapshot of what the agent knows, is uncertain about, and does not know.
    """

    __tablename__ = "epistemic_states"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    known_domains: Mapped[List[str]] = mapped_column(JSON, default=[])
    uncertain_domains: Mapped[List[str]] = mapped_column(JSON, default=[])
    unknown_domains: Mapped[List[str]] = mapped_column(JSON, default=[])
    confidence_calibration: Mapped[float | None] = mapped_column(Float, nullable=True)
    surprise_frequency: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("idx_epistemic_states_org_timestamp", "organization_id", "timestamp"),
    )
