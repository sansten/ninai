"""Strategy Library model for Outcome-Driven Strategy Learning.

Stores per-org EMA-smoothed success rates for (goal_type, domain) strategy fingerprints.
Used by StrategyLearningService to warm-start the planner with proven tool sequences.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class StrategyLibraryEntry(Base, UUIDMixin):
    __tablename__ = "strategy_library_entries"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Fingerprint key: goal_type + domain identifies a strategy cluster
    goal_type: Mapped[str] = mapped_column(String(100), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False, default="general")

    # Ordered list of tool names used in successful plans (e.g. ["memory.search", "memory.get"])
    tool_sequence: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Evidence topic categories that appeared in high-activation evidence cards
    evidence_pattern: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Aggregated planning metrics (EMA-smoothed)
    avg_plan_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    avg_iterations: Mapped[float] = mapped_column(Float, nullable=False, default=2.0)

    # EMA success rate (0.0–1.0); updated after each session with alpha=0.25
    success_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Total sessions ingested (used to gate retrieval: require >= 3)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id", "goal_type", "domain",
            name="uq_strategy_library_org_goaltype_domain",
        ),
    )
