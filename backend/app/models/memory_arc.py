"""Memory trajectory model (PR-2).

Tracks how memory strength/relevance evolves over time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, JSON, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class MemoryArc(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "memory_arcs"

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    measurements: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    trend: Mapped[str] = mapped_column(String(32), nullable=False, default="stable")
    trajectory_type: Mapped[str] = mapped_column(String(64), nullable=False, default="linear_decay")
    prediction_next_access: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "trend IN ('strengthening', 'stable', 'weakening', 'rediscovered')",
            name="ck_memory_arcs_trend",
        ),
        CheckConstraint(
            "trajectory_type IN ('exponential_decay', 'linear_decay', 'plateaued', 'recently_boosted')",
            name="ck_memory_arcs_trajectory_type",
        ),
    )
