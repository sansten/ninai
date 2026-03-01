"""Memory consolidation session model (PR-2).

Represents a single offline "sleep" cycle that performs memory quality
operations such as merging, pruning, and connection discovery.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class ConsolidationSession(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "consolidation_sessions"

    session_type: Mapped[str] = mapped_column(String(32), nullable=False, default="triggered")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="in_progress")

    operations: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    memory_quality_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    memory_quality_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "session_type IN ('nightly', 'weekly', 'triggered')",
            name="ck_consolidation_sessions_type",
        ),
        CheckConstraint(
            "status IN ('in_progress', 'completed', 'failed')",
            name="ck_consolidation_sessions_status",
        ),
    )
