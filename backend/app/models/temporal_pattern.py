"""Temporal pattern model for mined recurring time signals (Phase 62)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TemporalPattern(Base, UUIDMixin):
    __tablename__ = "temporal_patterns"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    pattern_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    pattern_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    topic_tags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_severity: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utc_now, server_default=func.now())
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    __table_args__ = (
        Index("ix_temporal_patterns_org_key", "org_id", "pattern_key"),
    )
