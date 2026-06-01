"""Materialized state-space snapshots for deterministic read-time access."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryStateSnapshot(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_state_snapshots"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    scope_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    symbolic_state: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    salience_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    staleness_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_memory_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
    )
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("organization_id", "scope_type", "scope_key", name="uq_memory_state_snapshot_scope"),
        Index("ix_memory_state_snapshots_org_scope", "organization_id", "scope_type", "scope_key"),
    )
