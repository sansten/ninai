"""Working memory item model for per-session active context buffering (Phase 61)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkingMemoryItem(Base, UUIDMixin):
    __tablename__ = "working_memory_items"

    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    memory_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    content_snapshot: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        default="",
    )

    item_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="memory",
        index=True,
    )

    activation: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        index=True,
    )

    inserted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_working_memory_items_session_activation", "session_id", "activation"),
    )
