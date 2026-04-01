"""Load snapshot model for system cognitive load tracking (Phase 55)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LoadSnapshot(Base, UUIDMixin):
    __tablename__ = "load_snapshots"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    queue_depths: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    active_workers: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    load_level: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="low",
        index=True,
    )

    sampled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_load_snapshots_org_sampled", "org_id", "sampled_at"),
    )
