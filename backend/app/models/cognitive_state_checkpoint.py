"""Cognitive state checkpoint model for crash-tolerant loop restore (Phase 80)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CognitiveStateCheckpoint(Base, UUIDMixin):
    __tablename__ = "cognitive_state_checkpoints"

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

    checkpoint_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    loop_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    active_goal: Mapped[str] = mapped_column(Text, nullable=False, default="")

    completed_step_indices: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    pending_step_indices: Mapped[list[int]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    working_memory_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    last_output: Mapped[str] = mapped_column(Text, nullable=False, default="")

    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="active",
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_cognitive_state_checkpoints_session_seq", "session_id", "checkpoint_seq"),
        Index("ix_cognitive_state_checkpoints_org_created", "org_id", "created_at"),
    )
