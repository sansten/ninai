"""Cognitive loop session model."""

from __future__ import annotations

from sqlalchemy import String, Text, Index, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class CognitiveSession(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "cognitive_sessions"

    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, index=True)
    agent_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True, index=True)

    goal_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("goals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="running|succeeded|failed|aborted",
    )

    goal: Mapped[str] = mapped_column(Text, nullable=False)

    context_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    trace_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    __table_args__ = (
        Index("ix_cognitive_sessions_org_status", "organization_id", "status"),
    )
