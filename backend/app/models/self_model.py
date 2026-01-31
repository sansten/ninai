"""SelfModel tables (profiles + events).

SelfModel tracks per-organization calibration signals used by Planner/Policy/Meta.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class SelfModelProfile(Base):
    __tablename__ = "self_model_profiles"

    # One row per org.
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        primary_key=True,
    )

    domain_confidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tool_reliability: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    agent_accuracy: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class SelfModelEvent(Base, UUIDMixin):
    __tablename__ = "self_model_events"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="tool_success|tool_failure|agent_corrected|agent_confirmed|policy_denial",
    )

    tool_name: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    agent_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    memory_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_self_model_events_org_type_created", "organization_id", "event_type", "created_at"),
    )
