"""SystemCognitionState model — per-org live cognitive state.

One row per org. Upserted by the cognitive heartbeat task every 5 minutes.
Records what the OS is currently attending to, how many anomalies are
unresolved, and what the next autonomous action is scheduled to be.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, generate_uuid, utc_now


class SystemCognitionState(Base, UUIDMixin):
    """Per-org cognitive state — the OS's live self-model.

    Tracks what the system is currently attending to, its cognitive load,
    and the outcome of the last heartbeat scan.
    """

    __tablename__ = "system_cognition_states"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        doc="Org this state belongs to (unique — one row per org)",
    )

    # --- Attention & focus ---
    current_focus: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Human-readable description of what the OS is currently attending to",
    )
    active_session_ids: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="UUIDs of cognitive loop sessions spawned by the last heartbeat",
    )

    # --- Load signals ---
    unresolved_anomalies_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of high-score anomalies with no loop session run on them",
    )
    stale_goals_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Active goals older than 2h with no recent loop session",
    )
    cognitive_load: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Normalised cognitive load 0.0–1.0",
    )

    # --- Scheduling ---
    last_heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        doc="Timestamp of the last heartbeat run for this org",
    )
    next_scheduled_action: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="Description of the next autonomous action the OS intends to take",
    )

    # --- Timestamps ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    __table_args__ = (
        UniqueConstraint("organization_id", name="uq_system_cognition_state_org"),
        Index("ix_system_cognition_states_org_id", "organization_id"),
        Index("ix_system_cognition_states_updated_at", "updated_at"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "current_focus": self.current_focus,
            "active_session_ids": self.active_session_ids or [],
            "unresolved_anomalies_count": self.unresolved_anomalies_count,
            "stale_goals_count": self.stale_goals_count,
            "cognitive_load": self.cognitive_load,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None,
            "next_scheduled_action": self.next_scheduled_action,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
