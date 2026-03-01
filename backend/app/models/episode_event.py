"""Episode Event model for timeline tracking (PR1: Advanced Memory Features).

Events represent individual actions/milestones within an episode's lifecycle.
Each event is timestamped and can reference a memory or contain standalone content.

Schema aligned with Ninai_Advanced_Memory_Features_Copilot_Requirements.md
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class EpisodeEventType(str, Enum):
    """Type of event in episode timeline."""

    USER_REPORT = "user_report"
    AGENT_ACTION = "agent_action"
    TOOL_RESULT = "tool_result"
    RESOLUTION = "resolution"
    FOLLOWUP = "followup"
    NOTE = "note"


class EpisodeActorType(str, Enum):
    """Who/what created the event."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class EpisodeEvent(Base, UUIDMixin, TimestampMixin):
    """Timeline event within an episode.

    Represents a discrete action, milestone, or update in the episode's history.
    """

    __tablename__ = "episode_events"

    # ── Tenant isolation ────────────────────────────────────────────
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this event belongs to (RLS enforcement)",
    )

    # ── Episode linkage ─────────────────────────────────────────────
    episode_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Parent episode",
    )

    # ── Memory linkage (optional) ───────────────────────────────────
    memory_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Associated memory (if event originated from memory write)",
    )

    # ── Event metadata ──────────────────────────────────────────────
    event_type: Mapped[EpisodeEventType] = mapped_column(
        SQLEnum(EpisodeEventType, native_enum=False),
        nullable=False,
        index=True,
        doc="Type of event",
    )

    event_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="When the event occurred (business time, may differ from created_at)",
    )

    actor_type: Mapped[EpisodeActorType] = mapped_column(
        SQLEnum(EpisodeActorType, native_enum=False),
        nullable=False,
        doc="Who/what created this event",
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="User/agent ID if applicable",
    )

    # ── Event content ───────────────────────────────────────────────
    content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Human-readable event description/message",
    )

    payload: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        doc="Structured event data (tool results, metadata, etc.)",
    )

    # ── Indexes ─────────────────────────────────────────────────────
    __table_args__ = (
        Index(
            "ix_episode_events_episode_ts",
            "episode_id",
            "event_ts",
            postgresql_using="btree",
        ),
        Index(
            "ix_episode_events_org_ts",
            "organization_id",
            "event_ts",
            postgresql_using="btree",
        ),
        Index(
            "ix_episode_events_type",
            "event_type",
            postgresql_using="btree",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EpisodeEvent(id={self.id}, episode_id={self.episode_id}, "
            f"type={self.event_type}, ts={self.event_ts})>"
        )
