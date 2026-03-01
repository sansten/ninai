"""Episode model for case continuity (PR1: Advanced Memory Features).

Episodes enable case histories (e.g., support tickets, research threads, legal cases)
so users don't repeat context. This is distinct from memory_episodes which handle
message-level conversational grouping.

Schema aligned with Ninai_Advanced_Memory_Features_Copilot_Requirements.md
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class EpisodeScopeType(str, Enum):
    """Scope of episode visibility."""

    PERSONAL = "personal"
    TEAM = "team"
    DEPARTMENT = "department"
    DIVISION = "division"
    ORGANIZATION = "organization"


class EpisodeStatus(str, Enum):
    """Episode lifecycle status."""

    OPEN = "open"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class Episode(Base, UUIDMixin, TimestampMixin):
    """Case/ticket/thread for preserving context across interactions.

    Examples: support cases, research threads, legal cases, customer incidents.
    """

    __tablename__ = "episodes"

    # ── Tenant isolation ────────────────────────────────────────────
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this episode belongs to (RLS enforcement)",
    )

    # ── Scope and ownership ─────────────────────────────────────────
    scope_type: Mapped[EpisodeScopeType] = mapped_column(
        SQLEnum(EpisodeScopeType, native_enum=False),
        nullable=False,
        default=EpisodeScopeType.PERSONAL,
        doc="Visibility scope: personal, team, department, division, organization",
    )

    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
        doc="ID of team/department/division if scoped (NULL for personal/org)",
    )

    owner_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="User who owns/initiated this episode",
    )

    # ── Episode metadata ────────────────────────────────────────────
    episode_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        doc="Type: support_case, research_thread, legal_case, customer_incident, etc.",
    )

    status: Mapped[EpisodeStatus] = mapped_column(
        SQLEnum(EpisodeStatus, native_enum=False),
        nullable=False,
        default=EpisodeStatus.OPEN,
        index=True,
        doc="Lifecycle status",
    )

    title: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Episode title/subject",
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="LLM-generated summary (async updated by episode_summarizer_task)",
    )

    # ── Timestamps ──────────────────────────────────────────────────
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="When episode was initiated",
    )

    last_event_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="Most recent event timestamp (for active episode sorting)",
    )

    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When episode was resolved/closed",
    )

    # ── Structured metadata ─────────────────────────────────────────
    tags: Mapped[Optional[list[str]]] = mapped_column(
        ARRAY(String),
        nullable=True,
        doc="Tags for categorization/filtering",
    )

    entities: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        doc="Extracted entities (customer_id, device_id, account_number, etc.)",
    )

    # ── Indexes ─────────────────────────────────────────────────────
    __table_args__ = (
        Index(
            "ix_episodes_org_last_event",
            "organization_id",
            "last_event_at",
            postgresql_using="btree",
        ),
        Index(
            "ix_episodes_tags",
            "tags",
            postgresql_using="gin",
        ),
        Index(
            "ix_episodes_entities",
            "entities",
            postgresql_using="gin",
        ),
        Index(
            "ix_episodes_status_org",
            "status",
            "organization_id",
            postgresql_using="btree",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Episode(id={self.id}, type={self.episode_type}, "
            f"status={self.status}, title={self.title[:30] if self.title else None})>"
        )
