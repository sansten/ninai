"""Episode–Memory association table (GAP-1: Four-Level Hierarchy).

Many-to-many link between MemoryMetadata (messages, Level 1) and
MemoryEpisode (episodes, Level 2).  A message belongs to exactly one
episode under normal operation, but we allow soft multi-membership for
edge cases (partial overlap during re-segmentation).

The ``position`` column preserves temporal ordering of messages inside
the episode so downstream consumers can reconstruct the original thread.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryEpisodeMembership(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_episode_memberships"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization (tenant isolation)",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Memory (message) id",
    )

    episode_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_episodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Episode id",
    )

    position: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Ordinal position of this message within the episode (0-based)",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        default="system",
        nullable=False,
        doc="Creator identifier (system/agent/user)",
    )

    __table_args__ = (
        Index(
            "ux_memory_episode_membership",
            "organization_id",
            "memory_id",
            "episode_id",
            unique=True,
        ),
        Index("ix_memory_episode_membership_episode", "episode_id", "position"),
    )
