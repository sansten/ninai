"""Topic memberships.

Links a memory to one or more topics within an organization/scope.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryTopicMembership(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_topic_memberships"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this membership belongs to (tenant isolation)",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Memory id",
    )

    topic_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Topic id",
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether this is the primary topic",
    )

    weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
        doc="Optional membership weight (0..1)",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="agent",
        doc="Creator identifier (agent/system/user)",
    )

    __table_args__ = (
        Index(
            "ux_memory_topic_membership",
            "organization_id",
            "memory_id",
            "topic_id",
            unique=True,
        ),
        Index("ix_memory_topic_membership_lookup", "organization_id", "memory_id", "is_primary"),
    )
