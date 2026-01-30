"""Memory feedback model.

Stores user-provided feedback signals about a memory (e.g., tag corrections,
classification overrides). Feedback is tenant-scoped and protected by RLS.
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime

from sqlalchemy import String, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, UUIDMixin, TimestampMixin


class MemoryFeedback(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_feedback"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this feedback belongs to",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Memory the feedback applies to",
    )

    actor_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
        doc="User who submitted feedback",
    )

    feedback_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Feedback type (tag_add/tag_remove/classification_override/entity_add/entity_remove/note/relevance)",
    )

    target_agent: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Optional agent name this feedback targets",
    )

    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Feedback payload (JSON)",
    )

    is_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Whether this feedback has been applied to memory metadata",
    )

    applied_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When feedback was applied",
    )

    applied_by: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="User (or system) that applied the feedback",
    )

    __table_args__ = (
        Index("ix_memory_feedback_lookup", "organization_id", "memory_id", "is_applied"),
        Index("ix_memory_feedback_actor", "organization_id", "actor_id", "created_at"),
    )
