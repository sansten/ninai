"""Materialized topics.

Stores topic labels per organization + scope.

This is the persistence layer for TopicModelingAgent outputs.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryTopic(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_topics"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this topic belongs to (tenant isolation)",
    )

    scope: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Scope: personal/team/department/division/organization/global",
    )

    scope_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Optional scope entity id (e.g. team_id when scope=team)",
    )

    scope_key: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        doc="Derived scope key for uniqueness (e.g. team:<uuid>, organization:)",
    )

    label: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        doc="Display label for the topic",
    )

    label_normalized: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        doc="Normalized label for stable uniqueness",
    )

    keywords: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Optional topic keywords (string list)",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="agent",
        doc="Creator identifier (agent/system/user)",
    )

    __table_args__ = (
        Index(
            "ux_memory_topics_scope_label",
            "organization_id",
            "scope_key",
            "label_normalized",
            unique=True,
        ),
        Index("ix_memory_topics_lookup", "organization_id", "scope", "scope_id"),
    )
