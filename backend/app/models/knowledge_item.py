"""Knowledge base items.

Represents a governed, publishable knowledge artifact (procedure / playbook / policy).
Items are tenant-scoped via organization_id.

HITL workflow:
- Contributors submit a new `KnowledgeItemVersion`.
- Admins approve/reject via `KnowledgeReviewRequest`.
- The item points at a published version for read-only consumption.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class KnowledgeItem(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "knowledge_items"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this item belongs to (tenant isolation)",
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        doc="Human-readable title",
    )

    key: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Optional stable key (unique per org) used for programmatic references",
    )

    is_published: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        doc="Whether the item currently has a published version",
    )

    published_version_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("knowledge_item_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Currently published version id",
    )

    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when current published version was set",
    )

    __table_args__ = (
        Index("ux_knowledge_items_org_key", "organization_id", "key", unique=True),
        Index("ix_knowledge_items_org_title", "organization_id", "title"),
    )
