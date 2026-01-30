"""Knowledge base item versions.

Each version is immutable content (for auditability and rollback).
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class KnowledgeItemVersion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "knowledge_item_versions"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this version belongs to (tenant isolation)",
    )

    item_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Parent knowledge item",
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Monotonic version number per item",
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Optional title override for the version",
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Full content for this version (markdown/plaintext)",
    )

    extra_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Arbitrary metadata for routing/rendering",
    )

    created_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="User who authored/submitted this version",
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Request/trace correlation id",
    )

    provenance: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Optional provenance/citations supporting this version",
    )

    __table_args__ = (
        Index("ux_knowledge_item_versions_item_vn", "item_id", "version_number", unique=True),
        Index("ix_knowledge_item_versions_org_item", "organization_id", "item_id"),
    )
