"""Human-in-the-loop review requests for knowledge item versions."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class KnowledgeReviewStatus:
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class KnowledgeReviewRequest(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "knowledge_review_requests"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this request belongs to (tenant isolation)",
    )

    item_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("knowledge_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    item_version_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("knowledge_item_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=KnowledgeReviewStatus.PENDING,
        server_default=KnowledgeReviewStatus.PENDING,
        index=True,
        doc="pending|approved|rejected",
    )

    requested_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="User who submitted the request",
    )

    reviewed_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Admin who reviewed the request",
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when reviewed",
    )

    decision_comment: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Optional reviewer comment",
    )

    __table_args__ = (
        Index("ix_knowledge_review_requests_org_status", "organization_id", "status"),
        Index("ix_knowledge_review_requests_org_item", "organization_id", "item_id"),
    )
