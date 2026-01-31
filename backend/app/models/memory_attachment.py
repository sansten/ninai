"""
Memory Attachment Model
=======================

Stores metadata for files attached to a long-term memory.
Actual bytes are stored on disk under a configured attachments directory.

NOTE: This is an MVP "multimodal" layer (images/docs/etc). Later we can add
text extraction + embedding for retrieval.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
from datetime import datetime

from sqlalchemy import String, ForeignKey, BigInteger, Index, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base, UUIDMixin, TimestampMixin

if TYPE_CHECKING:
    from app.models.memory import MemoryMetadata


class MemoryAttachment(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_attachments"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    uploaded_by: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    storage_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        doc="Relative path under MEMORY_ATTACHMENTS_DIR",
    )

    indexed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When extracted text was embedded and indexed",
    )
    index_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Last indexing error (if any)",
    )

    memory: Mapped["MemoryMetadata"] = relationship(
        "MemoryMetadata",
        primaryjoin="MemoryAttachment.memory_id == MemoryMetadata.id",
        viewonly=True,
    )

    __table_args__ = (
        Index(
            "ix_memory_attachments_org_memory",
            "organization_id",
            "memory_id",
        ),
    )
