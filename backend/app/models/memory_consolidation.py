"""Memory consolidation record model.

Stores a durable record of a consolidation operation that produced a consolidated
summary memory from a set of source memories.

This is separate from the in-place "dedupe into primary" workflow handled by
`ConsolidationService`.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import ForeignKey, String, Text, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class MemoryConsolidation(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "memory_consolidations"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    consolidated_memory_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Stored as JSON for cross-database compatibility (tests may run without PG ARRAY).
    source_memory_ids: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)

    created_by: Mapped[str] = mapped_column(String(50), nullable=False, default="manual")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_memory_consolidations_org_created_at", "organization_id", "created_at"),
    )
