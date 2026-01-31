"""Materialized Logseq exports.

Stores LogseqExportAgent outputs (markdown + graph) for a memory.

This table is tenant-scoped (organization_id) and protected by PostgreSQL RLS.
Persistence is additionally gated in application code (admin-only).
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryLogseqExport(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_logseq_exports"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this export belongs to (tenant isolation)",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Memory this export was derived from",
    )

    markdown: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Rendered Logseq Markdown",
    )

    graph: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Graph JSON (nodes/edges)",
    )

    item_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Number of exported items",
    )

    agent_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Agent version used to render this export",
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Trace/request ID associated with export generation",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="agent",
        doc="Creator identifier (agent/system/user)",
    )

    updated_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Admin user who allowed persistence (nullable)",
    )

    __table_args__ = (
        Index(
            "ux_memory_logseq_exports_org_memory",
            "organization_id",
            "memory_id",
            unique=True,
        ),
        Index("ix_memory_logseq_exports_lookup", "organization_id", "memory_id"),
    )
