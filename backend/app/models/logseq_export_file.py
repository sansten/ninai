"""Logseq write-to-disk export records.

Captures metadata about admin-only Logseq exports written to disk via the
Logseq API endpoint.

This table is tenant-scoped (organization_id) and protected by PostgreSQL RLS.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class LogseqExportFile(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "logseq_export_files"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this export belongs to (tenant isolation)",
    )

    relative_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        doc="Relative path returned to callers (no absolute host paths)",
    )

    bytes_written: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Bytes written to disk",
    )

    requested_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Admin user who initiated the export",
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Request/trace ID for correlation",
    )

    options: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Export options used (memory_ids/include_short_term/scope/limit)",
    )

    __table_args__ = (
        Index("ux_logseq_export_files_org_path", "organization_id", "relative_path", unique=True),
        Index("ix_logseq_export_files_lookup", "organization_id", "requested_by_user_id", "created_at"),
    )
