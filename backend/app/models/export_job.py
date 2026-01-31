"""Export job model.

Implements medium-effort Phase2 "snapshot/export semantics" by tracking
async export runs that write bundles to disk.

Security:
- org-scoped via organization_id
- download is admin-only unless a signed token is provided
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import String, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base, UUIDMixin, TimestampMixin


class ExportJob(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "export_jobs"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_by_user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Job identity
    job_type: Mapped[str] = mapped_column(String(50), nullable=False, default="snapshot")
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        doc="queued|running|succeeded|failed",
    )

    # Output
    file_path: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    file_bytes: Mapped[int | None] = mapped_column(nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)

    # Error handling
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    __table_args__ = (
        Index("ix_export_jobs_org_status", "organization_id", "status"),
    )
