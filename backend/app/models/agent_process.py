"""Agent process table.

Represents schedulable agent work with priorities and quotas.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class AgentProcess(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "agent_processes"

    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
        doc="Optional cognitive session this process corresponds to",
    )

    agent_run_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
        doc="Optional agent_run record for observability linkage",
    )

    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Stable agent identifier",
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Larger is higher priority; FIFO within equal priority",
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="queued",
        doc="queued|running|blocked|succeeded|failed",
    )

    attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="How many times this process has been started",
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        doc="Maximum starts allowed before fail-closed",
    )

    quota_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Optional token budget for this process",
    )

    quota_storage_mb: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Optional storage budget (MB) for this process",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the process last moved to running",
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the process finished (any terminal state)",
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        doc="Trace/request correlation id",
    )

    last_error: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Last known error/denial reason",
    )

    process_metadata: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Additional scheduler metadata",
    )

    __table_args__ = (
        Index(
            "ix_agent_processes_org_status_prio",
            "organization_id",
            "status",
            "priority",
            "created_at",
        ),
        Index("ix_agent_processes_session_lookup", "session_id"),
    )
