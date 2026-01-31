"""Agent run storage.

Implements the required agent_runs table described in AGENT_IMPLEMENTATION_GUIDE.md.
This records agent execution outcomes for observability + idempotency.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Float, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, UUIDMixin


class AgentRun(Base, UUIDMixin):
    __tablename__ = "agent_runs"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Organization this agent run belongs to",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Memory ID being processed",
    )

    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Agent name (stable identifier)",
    )

    agent_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Agent version",
    )

    inputs_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 hash of inputs + config used for idempotency",
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="success|retry|failed|skipped",
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        doc="Confidence score 0..1",
    )

    outputs: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Agent-specific outputs (JSON)",
    )

    warnings: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Warnings list",
    )

    errors: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Errors list",
    )

    started_at: Mapped[datetime] = mapped_column(
        nullable=False,
        doc="Start timestamp (UTC)",
    )

    finished_at: Mapped[datetime] = mapped_column(
        nullable=False,
        doc="Finish timestamp (UTC)",
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Trace/request/job correlation id",
    )

    provenance: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Citations/provenance used to produce outputs",
    )
    __table_args__ = (
        Index(
            "ux_agent_runs_idempotency",
            "organization_id",
            "memory_id",
            "agent_name",
            "agent_version",
            unique=True,
        ),
        Index("ix_agent_runs_lookup", "organization_id", "memory_id", "agent_name"),
    )
