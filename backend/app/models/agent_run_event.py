"""Agent run event storage.

This table captures step-level trajectory events for agent executions.
Examples: plan step, tool call, tool result, intermediate summary, final outcome.

The long-term intent is to make trajectories queryable and learnable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Index, func, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, utc_now


class AgentRunEvent(Base, UUIDMixin):
    __tablename__ = "agent_run_events"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Organization this event belongs to",
    )

    agent_run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Agent run this event belongs to",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Memory ID being processed",
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Event type (plan_step|tool_call|tool_result|summary|final|error|custom)",
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Monotonic step index within an agent run",
    )

    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Event payload (JSON)",
    )

    summary_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        doc="Human-readable summary text for retrieval/search",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        server_default=func.now(),
        doc="Event timestamp (UTC)",
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Trace/request/job correlation id",
    )

    __table_args__ = (
        Index(
            "ix_agent_run_events_lookup",
            "organization_id",
            "agent_run_id",
            "step_index",
        ),
        Index(
            "ix_agent_run_events_memory_lookup",
            "organization_id",
            "memory_id",
            "created_at",
        ),
    )
