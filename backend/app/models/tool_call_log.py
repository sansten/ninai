"""Tool call logs for Cognitive Loop execution."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class ToolCallLog(Base, UUIDMixin):
    __tablename__ = "tool_call_logs"

    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    iteration_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_iterations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    tool_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    tool_input: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    tool_output_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True, doc="success|denied|failed")
    denial_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_tool_call_logs_session_tool", "session_id", "tool_name"),
    )
