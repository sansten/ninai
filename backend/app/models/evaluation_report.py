"""Evaluation report for Cognitive Loop sessions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class EvaluationReport(Base, UUIDMixin):
    __tablename__ = "evaluation_reports"

    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    report: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    final_decision: Mapped[str] = mapped_column(String(20), nullable=False, index=True, doc="pass|fail|contested")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_evaluation_reports_session_created", "session_id", "created_at"),
    )
