"""Cognitive loop iteration model."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class CognitiveIteration(Base, UUIDMixin):
    __tablename__ = "cognitive_iterations"

    session_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    iteration_num: Mapped[int] = mapped_column(Integer, nullable=False)

    plan_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    execution_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    critique_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    evaluation: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc="pass|fail|retry|needs_evidence",
    )

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index(
            "ux_cognitive_iterations_session_iteration",
            "session_id",
            "iteration_num",
            unique=True,
        ),
        Index("ix_cognitive_iterations_session_eval", "session_id", "evaluation"),
    )
