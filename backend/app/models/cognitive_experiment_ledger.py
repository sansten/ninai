"""Experiment ledger for bounded cognitive self-experimentation."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CognitiveExperimentLedger(Base, UUIDMixin):
    __tablename__ = "cognitive_experiment_ledger"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    parameter_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    baseline_value: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_value: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_score: Mapped[float] = mapped_column(Float, nullable=False)
    candidate_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_delta: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="reverted", index=True)
    benchmark_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    rationale: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_cognitive_experiment_ledger_org_param", "org_id", "parameter_key"),
    )