"""Evaluation Run model for tracking eval suite executions."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Enum, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.drift_report import DriftReport
    from app.models.eval_suite import EvalSuite
    from app.models.organization import Organization


class EvalRun(Base):
    """Execution of an evaluation suite with computed metrics.
    
    Tracks: precision@k, recall@k, MRR, NDCG, cross-tenant leak rate,
    policy violations, stale recall rate, contradiction recall rate,
    topk Jaccard stability, and latency metrics.
    """

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    suite_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("eval_suites.id", ondelete="CASCADE"), nullable=False
    )
    
    started_at: Mapped[datetime] = mapped_column(nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    
    # Configuration used for this run (k values, thresholds, etc.)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    # Computed metrics: {precision_at_k, recall_at_k, mrr, ndcg, leak_rate, etc.}
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    status: Mapped[str] = mapped_column(
        Enum("running", "success", "failure", "cancelled", name="eval_run_status"),
        nullable=False,
        default="running"
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)

    # Relationships
    organization: Mapped[Organization] = relationship("Organization", back_populates="eval_runs")
    suite: Mapped[EvalSuite] = relationship("EvalSuite", back_populates="eval_runs")
    drift_reports_as_baseline: Mapped[list[DriftReport]] = relationship(
        "DriftReport",
        foreign_keys="DriftReport.baseline_run_id",
        back_populates="baseline_run",
    )
    drift_reports_as_current: Mapped[list[DriftReport]] = relationship(
        "DriftReport",
        foreign_keys="DriftReport.current_run_id",
        back_populates="current_run",
    )

    __table_args__ = (
        Index("ix_eval_runs_org_suite", "organization_id", "suite_id"),
        Index("ix_eval_runs_org_started", "organization_id", "started_at"),
        Index("ix_eval_runs_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<EvalRun(id={self.id}, suite={self.suite_id}, status={self.status})>"
