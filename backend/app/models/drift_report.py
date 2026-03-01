"""Drift Report model for detecting memory quality regressions."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import Enum, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.eval_run import EvalRun
    from app.models.organization import Organization


class DriftReport(Base):
    """Drift detection report comparing eval run metrics over time.
    
    Flags performance regressions (precision drops, increased leak rates,
    reduced stability, etc.) with severity levels for alerting.
    """

    __tablename__ = "drift_reports"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    baseline_run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    current_run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False
    )
    
    # Delta metrics: {precision_delta, recall_delta, leak_rate_delta, ...}
    delta: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    severity: Mapped[str] = mapped_column(
        Enum("none", "low", "medium", "high", "critical", name="drift_severity"),
        nullable=False,
        default="none"
    )
    
    # Flagged issues: ["precision_drop", "leak_rate_increase", "stability_drop"]
    flagged_issues: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)

    # Relationships
    organization: Mapped[Organization] = relationship("Organization", back_populates="drift_reports")
    baseline_run: Mapped[EvalRun] = relationship(
        "EvalRun",
        foreign_keys=[baseline_run_id],
        back_populates="drift_reports_as_baseline",
    )
    current_run: Mapped[EvalRun] = relationship(
        "EvalRun",
        foreign_keys=[current_run_id],
        back_populates="drift_reports_as_current",
    )

    __table_args__ = (
        Index("ix_drift_reports_org_severity", "organization_id", "severity"),
        Index("ix_drift_reports_org_created", "organization_id", "created_at"),
        Index("ix_drift_reports_baseline", "baseline_run_id"),
        Index("ix_drift_reports_current", "current_run_id"),
    )

    def __repr__(self) -> str:
        return f"<DriftReport(id={self.id}, severity={self.severity}, org={self.organization_id})>"
