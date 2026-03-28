"""Intelligence Digest Report — P46.

Records the outcome of a scheduled daily/weekly intelligence digest
run per tenant. One row per organization per run_date per digest_type.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Index, Integer, String, Text
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class DigestReport(Base, UUIDMixin, TimestampMixin):
    """Summary of a scheduled intelligence digest run.

    Columns:
    - organization_id  — tenant scoping (RLS)
    - run_date         — calendar date of the digest (UTC)
    - digest_type      — "daily" | "weekly"
    - memories_total   — total active memories at time of digest
    - memories_flagged — pending human review queue items
    - anomalies_count  — anomalies detected in the digest window
    - goals_active     — number of active goals
    - goals_completed  — goals completed in the window
    - top_insights     — top 3 enrichment outputs by confidence (JSONB list)
    - quality_trend    — 7-day avg quality delta (JSONB)
    - narrative        — heuristic or LLM-generated summary paragraph
    - delivered_at     — when digest was delivered via webhook
    - completed_at     — when pipeline finished building this digest
    """

    __tablename__ = "digest_reports"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    run_date: Mapped[date] = mapped_column(
        Date, nullable=False, index=True,
        doc="Calendar date (UTC) when the digest ran",
    )

    digest_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="daily",
        doc="daily | weekly",
    )

    memories_total: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Total active memories at time of digest",
    )

    memories_flagged: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Pending human review queue items",
    )

    anomalies_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Anomalies detected in the digest window",
    )

    goals_active: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Active goals at time of digest",
    )

    goals_completed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Goals completed within the digest window",
    )

    top_insights: Mapped[Optional[list]] = mapped_column(
        JSONB, nullable=True,
        doc="Top 3 enrichment outputs by confidence: [{title, summary, agent, confidence}]",
    )

    quality_trend: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True,
        doc="Quality trend: {avg_7d: float, delta: float, direction: str}",
    )

    narrative: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        doc="Human-readable digest summary (heuristic template or LLM-generated)",
    )

    delivered_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        doc="When digest was delivered via webhook",
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        doc="When the pipeline finished building this digest",
    )

    __table_args__ = (
        Index(
            "uq_digest_report_org_date_type",
            "organization_id",
            "run_date",
            "digest_type",
            unique=True,
        ),
    )
