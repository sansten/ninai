"""Sleep Cycle Report — Phase 40.

Records the outcome of a nightly memory sleep cycle run per tenant.
One row is written per organization per night after the pipeline completes.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class SleepCycleReport(Base, UUIDMixin, TimestampMixin):
    """Summary of a single nightly memory sleep cycle run.

    Columns:
    - organization_id    — tenant scoping (RLS)
    - run_date           — calendar date of the cycle (UTC)
    - memories_scanned   — total memories evaluated during the run
    - retained           — count with sleep_action == "retain"
    - strengthened       — count with sleep_action == "strengthen"
    - merged             — count with sleep_action == "merge"
    - pruned             — count with sleep_action == "prune"
    - associations_created — cross-links discovered during the creative-association pass
    - report_payload     — full per-action breakdown and metadata (JSONB)
    - completed_at       — timestamp when the pipeline finished
    """

    __tablename__ = "sleep_cycle_reports"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    run_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
        index=True,
        doc="Calendar date (UTC) when the sleep cycle ran",
    )

    memories_scanned: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total memories evaluated during the cycle",
    )

    retained: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Memories kept unchanged",
    )

    strengthened: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Memories boosted due to low freshness but high reference value",
    )

    merged: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Memories flagged for merge with near-duplicate candidates",
    )

    pruned: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Stale or highly redundant memories marked for archival",
    )

    associations_created: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Novel cross-memory association links discovered",
    )

    report_payload: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        doc="Full per-action breakdown and per-memory details (JSONB)",
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When the pipeline finished writing results",
    )

    __table_args__ = (
        Index(
            "uq_sleep_cycle_org_date",
            "organization_id",
            "run_date",
            unique=True,
        ),
    )
