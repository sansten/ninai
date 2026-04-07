"""CognitiveSchedule model (Phase 84).

User-defined scheduled cognitive actions that fire on a cron schedule.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class CognitiveSchedule(Base, UUIDMixin, TenantMixin, TimestampMixin):
    """A user-defined scheduled cognitive action for an organization."""

    __tablename__ = "cognitive_schedules"

    cron_expression: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="5-field standard cron expression (minute hour dom month dow)",
    )
    cognitive_verb: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Cognitive verb to invoke (analyze|summarize|plan|report|monitor|escalate|acknowledge)",
    )
    payload: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Arbitrary JSON payload passed to the cognitive verb at runtime",
    )
    label: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Human-readable label for the schedule",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether the schedule is currently enabled",
    )
    next_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="UTC timestamp of the next scheduled execution",
    )
    last_run_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="UTC timestamp of the most recent execution",
    )
