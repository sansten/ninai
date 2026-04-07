"""UsageEvent model for per-org daily usage counters."""

from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class UsageEvent(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "usage_events"

    date: Mapped[date] = mapped_column(Date, nullable=False)
    metric: Mapped[str] = mapped_column(String(100), nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        UniqueConstraint("organization_id", "date", "metric", name="uq_usage_event_org_date_metric"),
    )
