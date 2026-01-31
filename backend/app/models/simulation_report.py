"""Simulation report persistence.

SimulationReports store deterministic SimulationAgent outputs for later auditing.

Security:
- Tenant isolation is enforced by Postgres RLS on organization_id.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class SimulationReport(Base, UUIDMixin):
    __tablename__ = "simulation_reports"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    session_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cognitive_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    memory_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    report: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_simulation_reports_org_created", "organization_id", "created_at"),
        Index("ix_simulation_reports_session_created", "session_id", "created_at"),
    )
