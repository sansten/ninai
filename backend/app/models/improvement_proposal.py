"""Improvement proposal model for recursive self-improvement (Phase 60)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ImprovementProposal(Base, UUIDMixin):
    __tablename__ = "improvement_proposals"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    target_agent: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    proposal_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )

    evidence: Mapped[list[dict]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    expected_gain: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="proposed",
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_improvement_proposals_org_target", "org_id", "target_agent"),
    )
