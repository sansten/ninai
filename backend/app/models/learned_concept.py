"""Learned concept model for self-supervised concept clustering (Phase 59)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LearnedConcept(Base, UUIDMixin):
    __tablename__ = "learned_concepts"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    concept_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    member_memory_ids: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    canonical_terms: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    first_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        index=True,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
    )

    __table_args__ = (
        Index("ix_learned_concepts_org_name", "org_id", "concept_name"),
    )
