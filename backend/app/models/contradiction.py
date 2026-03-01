"""Contradiction model (PR3: Facts + Contradictions).

Stores conflicts between two facts for same subject/predicate but differing objects.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ContradictionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Contradiction(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "contradictions"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    fact_a: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_facts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    fact_b: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_facts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    reason: Mapped[str] = mapped_column(Text, nullable=False)

    severity: Mapped[ContradictionSeverity] = mapped_column(
        SQLEnum(ContradictionSeverity, native_enum=False),
        nullable=False,
        default=ContradictionSeverity.MEDIUM,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "ix_contradictions_org_created",
            "organization_id",
            "created_at",
            postgresql_using="btree",
        ),
    )
