"""Memory fact model (PR3: Facts + Contradictions).

Stores normalized subject-predicate-object facts extracted from memories.
Facts can be active, superseded, or disputed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, Enum as SQLEnum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryFactStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"


class MemoryFact(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_facts"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    subject: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    predicate: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    object: Mapped[str] = mapped_column(Text, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    source_memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    supersedes_fact_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_facts.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[MemoryFactStatus] = mapped_column(
        SQLEnum(MemoryFactStatus, native_enum=False),
        nullable=False,
        default=MemoryFactStatus.ACTIVE,
        index=True,
    )

    contradiction_group_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
    )

    __table_args__ = (
        Index(
            "ix_memory_facts_org_subject_pred_status",
            "organization_id",
            "subject",
            "predicate",
            "status",
            postgresql_using="btree",
        ),
    )
