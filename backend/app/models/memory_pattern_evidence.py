"""Pattern evidence.

Links a memory to a detected pattern with evidence snippets.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryPatternEvidence(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_pattern_evidence"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this evidence belongs to (tenant isolation)",
    )

    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Memory id",
    )

    pattern_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_patterns.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Pattern id",
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
        doc="Per-memory confidence (0..1)",
    )

    evidence: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Evidence snippets/markers",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="agent",
        doc="Creator identifier",
    )

    __table_args__ = (
        Index(
            "ux_memory_pattern_evidence",
            "organization_id",
            "memory_id",
            "pattern_id",
            unique=True,
        ),
        Index("ix_memory_pattern_evidence_lookup", "organization_id", "memory_id"),
    )
