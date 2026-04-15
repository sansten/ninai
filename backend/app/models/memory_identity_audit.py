"""
MemoryIdentityAudit Model
=========================

Stores the true actor identity for every memory write regardless of mode_applied.
RLS policy: only accessible to org_admin role.

This table exists for compliance/audit only — never returned in normal API responses.
Its existence allows soft anonymity: memory rows may show "anonymous" while this
table retains the real identity for regulated orgs.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class MemoryIdentityAudit(Base, UUIDMixin, TimestampMixin):
    """Soft-anonymity audit record per memory write. RLS-gated to org_admin."""

    __tablename__ = "memory_identity_audits"

    memory_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Memory this audit record belongs to",
    )

    org_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Organization (for RLS filtering)",
    )

    actual_actor_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Real actor_id regardless of mode_applied on the memory row",
    )

    actual_actor_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc='employee | bot | anonymous',
    )

    actual_role: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Actor role at write time (snapshot, not current)",
    )

    actual_department: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Actor department at write time (from AD/SCIM if enriched)",
    )

    mode_applied: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        doc='What was written to the memory row: full | role_only | anonymous',
    )

    mandate_was_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Whether org mandate was active when this memory was written",
    )

    identity_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Confidence score for identity resolution: 1.0=JWT+AD, 0.8=cache, 0.5=JWT fallback",
    )

    ad_enriched: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether identity was enriched from AD/SCIM",
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryIdentityAudit memory={self.memory_id[:8]} "
            f"actor={self.actual_actor_id} mode={self.mode_applied}>"
        )
