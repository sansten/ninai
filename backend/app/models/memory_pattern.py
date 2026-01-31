"""Materialized patterns.

Stores reusable patterns detected by PatternDetectionAgent.
Patterns are scoped (personal/team/department/division/organization/global).
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryPattern(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "memory_patterns"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this pattern belongs to (tenant isolation)",
    )

    scope: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Scope: personal/team/department/division/organization/global",
    )

    scope_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Optional scope entity id",
    )

    scope_key: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        doc="Derived scope key for uniqueness",
    )

    pattern_key: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        doc="Stable normalized pattern identifier",
    )

    pattern_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Pattern type/kind (e.g. support/domain/structure)",
    )

    confidence: Mapped[float] = mapped_column(
        nullable=False,
        default=0.5,
        doc="Aggregate confidence (0..1)",
    )

    details: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Arbitrary details (rationale, examples, etc.)",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="agent",
        doc="Creator identifier (agent/system/user)",
    )

    __table_args__ = (
        Index(
            "ux_memory_patterns_scope_key",
            "organization_id",
            "scope_key",
            "pattern_key",
            unique=True,
        ),
        Index("ix_memory_patterns_lookup", "organization_id", "scope", "scope_id"),
    )
