"""Playbook model (PR4: Procedural/Skill Memory).

Stores reusable procedural steps extracted from successful agent runs.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from sqlalchemy import Enum as SQLEnum, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class PlaybookScopeType(str, Enum):
    PERSONAL = "personal"
    TEAM = "team"
    DEPARTMENT = "department"
    DIVISION = "division"
    ORGANIZATION = "organization"


class Playbook(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "playbooks"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    scope_type: Mapped[PlaybookScopeType] = mapped_column(
        SQLEnum(PlaybookScopeType, native_enum=False),
        nullable=False,
        default=PlaybookScopeType.PERSONAL,
        index=True,
    )

    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(Text, nullable=False)

    problem_signature: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    signature_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    steps: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    constraints: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    success_rate: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_playbooks_org_scope", "organization_id", "scope_type"),
        Index("ix_playbooks_org_signature", "organization_id", "signature_hash"),
    )
