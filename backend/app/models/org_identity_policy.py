"""
OrgIdentityPolicy Model
=======================

Per-org identity attribution policy controlled by org admins.
Defines mandates and allowed modes for actor identity on memory writes.
"""

from __future__ import annotations

from typing import List

from sqlalchemy import Boolean, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class OrgIdentityPolicy(Base, UUIDMixin, TimestampMixin):
    """Per-org identity attribution policy controlled by org_admin."""

    __tablename__ = "org_identity_policies"

    org_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        doc="Organization this policy applies to",
    )

    mandate_actor_identity: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="If True, FULL attribution always — no individual user can opt out",
    )

    allowed_modes: Mapped[list] = mapped_column(
        JSONB,
        default=lambda: ["full", "role_only", "anonymous"],
        nullable=False,
        doc='Allowed modes when mandate is False: "full" | "role_only" | "anonymous"',
    )

    enrich_from_directory: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether to include AD-enriched department/location in memory rows",
    )

    audit_trail_always: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether soft-anonymity audit trail is enabled (identity stored in audit table even when memory shows anonymous)",
    )

    def __repr__(self) -> str:
        return f"<OrgIdentityPolicy org={self.org_id} mandate={self.mandate_actor_identity}>"
