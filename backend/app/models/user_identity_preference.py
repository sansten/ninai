"""
UserIdentityPreference Model
=============================

Per-user identity attribution preference. Overridden by OrgIdentityPolicy mandate.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin, TimestampMixin


class IdentityMode(str, Enum):
    """Identity attribution modes available to users."""

    FULL = "full"
    """actor_id + role + department stored on memory row."""

    ROLE_ONLY = "role_only"
    """role + department only; actor_id omitted from memory row."""

    ANONYMOUS = "anonymous"
    """Nothing stored on memory row (audit table may still capture if org requires)."""


class UserIdentityPreference(Base, UUIDMixin, TimestampMixin):
    """Per-user identity attribution preference. Overridden by org mandate."""

    __tablename__ = "user_identity_preferences"

    user_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        doc="User this preference belongs to",
    )

    org_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        doc="Organization scope for this preference",
    )

    preference: Mapped[str] = mapped_column(
        String(20),
        default=IdentityMode.FULL.value,
        nullable=False,
        doc='Identity mode: full | role_only | anonymous',
    )

    changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="When this preference was last changed",
    )

    changed_by: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        doc="Who changed this preference (admin override scenario)",
    )

    def __repr__(self) -> str:
        return f"<UserIdentityPreference user={self.user_id} preference={self.preference}>"
