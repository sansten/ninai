"""GDPR / Compliance Models.

Three models supporting right-to-be-forgotten, data retention policies,
and user consent tracking.

Tables:
  data_retention_policies  — per-org retention configuration
  user_data_deletions      — deletion requests with lifecycle tracking
  consent_records          — user consent grant / revoke history
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class DataRetentionPolicy(Base, UUIDMixin, TimestampMixin):
    """Per-organization data retention configuration.

    Columns:
    - organization_id         — tenant (unique: one policy per org)
    - retention_days_memory   — how long to keep memory records (default 365)
    - retention_days_audit    — how long to keep audit events (default 730)
    - auto_archive_after_days — inactivity threshold before auto-archive (default 90)
    - created_by_user_id      — admin who created the policy
    - is_active               — whether policy is enforced
    """

    __tablename__ = "data_retention_policies"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    retention_days_memory: Mapped[int] = mapped_column(
        Integer, nullable=False, default=365,
        doc="Days to retain memory records before eligible for purge",
    )

    retention_days_audit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=730,
        doc="Days to retain audit events (regulatory minimum often 2 years)",
    )

    auto_archive_after_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=90,
        doc="Days of inactivity before a memory is auto-archived",
    )

    created_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        doc="Whether this retention policy is currently enforced",
    )


class UserDataDeletion(Base, UUIDMixin, TimestampMixin):
    """Tracks GDPR right-to-be-forgotten deletion requests.

    Columns:
    - user_id               — subject of the deletion
    - organization_id       — org scope for the request
    - requested_by_user_id  — admin or self who initiated
    - reason                — stated reason for deletion
    - deletion_scope        — "account" | "org_data" | "all"
    - status                — "pending" | "processing" | "completed" | "failed"
    - deleted_at            — when deletion completed
    - error_detail          — failure message if status=failed
    """

    __tablename__ = "user_data_deletions"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    requested_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        doc="Reason provided for the deletion request",
    )

    deletion_scope: Mapped[str] = mapped_column(
        String(32), nullable=False, default="account",
        doc="account=anonymize user, org_data=erase org memories only, all=full cascade",
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True,
        doc="pending | processing | completed | failed",
    )

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        doc="Timestamp when deletion completed",
    )

    error_detail: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        doc="Error message if status=failed",
    )

    __table_args__ = (
        Index("ix_user_data_deletions_user_status", "user_id", "status"),
    )


class ConsentRecord(Base, UUIDMixin, TimestampMixin):
    """User consent grant and revoke history.

    Columns:
    - user_id       — consenting user
    - organization_id — org context
    - consent_type  — "data_processing" | "analytics" | "marketing"
    - granted       — True=granted, False=revoked
    - granted_at    — when consent was recorded
    - expires_at    — optional expiry date
    - revoked_at    — when consent was revoked (if applicable)
    - ip_address    — client IP at time of consent (for audit)
    """

    __tablename__ = "consent_records"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    consent_type: Mapped[str] = mapped_column(
        String(64), nullable=False,
        doc="data_processing | analytics | marketing",
    )

    granted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        doc="True=consent granted, False=consent revoked",
    )

    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.utcnow(),
    )

    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    ip_address: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
        doc="Client IP at time of consent record (for audit trail)",
    )

    __table_args__ = (
        Index("ix_consent_records_user_type", "user_id", "consent_type"),
    )
