"""Connector Registration — Phase 48.

Stores registered external system connectors per tenant.  Each row
represents one configured connection: target system, auth method, and
the set of action types it handles.

Credentials are NEVER stored here — only the credential_ref (e.g. a secret
manager key or env-var name) is persisted.  The actual secret is resolved at
dispatch time by ExternalConnectorService.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ConnectorRegistration(Base, UUIDMixin, TimestampMixin):
    """One registered external connector per tenant.

    Columns:
    - organization_id   — tenant scoping (RLS)
    - name              — human-readable label (e.g. "production-pagerduty")
    - connector_type    — "webhook" | "pagerduty" | "jira" | "slack" | "generic_rest"
    - target_url        — base URL or endpoint for this connector
    - credential_ref    — key name in secret store / env (not the secret itself)
    - headers_template  — static headers merged at dispatch time (JSONB)
    - event_types       — action types this connector handles (JSONB list of str)
    - is_active         — soft-disable without deleting
    - last_tested_at    — timestamp of most recent test ping
    - test_status       — "ok" | "failed" | "untested"
    """

    __tablename__ = "connector_registrations"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Human-readable connector label.",
    )

    connector_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="webhook | pagerduty | jira | slack | generic_rest",
    )

    target_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Base URL or endpoint for this connector.",
    )

    credential_ref: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Secret key name (not the secret value).",
    )

    headers_template: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Static headers merged at dispatch time.",
    )

    event_types: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Action types this connector handles.",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    last_tested_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    test_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="untested",
        doc="ok | failed | untested",
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id", "name",
            name="uq_connector_registration_org_name",
        ),
        Index("ix_connector_reg_org_type", "organization_id", "connector_type"),
        Index("ix_connector_reg_org_active", "organization_id", "is_active"),
    )
