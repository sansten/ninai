"""Feature readiness configuration — Phase 28.

Persists per-tenant, per-agent readiness state and admin parameter overrides.
Allows the UI to grey out features whose data conditions are not yet met, and
lets admins override thresholds or force-enable/disable individual agents.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class FeatureReadinessConfig(Base, UUIDMixin, TimestampMixin):
    """Per-tenant, per-agent feature readiness record.

    Columns:
    - organization_id  — tenant scoping (RLS)
    - agent_name       — identifies which enrichment agent this row covers
    - enabled          — admin override: False forces feature off regardless of readiness
    - custom_params    — JSONB overrides for agent default params (e.g. thresholds)
    - readiness_state  — JSONB snapshot from last evaluation
    - evaluated_at     — timestamp of last readiness evaluation
    """

    __tablename__ = "feature_readiness_config"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this config belongs to (tenant isolation)",
    )

    agent_name: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        doc="Enrichment agent name (e.g. EntityResolutionAgent)",
    )

    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Admin override — False disables agent UI regardless of readiness",
    )

    custom_params: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        doc="Param overrides applied on top of agent defaults (JSONB)",
    )

    readiness_state: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        default=None,
        doc="Latest readiness evaluation snapshot (JSONB)",
    )

    evaluated_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="Timestamp of the last readiness evaluation",
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Optional admin note (reason for override etc.)",
    )

    __table_args__ = (
        Index(
            "uq_feature_readiness_org_agent",
            "organization_id",
            "agent_name",
            unique=True,
        ),
    )
