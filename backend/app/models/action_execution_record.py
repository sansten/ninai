"""Action Execution Record — Phase 47.

Persists every external action dispatched by the Autonomous Action Engine.
Provides a full audit trail: what was dispatched, by whom, under which
policy decision, and whether it succeeded.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class ActionExecutionRecord(Base, UUIDMixin, TimestampMixin):
    """One row per external action dispatched (or denied) by AutonomousActionAgent.

    Columns:
    - organization_id       — tenant scoping (RLS)
    - session_id            — cognitive session that triggered this action (nullable)
    - action_type           — "webhook" | "pagerduty" | "jira" | "slack" | "generic_rest"
    - connector_id          — registered connector id from ConnectorRegistry (nullable)
    - target_url            — destination endpoint (stored for audit, not raw payload)
    - payload_summary       — safe metadata summary of the dispatched payload (JSONB)
    - status                — "pending" | "success" | "failed" | "denied" | "rolled_back"
    - attempt_count         — how many dispatch attempts were made
    - http_status_code      — final HTTP response code (nullable if denied/no dispatch)
    - error_message         — last error string if status == "failed"
    - policy_decision       — "auto_approved" | "human_review_required" | "denied"
    - confidence_at_dispatch— enrichment confidence when decision was made
    - completed_at          — timestamp of final status transition
    """

    __tablename__ = "action_execution_records"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    session_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
        doc="Cognitive session that triggered this action.",
    )

    action_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="webhook | pagerduty | jira | slack | generic_rest",
    )

    connector_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Registered connector id from ConnectorRegistry.",
    )

    target_url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Destination endpoint URL.",
    )

    payload_summary: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Safe metadata summary of the dispatched payload.",
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
        doc="pending | success | failed | denied | rolled_back",
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Total dispatch attempts made.",
    )

    http_status_code: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Final HTTP response status code.",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Last error string if status == failed.",
    )

    policy_decision: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        doc="auto_approved | human_review_required | denied",
    )

    confidence_at_dispatch: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Enrichment confidence score when the dispatch decision was made.",
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of the final status transition.",
    )

    __table_args__ = (
        Index("ix_action_exec_org_status", "organization_id", "status"),
        Index("ix_action_exec_org_type", "organization_id", "action_type"),
    )
