"""Organization-level feedback learning configuration.

Materializes FeedbackLearningAgent calibration outputs into a tenant-scoped table.

This table is protected by PostgreSQL RLS (organization_id = current org).
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class OrgFeedbackLearningConfig(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "org_feedback_learning_config"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this config belongs to (tenant isolation)",
    )

    updated_thresholds: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Calibrated heuristic thresholds (JSON)",
    )

    stopwords: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Learned stopwords (list of strings)",
    )

    heuristic_weights: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Optional heuristic weights (JSON)",
    )

    calibration_delta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Last calibration delta payload (JSON)",
    )

    updated_by_user_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="User who triggered/approved calibration (nullable)",
    )

    last_source_memory_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Memory ID that triggered the last calibration run (nullable)",
    )

    last_agent_version: Mapped[Optional[str]] = mapped_column(
        String(length=50),
        nullable=True,
        doc="Agent version that produced the last config update",
    )

    last_trace_id: Mapped[Optional[str]] = mapped_column(
        String(length=255),
        nullable=True,
        doc="Trace/request ID for the last calibration (nullable)",
    )

    __table_args__ = (
        Index("ux_org_feedback_learning_config_org", "organization_id", unique=True),
    )
