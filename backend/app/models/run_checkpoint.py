"""Run checkpoint model (PR5: Replayability / Time-Travel Debugging).

Stores deterministic snapshots of agent run state at each step for replay/audit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, Integer
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


class RunCheckpoint(Base, UUIDMixin):
    __tablename__ = "run_checkpoints"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    agent_run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Sequential step index in the run (0-based)",
    )

    input_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Inputs at this step (query, params, etc.)",
    )

    retrieval_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Retrieved memories: ids, scores, filter info",
    )

    model_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Model state: config, temperature, etc.",
    )

    output_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Step output: response, reasoning, etc.",
    )

    created_at: Mapped[datetime] = mapped_column(
        nullable=False,
        doc="Checkpoint creation timestamp (UTC)",
    )

    __table_args__ = (
        Index("ix_run_checkpoints_org_run", "organization_id", "agent_run_id"),
        Index("ix_run_checkpoints_lookup", "agent_run_id", "step_index"),
    )
