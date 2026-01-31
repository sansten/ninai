"""Pipeline task model for consolidation, critique, and evaluation workflows.

Pipeline tasks represent scheduled work items in the unified pipeline queue.
They support SLA-based ordering, backpressure, and rate limiting per tenant.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum as PyEnum

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.models.base import Base, UUIDMixin, TimestampMixin, TenantMixin


class PipelineTaskType(PyEnum):
    """Pipeline task type enum."""

    CONSOLIDATION = "consolidation"  # Memory consolidation/compression
    CRITIQUE = "critique"  # Agent output critique/validation
    EVALUATION = "evaluation"  # Multi-turn evaluation
    FEEDBACK_LOOP = "feedback_loop"  # User feedback processing
    EMBEDDING_REFRESH = "embedding_refresh"  # Vector embedding updates


class PipelineTaskStatus(PyEnum):
    """Pipeline task status enum."""

    QUEUED = "queued"
    RUNNING = "running"
    BLOCKED = "blocked"  # Backpressure/dependency block
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PipelineTask(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Pipeline task for consolidation, critique, and evaluation pipelines."""

    __tablename__ = "pipeline_tasks"

    # Task identification
    task_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Task type: consolidation, critique, evaluation, feedback_loop, embedding_refresh",
    )
    status = Column(
        String(20),
        default=PipelineTaskStatus.QUEUED.value,
        nullable=False,
        index=True,
        comment="Current status: queued, running, blocked, succeeded, failed",
    )

    # SLA and priority
    priority = Column(
        Integer,
        default=0,
        nullable=False,
        index=True,
        comment="Priority for SLA ordering (higher = sooner)",
    )
    sla_deadline = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="SLA deadline for this task",
    )
    sla_category = Column(
        String(50),
        nullable=True,
        comment="SLA category: critical, high, normal, low",
    )

    # Execution tracking
    attempts = Column(Integer, default=0, nullable=False, comment="Current attempt number")
    max_attempts = Column(Integer, default=3, nullable=False, comment="Max retry attempts")
    started_at = Column(
        DateTime(timezone=True), nullable=True, comment="When execution started"
    )
    finished_at = Column(
        DateTime(timezone=True), nullable=True, comment="When execution finished"
    )
    duration_ms = Column(Integer, nullable=True, comment="Execution duration in milliseconds")

    # Resource tracking
    estimated_tokens = Column(Integer, default=0, nullable=False, comment="Est. token cost")
    estimated_latency_ms = Column(
        Integer, default=0, nullable=False, comment="Est. latency in ms"
    )
    actual_tokens = Column(Integer, nullable=True, comment="Actual token cost")
    actual_latency_ms = Column(Integer, nullable=True, comment="Actual latency in ms")

    # Task data
    input_session_id = Column(
        String(100), nullable=False, comment="Source cognitive session ID"
    )
    target_resource_id = Column(
        String(100), nullable=False, comment="Target resource (memory/run/session ID)"
    )
    task_metadata = Column(JSON, default={}, nullable=False, comment="Task-specific metadata")

    # Backpressure and dependencies
    blocks_on_task_id = Column(
        String(100),
        nullable=True,
        comment="Task ID that blocks this one (dependency tracking)",
    )
    blocked_by_quota = Column(
        Boolean, default=False, nullable=False, comment="Is this blocked by quota?"
    )
    last_error = Column(Text, nullable=True, comment="Error message from last failure")

    # Indices for efficient querying
    __table_args__ = (
        # Note: Indices are created by migration; not declaring here to avoid
        # duplicate index errors in test environment
    )

    @property
    def sla_remaining_ms(self) -> int:
        """Calculate remaining time until SLA deadline in milliseconds."""
        if self.sla_deadline:
            remaining = self.sla_deadline - datetime.now(timezone.utc)
            return max(0, int(remaining.total_seconds() * 1000))
        return 0

    @property
    def sla_breached(self) -> bool:
        """Check if SLA deadline has been breached."""
        if self.sla_deadline and self.status not in (
            PipelineTaskStatus.SUCCEEDED.value,
            PipelineTaskStatus.FAILED.value,
        ):
            return datetime.now(timezone.utc) > self.sla_deadline
        return False

    def __repr__(self) -> str:
        return (
            f"PipelineTask(id={self.id}, type={self.task_type}, status={self.status}, "
            f"priority={self.priority}, sla_remaining_ms={self.sla_remaining_ms})"
        )
