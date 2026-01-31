"""Dead Letter Queue model for failed pipeline tasks.

Stores tasks that have exceeded maximum retry attempts or are identified
as poison messages that repeatedly fail and block queue processing.
"""

from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Text, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base, UUIDMixin, TimestampMixin, TenantMixin


class DeadLetterTask(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Dead letter queue for failed pipeline tasks."""

    __tablename__ = "dead_letter_queue"

    # Original task reference
    original_task_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Original pipeline task ID",
    )
    task_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of pipeline task",
    )
    
    # Failure details
    failure_reason = Column(
        String(100),
        nullable=False,
        comment="Reason for DLQ: max_retries_exceeded, poison_message, manual_quarantine",
    )
    total_attempts = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of attempts made",
    )
    last_error = Column(
        Text,
        nullable=True,
        comment="Last error message",
    )
    error_pattern = Column(
        String(200),
        nullable=True,
        comment="Detected error pattern (for poison messages)",
    )
    
    # Task payload (for potential replay)
    task_payload = Column(
        JSON,
        nullable=True,
        comment="Original task payload for manual review/replay",
    )
    task_metadata = Column(
        JSON,
        nullable=True,
        comment="Additional metadata",
    )
    
    # DLQ management
    quarantined_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.utcnow(),
        index=True,
        comment="When task was moved to DLQ",
    )
    reviewed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When task was reviewed by admin",
    )
    reviewed_by = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User ID who reviewed the task",
    )
    resolution = Column(
        String(50),
        nullable=True,
        comment="Resolution: requeued, discarded, fixed_and_requeued",
    )
    resolution_notes = Column(
        Text,
        nullable=True,
        comment="Admin notes on resolution",
    )
    is_resolved = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        comment="Whether task has been resolved",
    )
    
    # Priority for review
    review_priority = Column(
        Integer,
        default=5,
        nullable=False,
        comment="Priority for manual review (1-10, 10=critical)",
    )

    def __repr__(self):
        return (
            f"DeadLetterTask(id={self.id}, task_type={self.task_type}, "
            f"reason={self.failure_reason}, attempts={self.total_attempts})"
        )
