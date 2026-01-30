"""Dead Letter Queue schemas."""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class DeadLetterTaskResponse(BaseModel):
    """Dead Letter Queue task response."""

    id: UUID
    organization_id: UUID
    original_task_id: UUID
    task_type: str
    failure_reason: str
    total_attempts: int
    last_error: Optional[str] = None
    error_pattern: Optional[str] = None
    task_payload: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    quarantined_at: datetime
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[UUID] = None
    resolution: Optional[str] = None
    resolution_notes: Optional[str] = None
    is_resolved: bool
    review_priority: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DeadLetterTaskRequeue(BaseModel):
    """Request to requeue a DLQ task."""

    notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Notes on why task is being requeued",
    )


class DeadLetterTaskDiscard(BaseModel):
    """Request to discard a DLQ task."""

    notes: str = Field(
        ...,
        max_length=500,
        description="Required notes on why task is being discarded",
    )


class DeadLetterQueueStats(BaseModel):
    """Dead Letter Queue statistics."""

    total_unresolved: int
    by_failure_reason: Dict[str, int]
    by_task_type: Dict[str, int]
    high_priority_count: int
