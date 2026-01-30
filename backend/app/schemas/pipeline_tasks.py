"""Pipeline task schemas for API requests and responses."""

from datetime import datetime
from typing import Optional
from pydantic import Field

from app.schemas.base import BaseSchema


class PipelineTaskResponse(BaseSchema):
    """Response schema for a single pipeline task."""
    
    id: str
    organization_id: str
    task_type: str
    status: str
    priority: int
    
    # SLA fields
    sla_deadline: Optional[datetime] = None
    sla_category: Optional[str] = None
    sla_remaining_ms: Optional[int] = None
    sla_breached: bool = False
    
    # Resource tracking
    estimated_tokens: Optional[int] = None
    actual_tokens: Optional[int] = None
    estimated_latency_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    
    # Backpressure
    blocks_on_task_id: Optional[str] = None
    blocked_by_quota: bool = False
    
    # Retry tracking
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    
    # Metadata
    metadata: Optional[dict] = None
    trace_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime


class PipelineTaskCreate(BaseSchema):
    """Request schema for creating a pipeline task."""
    
    task_type: str = Field(..., description="Task type: consolidation, critique, evaluation, feedback_loop, embedding_refresh")
    priority: int = Field(default=5, ge=1, le=10, description="Priority 1-10 (10=highest)")
    
    # SLA
    sla_category: Optional[str] = Field(None, description="SLA category: critical, high, medium, low")
    sla_deadline_minutes: Optional[int] = Field(None, ge=1, description="Minutes until SLA deadline")
    
    # Resource estimates
    estimated_tokens: Optional[int] = Field(None, ge=0, description="Estimated token consumption")
    estimated_latency_ms: Optional[int] = Field(None, ge=0, description="Estimated latency in ms")
    
    # Metadata
    metadata: Optional[dict] = None
    blocks_on_task_id: Optional[str] = Field(None, description="Task ID this depends on")


class PipelineTaskUpdatePriority(BaseSchema):
    """Request schema for updating task priority."""
    
    priority: int = Field(..., ge=1, le=10, description="New priority 1-10 (10=highest)")
    reason: Optional[str] = Field(None, description="Reason for priority change")


class PipelineStatsResponse(BaseSchema):
    """Response schema for pipeline queue statistics."""
    
    total_tasks: int = 0
    queued_tasks: int = 0
    running_tasks: int = 0
    blocked_tasks: int = 0
    succeeded_tasks_last_hour: int = 0
    failed_tasks_last_hour: int = 0
    
    # SLA metrics
    sla_breached_count: int = 0
    sla_compliance_rate: float = 0.0
    avg_queue_time_ms: Optional[float] = None
    avg_execution_time_ms: Optional[float] = None
    
    # Resource utilization
    total_tokens_consumed_last_hour: int = 0
    avg_tokens_per_task: Optional[float] = None
    
    # Queue depth by type
    queue_depth_by_type: dict[str, int] = {}
    
    # SLA breach by category
    sla_breach_by_category: dict[str, int] = {}
