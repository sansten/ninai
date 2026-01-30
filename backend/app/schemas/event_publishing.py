"""
Schemas for Event Publishing and Batch Operations
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime


# ============================================================================
# EVENT SCHEMAS
# ============================================================================

class EventResponse(BaseModel):
    """Event response schema."""
    id: str
    event_type: str
    event_version: int
    organization_id: str
    resource_type: str
    resource_id: str
    payload: Optional[dict] = None
    actor_user_id: Optional[str] = None
    actor_agent_id: Optional[str] = None
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    created_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EventListResponse(BaseModel):
    """List of events."""
    events: List[EventResponse]
    total: int
    limit: int
    offset: int


# ============================================================================
# WEBHOOK SCHEMAS
# ============================================================================

class WebhookSubscriptionCreate(BaseModel):
    """Create webhook subscription request."""
    url: HttpUrl = Field(..., description="Webhook endpoint URL (must be HTTPS)")
    event_types: str = Field(default="*", description="CSV or '*' for all")
    resource_types: Optional[str] = Field(None, description="CSV of resource types")
    secret: Optional[str] = Field(None, description="Secret for signing (auto-generated if not provided)")
    max_retries: int = Field(default=5, ge=0, le=10)
    retry_delay_seconds: int = Field(default=60, ge=10)
    rate_limit_per_minute: Optional[int] = Field(None, ge=1)
    description: Optional[str] = Field(None, max_length=512)
    custom_headers: Optional[dict] = Field(None)


class WebhookSubscriptionUpdate(BaseModel):
    """Update webhook subscription request."""
    url: Optional[HttpUrl] = None
    event_types: Optional[str] = None
    resource_types: Optional[str] = None
    active: Optional[bool] = None
    paused_reason: Optional[str] = None
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    rate_limit_per_minute: Optional[int] = Field(None, ge=1)
    description: Optional[str] = None
    custom_headers: Optional[dict] = None


class WebhookSubscriptionResponse(BaseModel):
    """Webhook subscription response."""
    id: str
    organization_id: str
    url: str
    event_types: str
    resource_types: Optional[str]
    active: bool
    paused_at: Optional[datetime]
    paused_reason: Optional[str]
    max_retries: int
    retry_delay_seconds: int
    rate_limit_per_minute: Optional[int]
    delivered_count: int
    failed_count: int
    last_delivery_at: Optional[datetime]
    last_error: Optional[str]
    description: Optional[str]
    custom_headers: Optional[dict]
    created_at: datetime
    created_by_user_id: Optional[str]
    updated_at: datetime

    class Config:
        from_attributes = True


class WebhookSubscriptionListResponse(BaseModel):
    """List of webhook subscriptions."""
    subscriptions: List[WebhookSubscriptionResponse]
    total: int
    limit: int
    offset: int


class WebhookTestRequest(BaseModel):
    """Test webhook delivery."""
    subscription_id: str = Field(..., description="Subscription ID to test")


class WebhookTestResponse(BaseModel):
    """Test webhook response."""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: int
    error: Optional[str] = None


# ============================================================================
# BATCH OPERATION SCHEMAS
# ============================================================================

class BatchUpdateMemoryRequest(BaseModel):
    """Bulk update memory items."""
    memory_ids: List[str] = Field(..., min_items=1, max_items=1000)
    tags: Optional[List[str]] = None
    is_starred: Optional[bool] = None
    status: Optional[str] = None
    metadata: Optional[dict] = None


class BatchDeleteMemoryRequest(BaseModel):
    """Bulk delete memory items."""
    memory_ids: List[str] = Field(..., min_items=1, max_items=1000)
    soft_delete: bool = Field(default=True)


class BatchShareMemoryRequest(BaseModel):
    """Bulk share memory items."""
    memory_ids: List[str] = Field(..., min_items=1, max_items=1000)
    shared_with_user_ids: Optional[List[str]] = None
    shared_with_team_ids: Optional[List[str]] = None
    access_level: str = Field(default="view", pattern="^(view|edit|admin)$")


class BatchOperationResult(BaseModel):
    """Result of a batch operation."""
    operation_type: str  # update, delete, share
    resource_type: str  # memory, knowledge
    total_items: int
    successful: int
    failed: int
    errors: dict[str, str] = Field(default_factory=dict)
    duration_seconds: float


class BatchUpdateKnowledgeRequest(BaseModel):
    """Bulk update knowledge items."""
    knowledge_ids: List[str] = Field(..., min_items=1, max_items=1000)
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    is_published: Optional[bool] = None
    metadata: Optional[dict] = None


class BatchDeleteKnowledgeRequest(BaseModel):
    """Bulk delete knowledge items."""
    knowledge_ids: List[str] = Field(..., min_items=1, max_items=1000)
    soft_delete: bool = Field(default=True)


# ============================================================================
# SNAPSHOT/EXPORT SCHEMAS
# ============================================================================

class SnapshotCreateRequest(BaseModel):
    """Create a snapshot/export."""
    resource_type: str = Field(..., pattern="^(memory|knowledge)$")
    format: str = Field(default="json", pattern="^(json|csv|jsonl|pdf|parquet)$")
    name: Optional[str] = None
    filters: Optional[dict] = None
    include_deleted: bool = False
    include_unpublished: bool = False
    expires_in_days: int = Field(default=30, ge=1, le=365)


class SnapshotResponse(BaseModel):
    """Snapshot/export response."""
    id: str
    organization_id: str
    name: str
    description: Optional[str]
    format: str
    resource_type: Optional[str]
    filters: Optional[dict]
    storage_path: str
    size_bytes: int
    item_count: int
    status: str  # pending, processing, completed, failed
    progress_percent: int
    error_message: Optional[str]
    is_public: bool
    download_token: Optional[str]
    expires_at: Optional[datetime]
    is_archived: bool
    checksum: Optional[str]
    created_by_user_id: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class SnapshotListResponse(BaseModel):
    """List of snapshots."""
    snapshots: List[SnapshotResponse]
    total: int
    limit: int
    offset: int


class SnapshotDownloadResponse(BaseModel):
    """Response for snapshot download."""
    url: str
    expires_in_seconds: int
    format: str
    checksum: str
