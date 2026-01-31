"""
Webhook Subscription Model - For event delivery

Subscriptions define where and which events to send webhooks.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Boolean, Integer, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as SQLA_UUID
from app.core.database import Base


class WebhookSubscription(Base):
    """
    Webhook subscription - defines where events are sent.
    
    Subscriptions are:
    - Org-scoped (tied to organization)
    - Event-type selective (can filter by event_type)
    - Signed for security
    - Retryable with exponential backoff
    - Rate-limited to prevent abuse
    """
    __tablename__ = "webhook_subscriptions"

    # Identification
    id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Scoping
    organization_id: Mapped[str] = mapped_column(SQLA_UUID(as_uuid=False), nullable=False, index=True)
    
    # Webhook endpoint
    url: Mapped[str] = mapped_column(String(512), nullable=False)  # HTTPS required in production
    
    # Event filtering
    event_types: Mapped[str] = mapped_column(String(512), nullable=False, default="*")  # CSV or "*" for all
    resource_types: Mapped[str | None] = mapped_column(String(256), nullable=True)  # CSV: memory, knowledge, etc.
    
    # Security
    secret: Mapped[str] = mapped_column(String(256), nullable=False)  # For HMAC signing
    signing_algorithm: Mapped[str] = mapped_column(String(32), default="sha256", nullable=False)
    
    # Status and control
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    paused_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # When manually paused
    paused_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Retry configuration
    max_retries: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    retry_delay_seconds: Mapped[int] = mapped_column(Integer, default=60, nullable=False)
    
    # Rate limiting
    rate_limit_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)  # None = unlimited
    
    # Metrics
    delivered_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_delivery_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Metadata
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    custom_headers: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # Extra headers to send
    
    # Lifecycle
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_by_user_id: Mapped[str | None] = mapped_column(SQLA_UUID(as_uuid=False), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Indices
    __table_args__ = (
        Index("idx_webhook_org_active", "organization_id", "active"),
        Index("idx_webhook_created", "created_at"),
    )

    def __repr__(self):
        return f"<WebhookSubscription id={self.id} url={self.url} active={self.active}>"
