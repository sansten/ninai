"""Webhook models.

Outgoing webhooks using an outbox + delivery tracking model.

- Subscriptions are org-scoped.
- Events are emitted from audit events (org-scoped).
- Deliveries are retried by Celery beat.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class WebhookSubscription(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "webhook_subscriptions"

    url: Mapped[str] = mapped_column(String(2000), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Empty list => all events
    event_types: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    # Stored encrypted; returned only once at creation.
    secret_encrypted: Mapped[str] = mapped_column(Text, nullable=False)

    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_webhook_subscriptions_org_active", "organization_id", "is_active"),
    )


class WebhookOutboxEvent(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "webhook_outbox_events"

    event_type: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_webhook_outbox_org_type", "organization_id", "event_type"),
    )


class WebhookDelivery(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "webhook_deliveries"

    subscription_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("webhook_subscriptions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    outbox_event_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("webhook_outbox_events.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_webhook_deliveries_due", "status", "next_attempt_at"),
        Index("ix_webhook_deliveries_org_sub", "organization_id", "subscription_id"),
    )
