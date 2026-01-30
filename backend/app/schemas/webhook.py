"""Webhook schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, HttpUrl

from app.schemas.base import BaseSchema


class WebhookSubscriptionCreateRequest(BaseSchema):
    url: HttpUrl
    event_types: list[str] | None = None
    description: str | None = Field(default=None, max_length=500)


class WebhookSubscriptionResponse(BaseSchema):
    id: str
    url: str
    is_active: bool
    event_types: list[str]
    description: str | None = None
    created_at: datetime


class WebhookSubscriptionCreateResponse(WebhookSubscriptionResponse):
    secret: str


class WebhookDeliveryResponse(BaseSchema):
    id: str
    subscription_id: str
    event_type: str
    status: str  # pending, delivered, failed
    attempts: int
    next_attempt_at: datetime
    delivered_at: datetime | None = None
    last_http_status: int | None = None
    last_error: str | None = None
    created_at: datetime


class WebhookDeliveryHistoryResponse(BaseSchema):
    deliveries: list[WebhookDeliveryResponse]
    total: int
    pending_count: int
    failed_count: int
