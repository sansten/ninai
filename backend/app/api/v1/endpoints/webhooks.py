"""Webhook management endpoints (admin-only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.webhook import WebhookSubscription
from app.models.webhook import WebhookDelivery, WebhookOutboxEvent
from app.schemas.webhook import (
    WebhookSubscriptionCreateRequest,
    WebhookSubscriptionCreateResponse,
    WebhookSubscriptionResponse,
    WebhookDeliveryResponse,
    WebhookDeliveryHistoryResponse,
)
from app.services.webhook_service import WebhookService


router = APIRouter()


@router.get("/webhooks", response_model=list[WebhookSubscriptionResponse])
async def list_webhooks(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(
        select(WebhookSubscription)
        .where(WebhookSubscription.organization_id == tenant.org_id)
        .order_by(WebhookSubscription.created_at.desc())
    )
    subs = res.scalars().all()

    return [
        WebhookSubscriptionResponse(
            id=s.id,
            url=s.url,
            is_active=s.is_active,
            event_types=s.event_types,
            description=s.description,
            created_at=s.created_at,
        )
        for s in subs
    ]


@router.post("/webhooks", response_model=WebhookSubscriptionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    body: WebhookSubscriptionCreateRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = WebhookService(db)
    sub, secret = await svc.create_subscription(
        organization_id=tenant.org_id,
        url=str(body.url),
        event_types=body.event_types,
        description=body.description,
    )
    await db.commit()

    return WebhookSubscriptionCreateResponse(
        id=sub.id,
        url=sub.url,
        is_active=sub.is_active,
        event_types=sub.event_types,
        description=sub.description,
        created_at=sub.created_at,
        secret=secret,
    )


@router.delete("/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(
        select(WebhookSubscription).where(
            WebhookSubscription.id == webhook_id,
            WebhookSubscription.organization_id == tenant.org_id,
        )
    )
    sub = res.scalar_one_or_none()
    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")

    await db.delete(sub)
    await db.commit()
    return None


@router.get("/webhooks/{webhook_id}/deliveries", response_model=WebhookDeliveryHistoryResponse)
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = 50,
    offset: int = 0,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Get delivery history for a webhook subscription."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    # Verify webhook exists and belongs to org
    res = await db.execute(
        select(WebhookSubscription).where(
            WebhookSubscription.id == webhook_id,
            WebhookSubscription.organization_id == tenant.org_id,
        )
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")

    # Get delivery counts
    total_res = await db.execute(
        select(func.count(WebhookDelivery.id)).where(
            WebhookDelivery.subscription_id == webhook_id,
        )
    )
    total = total_res.scalar() or 0

    pending_res = await db.execute(
        select(func.count(WebhookDelivery.id)).where(
            WebhookDelivery.subscription_id == webhook_id,
            WebhookDelivery.status == "pending",
        )
    )
    pending_count = pending_res.scalar() or 0

    failed_res = await db.execute(
        select(func.count(WebhookDelivery.id)).where(
            WebhookDelivery.subscription_id == webhook_id,
            WebhookDelivery.status == "failed",
        )
    )
    failed_count = failed_res.scalar() or 0

    # Get deliveries
    deliveries_res = await db.execute(
        select(WebhookDelivery)
        .where(WebhookDelivery.subscription_id == webhook_id)
        .order_by(WebhookDelivery.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    deliveries = deliveries_res.scalars().all()

    # Get event types for these deliveries
    outbox_ids = {d.outbox_event_id for d in deliveries}
    events_res = await db.execute(
        select(WebhookOutboxEvent).where(WebhookOutboxEvent.id.in_(outbox_ids))
    )
    events_by_id = {e.id: e for e in events_res.scalars().all()}

    return WebhookDeliveryHistoryResponse(
        deliveries=[
            WebhookDeliveryResponse(
                id=d.id,
                subscription_id=d.subscription_id,
                event_type=events_by_id.get(d.outbox_event_id, WebhookOutboxEvent()).event_type or "unknown",
                status=d.status,
                attempts=d.attempts,
                next_attempt_at=d.next_attempt_at,
                delivered_at=d.delivered_at,
                last_http_status=d.last_http_status,
                last_error=d.last_error,
                created_at=d.created_at,
            )
            for d in deliveries
        ],
        total=total,
        pending_count=pending_count,
        failed_count=failed_count,
    )
