"""Billing endpoints for subscription status, checkout, and Stripe webhook."""

from __future__ import annotations

import hashlib
import hmac
import json
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.billing_service import BillingService

router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan: str
    seats: int
    email: str
    org_name: str


class SubscriptionResponse(BaseModel):
    plan: str
    status: str
    seat_limit: int
    seat_count: int
    current_period_end: str | None
    has_license: bool


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    svc = BillingService(db)
    sub = await svc.get_subscription(tenant.org_id)
    if not sub:
        return SubscriptionResponse(
            plan="community",
            status="active",
            seat_limit=0,
            seat_count=0,
            current_period_end=None,
            has_license=False,
        )

    return SubscriptionResponse(
        plan=sub.plan,
        status=sub.status,
        seat_limit=sub.seat_limit,
        seat_count=sub.seat_count,
        current_period_end=str(sub.current_period_end) if sub.current_period_end else None,
        has_license=bool(sub.license_token),
    )


@router.post("/checkout", status_code=status.HTTP_201_CREATED)
async def create_checkout(
    body: CheckoutRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    if body.plan not in ("enterprise_self", "enterprise_managed"):
        raise HTTPException(status_code=400, detail="Invalid plan")
    if body.seats < 10:
        raise HTTPException(status_code=400, detail="Minimum 10 seats")

    svc = BillingService(db)
    existing = await svc.get_subscription(tenant.org_id)
    if existing and existing.status == "active":
        raise HTTPException(status_code=409, detail="Active subscription exists")

    customer_id = await svc.create_stripe_customer(
        org_id=tenant.org_id,
        email=body.email,
        org_name=body.org_name,
    )
    sub = await svc.create_subscription(
        org_id=tenant.org_id,
        stripe_customer_id=customer_id,
        plan=body.plan,
        seats=body.seats,
        email=body.email,
    )
    await db.commit()
    return {"subscription_id": sub.id, "status": sub.status, "plan": sub.plan}


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(default=None, alias="stripe-signature"),
    db: AsyncSession = Depends(get_db),
):
    body = await request.body()
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    if secret and stripe_signature:
        timestamp = stripe_signature.split(",")[0].replace("t=", "")
        expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
        received = stripe_signature.split("v1=")[-1].split(",")[0]
        if not hmac.compare_digest(expected, received):
            raise HTTPException(status_code=400, detail="Invalid signature")

    event = json.loads(body)
    svc = BillingService(db)
    result = await svc.handle_webhook(event_type=event.get("type", ""), payload=event)
    await db.commit()
    return {"status": result}
