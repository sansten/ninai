"""BillingService for Stripe-backed org subscription management."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.license_token import sign_license_token
from app.models.org_subscription import OrgSubscription

_STRIPE_API_KEY = os.getenv("STRIPE_SECRET_KEY", "")
_LICENSE_PRIVATE_KEY = os.getenv("LICENSE_PRIVATE_KEY_PEM", "").encode()

PLAN_FEATURES: dict[str, list[str]] = {
    "community": [],
    "enterprise_self": [
        "enterprise.admin_ops",
        "enterprise.autoevalbench",
        "enterprise.drift_detection",
        "enterprise.resource_control",
        "enterprise.scim",
        "enterprise.governance_dashboard",
        "enterprise.meta_monitoring",
    ],
    "enterprise_managed": [
        "enterprise.admin_ops",
        "enterprise.autoevalbench",
        "enterprise.drift_detection",
        "enterprise.resource_control",
        "enterprise.scim",
        "enterprise.governance_dashboard",
        "enterprise.meta_monitoring",
        "enterprise.managed_hosting",
    ],
}


class BillingService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_subscription(self, org_id: str) -> OrgSubscription | None:
        res = await self.session.execute(
            select(OrgSubscription).where(OrgSubscription.organization_id == org_id)
        )
        return res.scalar_one_or_none()

    async def create_stripe_customer(self, *, org_id: str, email: str, org_name: str) -> str:
        if not _STRIPE_API_KEY:
            return f"cus_mock_{org_id[:8]}"

        import stripe  # type: ignore

        stripe.api_key = _STRIPE_API_KEY
        customer = stripe.Customer.create(
            email=email,
            name=org_name,
            metadata={"org_id": org_id, "platform": "ninai"},
        )
        return customer["id"]

    async def create_subscription(
        self,
        *,
        org_id: str,
        stripe_customer_id: str,
        plan: str,
        seats: int,
        email: str,
    ) -> OrgSubscription:
        from uuid import uuid4

        stripe_sub_id = await self._create_stripe_subscription(stripe_customer_id, plan, seats)
        token = self._issue_license_token(org_id, plan, seats)

        sub = OrgSubscription(
            id=str(uuid4()),
            organization_id=org_id,
            plan=plan,
            status="active",
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_sub_id,
            seat_limit=seats,
            seat_count=0,
            current_period_end=datetime.now(timezone.utc) + timedelta(days=30),
            license_token=token,
        )
        self.session.add(sub)
        await self.session.flush()
        return sub

    async def handle_webhook(self, *, event_type: str, payload: dict[str, Any]) -> str:
        if event_type == "customer.subscription.updated":
            return await self._sync_subscription(payload["data"]["object"])
        if event_type == "customer.subscription.deleted":
            return await self._cancel_subscription(payload["data"]["object"])
        if event_type == "invoice.payment_failed":
            return await self._mark_past_due(payload["data"]["object"])
        return "ignored"

    async def _create_stripe_subscription(self, customer_id: str, plan: str, seats: int) -> str:
        if not _STRIPE_API_KEY:
            return f"sub_mock_{customer_id[:8]}"

        import stripe  # type: ignore

        stripe.api_key = _STRIPE_API_KEY
        price_map = {
            "enterprise_self": "price_enterprise_self",
            "enterprise_managed": "price_enterprise_managed",
        }
        sub = stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_map.get(plan, "price_community"), "quantity": seats}],
        )
        return sub["id"]

    def _issue_license_token(self, org_id: str, plan: str, seats: int) -> str | None:
        if not _LICENSE_PRIVATE_KEY:
            return None

        payload = {
            "org_id": org_id,
            "plan": plan,
            "features": PLAN_FEATURES.get(plan, []),
            "seat_limit": seats,
            "iat": int(time.time()),
            "exp": int(time.time()) + 365 * 86400,
        }
        return sign_license_token(private_key_pem=_LICENSE_PRIVATE_KEY, payload=payload)

    async def _sync_subscription(self, stripe_obj: dict) -> str:
        sub_id = stripe_obj["id"]
        res = await self.session.execute(
            select(OrgSubscription).where(OrgSubscription.stripe_subscription_id == sub_id)
        )
        sub = res.scalar_one_or_none()
        if sub:
            sub.status = stripe_obj.get("status", sub.status)
        return "synced"

    async def _cancel_subscription(self, stripe_obj: dict) -> str:
        sub_id = stripe_obj["id"]
        res = await self.session.execute(
            select(OrgSubscription).where(OrgSubscription.stripe_subscription_id == sub_id)
        )
        sub = res.scalar_one_or_none()
        if sub:
            sub.status = "canceled"
            sub.plan = "community"
        return "canceled"

    async def _mark_past_due(self, stripe_obj: dict) -> str:
        customer_id = stripe_obj.get("customer")
        res = await self.session.execute(
            select(OrgSubscription).where(OrgSubscription.stripe_customer_id == customer_id)
        )
        sub = res.scalar_one_or_none()
        if sub:
            sub.status = "past_due"
        return "past_due"
