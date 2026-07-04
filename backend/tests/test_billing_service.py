from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.v1.endpoints.billing import (
    CheckoutRequest,
    create_checkout,
    get_subscription,
    stripe_webhook,
)
from app.middleware.tenant_context import TenantContext
from app.models.org_subscription import OrgSubscription
from app.services.billing_service import BillingService, PLAN_FEATURES


@pytest.fixture
def tenant_ctx() -> TenantContext:
    return TenantContext(
        user_id="user-1",
        org_id="org-1",
        roles=["org_admin"],
        clearance_level=1,
    )


def _session_with_scalar(value) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    session.execute.return_value = result
    return session


def _mk_sub(**kwargs):
    base = dict(
        id="sub-1",
        organization_id="org-1",
        plan="enterprise_self",
        status="active",
        stripe_customer_id="cus_123",
        stripe_subscription_id="sub_123",
        seat_limit=20,
        seat_count=3,
        current_period_end=None,
        license_token=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_get_subscription_none_for_new_org():
    svc = BillingService(_session_with_scalar(None))
    assert await svc.get_subscription("org-1") is None


@pytest.mark.asyncio
async def test_get_subscription_returns_existing():
    sub = _mk_sub()
    svc = BillingService(_session_with_scalar(sub))
    assert await svc.get_subscription("org-1") == sub


@pytest.mark.asyncio
async def test_create_stripe_customer_mock_without_key():
    svc = BillingService(AsyncMock())
    customer_id = await svc.create_stripe_customer(org_id="org-1234", email="a@b.com", org_name="Acme")
    assert customer_id.startswith("cus_mock_")


@pytest.mark.asyncio
async def test_create_subscription_sets_fields():
    session = AsyncMock()
    session.add = MagicMock()
    svc = BillingService(session)

    sub = await svc.create_subscription(
        org_id="org-1",
        stripe_customer_id="cus_1",
        plan="enterprise_self",
        seats=25,
        email="admin@acme.com",
    )
    assert sub.plan == "enterprise_self"
    assert sub.seat_limit == 25
    assert sub.status == "active"
    assert sub.license_token is None


@pytest.mark.asyncio
async def test_handle_webhook_updated_syncs_status():
    sub = _mk_sub(status="trialing")
    svc = BillingService(_session_with_scalar(sub))
    result = await svc.handle_webhook(
        event_type="customer.subscription.updated",
        payload={"data": {"object": {"id": "sub_123", "status": "past_due"}}},
    )
    assert result == "synced"
    assert sub.status == "past_due"


@pytest.mark.asyncio
async def test_handle_webhook_deleted_sets_canceled_and_community():
    sub = _mk_sub(plan="enterprise_managed")
    svc = BillingService(_session_with_scalar(sub))
    result = await svc.handle_webhook(
        event_type="customer.subscription.deleted",
        payload={"data": {"object": {"id": "sub_123"}}},
    )
    assert result == "canceled"
    assert sub.status == "canceled"
    assert sub.plan == "community"


@pytest.mark.asyncio
async def test_handle_webhook_payment_failed_sets_past_due():
    sub = _mk_sub(stripe_customer_id="cus_123")
    svc = BillingService(_session_with_scalar(sub))
    result = await svc.handle_webhook(
        event_type="invoice.payment_failed",
        payload={"data": {"object": {"customer": "cus_123"}}},
    )
    assert result == "past_due"
    assert sub.status == "past_due"


@pytest.mark.asyncio
async def test_handle_webhook_unknown_returns_ignored():
    svc = BillingService(AsyncMock())
    result = await svc.handle_webhook(event_type="customer.created", payload={})
    assert result == "ignored"


def test_plan_features_shape():
    assert PLAN_FEATURES["community"] == []
    assert len(PLAN_FEATURES["enterprise_self"]) == 7


def test_org_subscription_model_contract():
    assert OrgSubscription.__tablename__ == "org_subscriptions"
    assert hasattr(OrgSubscription, "stripe_customer_id")
    assert hasattr(OrgSubscription, "stripe_subscription_id")


@pytest.mark.asyncio
async def test_endpoint_get_subscription_returns_community_when_missing(tenant_ctx):
    db = AsyncMock()
    with patch("app.api.v1.endpoints.billing.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.billing.BillingService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.get_subscription = AsyncMock(return_value=None)
        resp = await get_subscription(tenant=tenant_ctx, db=db)

    assert resp.plan == "community"
    assert resp.status == "active"


@pytest.mark.asyncio
async def test_endpoint_checkout_rejects_invalid_plan(tenant_ctx):
    db = AsyncMock()
    body = CheckoutRequest(plan="bad", seats=10, email="a@b.com", org_name="Acme")
    with patch("app.api.v1.endpoints.billing.set_tenant_context", new=AsyncMock()):
        with pytest.raises(HTTPException) as exc:
            await create_checkout(body=body, tenant=tenant_ctx, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_checkout_requires_min_10_seats(tenant_ctx):
    db = AsyncMock()
    body = CheckoutRequest(plan="enterprise_self", seats=9, email="a@b.com", org_name="Acme")
    with patch("app.api.v1.endpoints.billing.set_tenant_context", new=AsyncMock()):
        with pytest.raises(HTTPException) as exc:
            await create_checkout(body=body, tenant=tenant_ctx, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_endpoint_checkout_duplicate_active_returns_409(tenant_ctx):
    db = AsyncMock()
    body = CheckoutRequest(plan="enterprise_self", seats=10, email="a@b.com", org_name="Acme")
    active_sub = _mk_sub(status="active")

    with patch("app.api.v1.endpoints.billing.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.billing.BillingService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.get_subscription = AsyncMock(return_value=active_sub)
        with pytest.raises(HTTPException) as exc:
            await create_checkout(body=body, tenant=tenant_ctx, db=db)

    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_endpoint_checkout_creates_subscription(tenant_ctx):
    db = AsyncMock()
    body = CheckoutRequest(plan="enterprise_self", seats=12, email="a@b.com", org_name="Acme")
    created = _mk_sub(id="db-sub", plan="enterprise_self", status="active", seat_limit=12)

    with patch("app.api.v1.endpoints.billing.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.billing.BillingService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.get_subscription = AsyncMock(return_value=None)
        svc.create_stripe_customer = AsyncMock(return_value="cus_mock_org")
        svc.create_subscription = AsyncMock(return_value=created)

        resp = await create_checkout(body=body, tenant=tenant_ctx, db=db)

    assert resp["plan"] == "enterprise_self"
    assert resp["status"] == "active"


@pytest.mark.asyncio
async def test_webhook_accepts_without_secret():
    payload = {"type": "customer.subscription.updated", "data": {"object": {"id": "sub_1"}}}

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/billing/webhook", "headers": []}, receive)
    db = AsyncMock()

    with patch.dict("os.environ", {"STRIPE_WEBHOOK_SECRET": ""}, clear=False), patch(
        "app.api.v1.endpoints.billing.BillingService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.handle_webhook = AsyncMock(return_value="synced")
        resp = await stripe_webhook(request=request, stripe_signature=None, db=db)

    assert resp["status"] == "synced"


@pytest.mark.asyncio
async def test_webhook_rejects_missing_signature_header_when_secret_configured():
    """Regression: `if secret and stripe_signature:` used to skip verification
    entirely whenever the caller simply omitted the stripe-signature header,
    even with a secret configured — letting a forged event through unverified."""
    payload = {"type": "customer.subscription.updated", "data": {"object": {"id": "sub_1"}}}

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/billing/webhook", "headers": []}, receive)
    db = AsyncMock()

    with patch.dict("os.environ", {"STRIPE_WEBHOOK_SECRET": "whsec"}, clear=False):
        with pytest.raises(HTTPException) as exc:
            await stripe_webhook(request=request, stripe_signature=None, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_webhook_rejects_bad_signature():
    payload = {"type": "customer.subscription.updated", "data": {"object": {"id": "sub_1"}}}

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/billing/webhook", "headers": []}, receive)
    db = AsyncMock()

    with patch.dict("os.environ", {"STRIPE_WEBHOOK_SECRET": "whsec"}, clear=False):
        with pytest.raises(HTTPException) as exc:
            await stripe_webhook(request=request, stripe_signature="t=123,v1=bad", db=db)
    assert exc.value.status_code == 400
