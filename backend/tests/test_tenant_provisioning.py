from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.onboarding import SignupRequest, signup
from app.services.tenant_provisioning_service import ProvisionResult, TenantProvisioningService


def _exec_result(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


def _session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    return session


@pytest.mark.asyncio
async def test_provision_returns_result_fields():
    session = _session()
    session.execute.return_value = _exec_result(None)
    svc = TenantProvisioningService(session)

    result = await svc.provision(
        org_name="Acme Inc",
        admin_email="admin@acme.com",
        admin_password="supersecure",
    )

    assert isinstance(result, ProvisionResult)
    assert result.org_id
    assert result.admin_user_id
    assert result.subscription_id is None


@pytest.mark.asyncio
async def test_provision_community_skips_subscription():
    session = _session()
    session.execute.return_value = _exec_result(None)
    svc = TenantProvisioningService(session)

    with patch("app.services.tenant_provisioning_service.BillingService", create=True) as billing_cls:
        await svc.provision(
            org_name="Acme",
            admin_email="admin@acme.com",
            admin_password="supersecure",
            plan="community",
            seats=0,
        )
    assert not billing_cls.called


@pytest.mark.asyncio
async def test_provision_enterprise_calls_subscription_when_customer_present():
    session = _session()
    session.execute.return_value = _exec_result(None)
    svc = TenantProvisioningService(session)

    fake_sub = SimpleNamespace(id="sub-123")
    with patch("app.services.billing_service.BillingService") as billing_cls:
        billing = billing_cls.return_value
        billing.create_subscription = AsyncMock(return_value=fake_sub)

        result = await svc.provision(
            org_name="Acme",
            admin_email="admin@acme.com",
            admin_password="supersecure",
            plan="enterprise_self",
            seats=20,
            stripe_customer_id="cus_123",
        )

    assert result.subscription_id == "sub-123"


@pytest.mark.asyncio
async def test_deprovision_not_found():
    session = _session()
    session.execute.return_value = _exec_result(None)
    svc = TenantProvisioningService(session)

    result = await svc.deprovision(org_id="missing")
    assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_deprovision_marks_org_setting():
    org = SimpleNamespace(settings={"region": "us"})
    session = _session()
    session.execute.return_value = _exec_result(org)
    svc = TenantProvisioningService(session)

    result = await svc.deprovision(org_id="org-1", reason="canceled")
    assert result["status"] == "deprovisioned"
    assert org.settings["deprovisioned"] is True


def test_slugify_lowercases_and_replaces_chars():
    assert TenantProvisioningService._slugify("My Org! #1") == "my-org---1"


def test_slugify_truncates_to_50():
    name = "A" * 80
    slug = TenantProvisioningService._slugify(name)
    assert len(slug) == 50


@pytest.mark.asyncio
async def test_assign_org_admin_role_creates_when_missing():
    session = _session()
    session.execute.return_value = _exec_result(None)
    svc = TenantProvisioningService(session)

    await svc._assign_org_admin_role("user-1", "org-1")
    assert session.add.call_count >= 2


@pytest.mark.asyncio
async def test_assign_org_admin_role_reuses_existing_role():
    role = SimpleNamespace(id="role-1")
    session = _session()
    session.execute.return_value = _exec_result(role)
    svc = TenantProvisioningService(session)

    await svc._assign_org_admin_role("user-1", "org-1")
    # Existing role should not trigger role creation flush path.
    assert session.flush.await_count == 0


@pytest.mark.asyncio
async def test_signup_requires_min_password():
    db = AsyncMock()
    body = SignupRequest(
        org_name="Acme",
        admin_email="admin@acme.com",
        admin_password="short",
    )

    with pytest.raises(HTTPException) as exc:
        await signup(body=body, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_signup_rejects_invalid_plan():
    db = AsyncMock()
    body = SignupRequest(
        org_name="Acme",
        admin_email="admin@acme.com",
        admin_password="goodpass123",
        plan="bad-plan",
    )

    with pytest.raises(HTTPException) as exc:
        await signup(body=body, db=db)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_signup_returns_201_payload_shape():
    db = AsyncMock()
    body = SignupRequest(
        org_name="Acme",
        admin_email="admin@acme.com",
        admin_password="goodpass123",
        plan="community",
    )

    fake = ProvisionResult(
        org_id="org-1",
        admin_user_id="user-1",
        subscription_id=None,
        provisioned_at="2026-04-07T00:00:00+00:00",
    )

    with patch("app.api.v1.endpoints.onboarding.TenantProvisioningService") as svc_cls:
        svc = svc_cls.return_value
        svc.provision = AsyncMock(return_value=fake)
        resp = await signup(body=body, db=db)

    assert resp.org_id == "org-1"
    assert resp.admin_user_id == "user-1"
    assert resp.plan == "community"
