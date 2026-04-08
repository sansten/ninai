from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.endpoints.dpa import CURRENT_DPA_VERSION, accept_dpa, get_dpa_status
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.base import TenantMixin
from app.models.dpa_acceptance import DpaAcceptance


@pytest.fixture
def tenant_ctx() -> TenantContext:
    return TenantContext(
        user_id="user-1",
        org_id="org-1",
        roles=["org_admin"],
        clearance_level=1,
    )


@pytest.fixture
def non_admin_ctx() -> TenantContext:
    return TenantContext(
        user_id="user-2",
        org_id="org-1",
        roles=["member"],
        clearance_level=1,
    )


def _session_with_scalar(value) -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    session.execute.return_value = result
    return session


def _mk_row(**kwargs):
    base = {
        "id": "dpa-1",
        "organization_id": "org-1",
        "dpa_version": CURRENT_DPA_VERSION,
        "accepted": False,
        "accepted_by_user_id": None,
        "accepted_at": None,
        "ip_address": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_status_returns_not_accepted_for_new_org(tenant_ctx):
    db = _session_with_scalar(None)
    with patch("app.api.v1.endpoints.dpa.set_tenant_context", new=AsyncMock()):
        out = await get_dpa_status(tenant=tenant_ctx, db=db)

    assert out["current_version"] == CURRENT_DPA_VERSION
    assert out["accepted"] is False
    assert out["accepted_at"] is None
    assert out["accepted_by"] is None


@pytest.mark.asyncio
async def test_status_returns_accepted_after_accept(tenant_ctx):
    row = _mk_row(accepted=True, accepted_by_user_id="user-1", accepted_at="2026-04-07T00:00:00+00:00")
    db = _session_with_scalar(row)
    with patch("app.api.v1.endpoints.dpa.set_tenant_context", new=AsyncMock()):
        out = await get_dpa_status(tenant=tenant_ctx, db=db)

    assert out["accepted"] is True
    assert out["accepted_by"] == "user-1"
    assert out["accepted_at"] is not None


@pytest.mark.asyncio
async def test_accept_creates_row_when_missing(tenant_ctx):
    db = _session_with_scalar(None)
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    with patch("app.api.v1.endpoints.dpa.set_tenant_context", new=AsyncMock()):
        out = await accept_dpa(request=request, tenant=tenant_ctx, db=db)

    assert out["accepted"] is True
    assert out["version"] == CURRENT_DPA_VERSION
    assert db.add.call_count == 1


@pytest.mark.asyncio
async def test_accept_updates_existing_row_and_records_fields(tenant_ctx):
    row = _mk_row()
    db = _session_with_scalar(row)
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.5"))

    with patch("app.api.v1.endpoints.dpa.set_tenant_context", new=AsyncMock()):
        out = await accept_dpa(request=request, tenant=tenant_ctx, db=db)

    assert out["accepted"] is True
    assert row.accepted is True
    assert row.accepted_by_user_id == "user-1"
    assert isinstance(row.accepted_at, str)
    assert "T" in row.accepted_at
    assert row.ip_address == "10.0.0.5"


@pytest.mark.asyncio
async def test_accept_is_idempotent_on_existing_row(tenant_ctx):
    row = _mk_row(accepted=True, accepted_at="2026-04-07T00:00:00+00:00")
    db = _session_with_scalar(row)
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.6"))

    with patch("app.api.v1.endpoints.dpa.set_tenant_context", new=AsyncMock()):
        first = await accept_dpa(request=request, tenant=tenant_ctx, db=db)
        second = await accept_dpa(request=request, tenant=tenant_ctx, db=db)

    assert second["accepted"] is True
    assert first["accepted_at"] <= second["accepted_at"]


@pytest.mark.asyncio
async def test_accept_handles_missing_client_ip(tenant_ctx):
    row = _mk_row()
    db = _session_with_scalar(row)
    request = SimpleNamespace(client=None)

    with patch("app.api.v1.endpoints.dpa.set_tenant_context", new=AsyncMock()):
        await accept_dpa(request=request, tenant=tenant_ctx, db=db)

    assert row.ip_address is None


@pytest.mark.asyncio
async def test_require_org_admin_blocks_non_admin(non_admin_ctx):
    dep = require_org_admin()
    with pytest.raises(Exception):
        await dep(tenant=non_admin_ctx)


def test_current_version_contract():
    assert CURRENT_DPA_VERSION == "2026-04-01"


def test_model_contract():
    assert DpaAcceptance.__tablename__ == "dpa_acceptances"
    assert issubclass(DpaAcceptance, TenantMixin)
    assert hasattr(DpaAcceptance, "accepted")
    assert hasattr(DpaAcceptance, "dpa_version")
    assert hasattr(DpaAcceptance, "ip_address")


def test_ip_address_column_nullable():
    col = DpaAcceptance.__table__.columns["ip_address"]
    assert col.nullable is True


def test_dpa_version_column_length_contract():
    col = DpaAcceptance.__table__.columns["dpa_version"]
    assert getattr(col.type, "length", None) == 20


def test_router_registration_contains_dpa_paths():
    from app.main import app

    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert "/api/v1/admin/dpa/status" in paths
    assert "/api/v1/admin/dpa/accept" in paths
