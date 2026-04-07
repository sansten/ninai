from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.data_residency import REGION_TO_GCP, ResidencyRequest, get_residency, set_residency
from app.middleware.tenant_context import TenantContext
from app.models.base import TenantMixin
from app.models.org_data_residency import OrgDataResidency, VALID_REGIONS


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
    session.add = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    session.execute.return_value = result
    return session


def _mk_row(**kwargs):
    base = dict(
        id="row-1",
        organization_id="org-1",
        region="us",
        gdpr_required=False,
        backup_region=None,
        gcp_region="us-central1",
        declared_at=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_get_returns_default_when_no_row(tenant_ctx):
    db = _session_with_scalar(None)

    with patch("app.api.v1.endpoints.data_residency.set_tenant_context", new=AsyncMock()):
        resp = await get_residency(tenant=tenant_ctx, db=db)

    assert resp["region"] == "us"
    assert resp["gdpr_required"] is False
    assert resp["gcp_region"] == "us-central1"


@pytest.mark.asyncio
async def test_get_returns_existing_row(tenant_ctx):
    row = _mk_row(region="eu", gdpr_required=True, gcp_region="europe-west1", backup_region="us")
    db = _session_with_scalar(row)

    with patch("app.api.v1.endpoints.data_residency.set_tenant_context", new=AsyncMock()):
        resp = await get_residency(tenant=tenant_ctx, db=db)

    assert resp["region"] == "eu"
    assert resp["gdpr_required"] is True
    assert resp["gcp_region"] == "europe-west1"
    assert resp["backup_region"] == "us"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "region,expected_gcp,expected_gdpr",
    [
        ("us", "us-central1", False),
        ("eu", "europe-west1", True),
        ("apac", "asia-southeast1", False),
        ("ca", "northamerica-northeast1", False),
    ],
)
async def test_put_sets_region_mappings(tenant_ctx, region, expected_gcp, expected_gdpr):
    row = _mk_row()
    db = _session_with_scalar(row)
    body = ResidencyRequest(region=region)

    with patch("app.api.v1.endpoints.data_residency.set_tenant_context", new=AsyncMock()):
        resp = await set_residency(body=body, tenant=tenant_ctx, db=db)

    assert resp["region"] == region
    assert resp["gcp_region"] == expected_gcp
    assert resp["gdpr_required"] is expected_gdpr
    assert row.gcp_region == expected_gcp
    assert row.gdpr_required is expected_gdpr


@pytest.mark.asyncio
async def test_put_invalid_region_returns_400(tenant_ctx):
    db = _session_with_scalar(None)
    body = ResidencyRequest(region="moon")

    with patch("app.api.v1.endpoints.data_residency.set_tenant_context", new=AsyncMock()):
        with pytest.raises(HTTPException) as exc:
            await set_residency(body=body, tenant=tenant_ctx, db=db)

    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_put_creates_row_when_missing(tenant_ctx):
    db = _session_with_scalar(None)
    body = ResidencyRequest(region="us", backup_region="ca")

    with patch("app.api.v1.endpoints.data_residency.set_tenant_context", new=AsyncMock()):
        await set_residency(body=body, tenant=tenant_ctx, db=db)

    assert db.add.call_count == 1


@pytest.mark.asyncio
async def test_put_updates_existing_row_and_sets_declared_at(tenant_ctx):
    row = _mk_row(region="us")
    db = _session_with_scalar(row)
    body = ResidencyRequest(region="ca", backup_region="us")

    with patch("app.api.v1.endpoints.data_residency.set_tenant_context", new=AsyncMock()):
        await set_residency(body=body, tenant=tenant_ctx, db=db)

    assert row.region == "ca"
    assert row.backup_region == "us"
    assert isinstance(row.declared_at, str)
    assert "T" in row.declared_at


def test_valid_regions_contract():
    assert VALID_REGIONS == frozenset({"us", "eu", "apac", "ca"})


def test_region_mapping_contract():
    assert REGION_TO_GCP["us"] == "us-central1"
    assert REGION_TO_GCP["eu"] == "europe-west1"
    assert REGION_TO_GCP["apac"] == "asia-southeast1"
    assert REGION_TO_GCP["ca"] == "northamerica-northeast1"


def test_model_contract():
    assert OrgDataResidency.__tablename__ == "org_data_residency"
    assert issubclass(OrgDataResidency, TenantMixin)
    assert hasattr(OrgDataResidency, "region")
    assert hasattr(OrgDataResidency, "gdpr_required")
    assert hasattr(OrgDataResidency, "gcp_region")
    assert hasattr(OrgDataResidency, "backup_region")
    assert hasattr(OrgDataResidency, "declared_at")
