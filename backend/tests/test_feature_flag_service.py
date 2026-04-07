from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.endpoints.feature_flags import (
    FeatureFlagBatchItem,
    FeatureFlagBatchUpsertRequest,
    FeatureFlagUpsertRequest,
    list_feature_flags,
    upsert_feature_flag,
    upsert_feature_flags,
)
from app.middleware.tenant_context import TenantContext
from app.models.base import TenantMixin
from app.models.org_feature_flag import OrgFeatureFlag
from app.services.feature_flag_service import FeatureFlagService


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


def _session_with_scalars(values) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = values
    result.scalars.return_value = scalars
    session.execute.return_value = result
    return session


def _mk_row(**kwargs):
    base = {
        "id": "ff-1",
        "organization_id": "org-1",
        "flag_name": "beta_llm_router",
        "enabled": False,
        "rollout_pct": 100,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_is_enabled_false_when_missing():
    svc = FeatureFlagService(_session_with_scalar(None), "org-1")
    assert await svc.is_enabled("f1") is False


@pytest.mark.asyncio
async def test_is_enabled_true_when_enabled():
    svc = FeatureFlagService(_session_with_scalar(_mk_row(enabled=True)), "org-1")
    assert await svc.is_enabled("f1") is True


@pytest.mark.asyncio
async def test_is_enabled_false_when_disabled():
    svc = FeatureFlagService(_session_with_scalar(_mk_row(enabled=False)), "org-1")
    assert await svc.is_enabled("f1") is False


@pytest.mark.asyncio
async def test_set_flag_creates_row_when_missing():
    db = _session_with_scalar(None)
    svc = FeatureFlagService(db, "org-1")
    row = await svc.set_flag(flag_name="f1", enabled=True)
    assert row.flag_name == "f1"
    assert db.add.call_count == 1


@pytest.mark.asyncio
async def test_set_flag_updates_existing_row():
    row = _mk_row(flag_name="f1", enabled=False, rollout_pct=10)
    db = _session_with_scalar(row)
    svc = FeatureFlagService(db, "org-1")
    updated = await svc.set_flag(flag_name="f1", enabled=True, rollout_pct=80)
    assert updated.enabled is True
    assert updated.rollout_pct == 80


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_pct,expected",
    [(-1, 0), (0, 0), (1, 1), (100, 100), (120, 100)],
)
async def test_set_flag_clamps_rollout_pct(input_pct, expected):
    row = _mk_row(flag_name="f1")
    db = _session_with_scalar(row)
    svc = FeatureFlagService(db, "org-1")
    updated = await svc.set_flag(flag_name="f1", enabled=True, rollout_pct=input_pct)
    assert updated.rollout_pct == expected


@pytest.mark.asyncio
async def test_list_flags_empty_list():
    svc = FeatureFlagService(_session_with_scalars([]), "org-1")
    assert await svc.list_flags() == []


@pytest.mark.asyncio
async def test_list_flags_returns_dict_shape():
    rows = [_mk_row(flag_name="a", enabled=True, rollout_pct=30), _mk_row(flag_name="b", enabled=False, rollout_pct=100)]
    svc = FeatureFlagService(_session_with_scalars(rows), "org-1")
    out = await svc.list_flags()
    assert len(out) == 2
    assert set(out[0].keys()) == {"flag_name", "enabled", "rollout_pct"}


def test_model_contract_tablename_and_columns():
    assert OrgFeatureFlag.__tablename__ == "org_feature_flags"
    assert issubclass(OrgFeatureFlag, TenantMixin)
    assert hasattr(OrgFeatureFlag, "flag_name")
    assert hasattr(OrgFeatureFlag, "enabled")
    assert hasattr(OrgFeatureFlag, "rollout_pct")
    assert hasattr(OrgFeatureFlag, "metadata_")


def test_model_has_unique_constraint_name():
    names = [c.name for c in OrgFeatureFlag.__table__.constraints if c.name]
    assert "uq_org_feature_flags_org_flag" in names


def test_flag_name_column_length_200():
    col = OrgFeatureFlag.__table__.columns["flag_name"]
    assert getattr(col.type, "length", None) == 200


@pytest.mark.asyncio
async def test_endpoint_list_feature_flags_returns_flags_payload(tenant_ctx):
    db = AsyncMock()
    expected = [{"flag_name": "f1", "enabled": True, "rollout_pct": 50}]
    with patch("app.api.v1.endpoints.feature_flags.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.feature_flags.FeatureFlagService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.list_flags = AsyncMock(return_value=expected)
        out = await list_feature_flags(tenant=tenant_ctx, db=db)
    assert out == {"flags": expected}


@pytest.mark.asyncio
async def test_endpoint_upsert_single_returns_row_payload(tenant_ctx):
    db = AsyncMock()
    body = FeatureFlagUpsertRequest(enabled=True, rollout_pct=80)
    row = _mk_row(flag_name="beta", enabled=True, rollout_pct=80)
    with patch("app.api.v1.endpoints.feature_flags.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.feature_flags.FeatureFlagService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.set_flag = AsyncMock(return_value=row)
        out = await upsert_feature_flag(flag_name="beta", body=body, tenant=tenant_ctx, db=db)
    assert out == {"flag_name": "beta", "enabled": True, "rollout_pct": 80}


@pytest.mark.asyncio
async def test_endpoint_upsert_batch_commits_and_returns_flags(tenant_ctx):
    db = AsyncMock()
    body = FeatureFlagBatchUpsertRequest(
        flags=[
            FeatureFlagBatchItem(flag_name="f1", enabled=True, rollout_pct=10),
            FeatureFlagBatchItem(flag_name="f2", enabled=False, rollout_pct=100),
        ]
    )
    expected = [
        {"flag_name": "f1", "enabled": True, "rollout_pct": 10},
        {"flag_name": "f2", "enabled": False, "rollout_pct": 100},
    ]

    with patch("app.api.v1.endpoints.feature_flags.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.feature_flags.FeatureFlagService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.set_flag = AsyncMock(return_value=_mk_row())
        svc.list_flags = AsyncMock(return_value=expected)
        out = await upsert_feature_flags(body=body, tenant=tenant_ctx, db=db)

    assert svc.set_flag.await_count == 2
    assert out == {"flags": expected}


@pytest.mark.asyncio
async def test_endpoint_put_single_allows_rollout_zero(tenant_ctx):
    db = AsyncMock()
    body = FeatureFlagUpsertRequest(enabled=True, rollout_pct=0)
    row = _mk_row(flag_name="f", enabled=True, rollout_pct=0)
    with patch("app.api.v1.endpoints.feature_flags.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.feature_flags.FeatureFlagService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.set_flag = AsyncMock(return_value=row)
        out = await upsert_feature_flag(flag_name="f", body=body, tenant=tenant_ctx, db=db)
    assert out["rollout_pct"] == 0


@pytest.mark.asyncio
async def test_endpoint_put_single_allows_rollout_hundred(tenant_ctx):
    db = AsyncMock()
    body = FeatureFlagUpsertRequest(enabled=False, rollout_pct=100)
    row = _mk_row(flag_name="f", enabled=False, rollout_pct=100)
    with patch("app.api.v1.endpoints.feature_flags.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.feature_flags.FeatureFlagService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.set_flag = AsyncMock(return_value=row)
        out = await upsert_feature_flag(flag_name="f", body=body, tenant=tenant_ctx, db=db)
    assert out["rollout_pct"] == 100


def test_endpoint_request_model_rejects_rollout_below_zero():
    with pytest.raises(Exception):
        FeatureFlagUpsertRequest(enabled=True, rollout_pct=-1)


def test_endpoint_request_model_rejects_rollout_above_hundred():
    with pytest.raises(Exception):
        FeatureFlagUpsertRequest(enabled=True, rollout_pct=101)


@pytest.mark.asyncio
async def test_service_set_flag_is_idempotent_same_name():
    row = _mk_row(flag_name="f1", enabled=False)
    db = _session_with_scalar(row)
    svc = FeatureFlagService(db, "org-1")
    await svc.set_flag(flag_name="f1", enabled=True, rollout_pct=50)
    await svc.set_flag(flag_name="f1", enabled=False, rollout_pct=20)
    assert row.flag_name == "f1"
    assert row.enabled is False
    assert row.rollout_pct == 20


def test_router_registration_contains_feature_flags_paths():
    from app.main import app

    paths = {getattr(r, "path", None) for r in app.router.routes}
    assert "/api/v1/admin/feature-flags" in paths
    assert "/api/v1/admin/feature-flags/{flag_name}" in paths


@pytest.mark.asyncio
async def test_service_returns_flag_names_with_underscores_dots_and_dashes():
    rows = [
        _mk_row(flag_name="phase_81_playbook_synthesis"),
        _mk_row(flag_name="beta.llm.router"),
        _mk_row(flag_name="drift-detection-v2"),
    ]
    svc = FeatureFlagService(_session_with_scalars(rows), "org-1")
    out = await svc.list_flags()
    names = {x["flag_name"] for x in out}
    assert "phase_81_playbook_synthesis" in names
    assert "beta.llm.router" in names
    assert "drift-detection-v2" in names
