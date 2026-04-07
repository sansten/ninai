from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.endpoints.usage import usage_daily, usage_summary
from app.middleware.tenant_context import TenantContext
from app.services.usage_service import UsageService


@pytest.fixture
def tenant_ctx() -> TenantContext:
    return TenantContext(
        user_id="user-1",
        org_id="org-1",
        roles=["org_admin"],
        clearance_level=1,
    )


@pytest.mark.asyncio
async def test_usage_service_increment_executes_upsert():
    db = AsyncMock()
    svc = UsageService(db, "org-1")

    await svc.increment(metric="memory_writes", value=2)

    assert db.execute.await_count == 1


@pytest.mark.asyncio
async def test_usage_service_summary_maps_totals():
    db = AsyncMock()
    row = MagicMock()
    row.metric = "memory_writes"
    row.total = 9
    res = MagicMock()
    res.all.return_value = [row]
    db.execute.return_value = res

    svc = UsageService(db, "org-1")
    summary = await svc.get_summary(days=7)

    assert summary == {"memory_writes": 9}


@pytest.mark.asyncio
async def test_usage_summary_endpoint_returns_payload(tenant_ctx):
    db = AsyncMock()

    with patch("app.api.v1.endpoints.usage.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.usage.UsageService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.get_summary = AsyncMock(return_value={"memory_writes": 4})

        resp = await usage_summary(days=14, tenant=tenant_ctx, db=db)

    assert resp["days"] == 14
    assert resp["summary"]["memory_writes"] == 4


@pytest.mark.asyncio
async def test_usage_daily_endpoint_returns_points(tenant_ctx):
    db = AsyncMock()

    points = [{"date": "2026-04-07", "value": 2}]
    with patch("app.api.v1.endpoints.usage.set_tenant_context", new=AsyncMock()), patch(
        "app.api.v1.endpoints.usage.UsageService"
    ) as svc_cls:
        svc = svc_cls.return_value
        svc.get_daily = AsyncMock(return_value=points)

        resp = await usage_daily(metric="memory_writes", days=30, tenant=tenant_ctx, db=db)

    assert resp["metric"] == "memory_writes"
    assert resp["points"] == points
