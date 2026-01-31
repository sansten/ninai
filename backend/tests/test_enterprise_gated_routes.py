import pytest


@pytest.mark.asyncio
async def test_autoevalbench_run_is_absent_in_community(client, auth_headers):
    resp = await client.post(
        "/api/v1/memory-activation/admin/autoevalbench/run",
        headers=auth_headers,
        json={"lookback_days": 30, "recent_days": 7, "limit": 10, "abs_drift_threshold": 0.05},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_meta_drift_endpoints_are_absent_in_community(client, auth_headers):
    resp = await client.get("/api/v1/meta/drift/latest", headers=auth_headers)
    assert resp.status_code == 404

    resp2 = await client.post("/api/v1/meta/drift/run", headers=auth_headers)
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_memory_activation_observability_routes_are_absent_in_community(client, auth_headers):
    resp = await client.get(
        "/api/v1/memory-activation/admin/observability/coactivation/top-edges",
        headers=auth_headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_ops_router_is_absent_in_community(client, auth_headers):
    resp = await client.get("/api/v1/admin/ops/metrics", headers=auth_headers)
    assert resp.status_code == 404
