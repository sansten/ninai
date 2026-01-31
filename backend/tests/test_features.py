"""Test feature detection endpoint in OSS and Enterprise builds."""

import pytest


@pytest.mark.asyncio
async def test_features_endpoint_community_build(pg_client, auth_headers):
    """Test /features returns all flags False in Community build (no enterprise plugin)."""
    response = await pg_client.get("/api/v1/features", headers=auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    
    # Community build - all enterprise features should be False
    assert data["admin_operations"] is False
    assert data["drift_detection"] is False
    assert data["auto_eval_benchmarks"] is False
    assert data["memory_observability"] is False


@pytest.mark.asyncio
async def test_features_endpoint_requires_auth(pg_client):
    """Test /features requires authentication."""
    response = await pg_client.get("/api/v1/features")
    
    # Should return 401 Unauthorized when not authenticated
    assert response.status_code == 401
