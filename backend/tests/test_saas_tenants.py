"""Tests for super-admin SaaS tenant management endpoints."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_tenants_requires_system_admin(client: AsyncClient, user_token: str):
    r = await client.get("/api/v1/admin/saas/tenants",
                         headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_list_tenants_as_system_admin(client: AsyncClient, superadmin_token: str):
    r = await client.get("/api/v1/admin/saas/tenants",
                         headers={"Authorization": f"Bearer {superadmin_token}"})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_patch_tenant_status(client: AsyncClient, superadmin_token: str, test_org_id: str):
    r = await client.patch(
        f"/api/v1/admin/saas/tenants/{test_org_id}",
        json={"status": "suspended"},
        headers={"Authorization": f"Bearer {superadmin_token}"},
    )
    assert r.status_code == 200

    # Restore
    await client.patch(
        f"/api/v1/admin/saas/tenants/{test_org_id}",
        json={"status": "active"},
        headers={"Authorization": f"Bearer {superadmin_token}"},
    )


@pytest.mark.asyncio
async def test_impersonate_returns_short_lived_token(
    client: AsyncClient, superadmin_token: str, test_org_id: str
):
    r = await client.post(
        f"/api/v1/admin/saas/tenants/{test_org_id}/impersonate",
        headers={"Authorization": f"Bearer {superadmin_token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["expires_in"] == 900


@pytest.mark.asyncio
async def test_patch_nonexistent_org_returns_404(client: AsyncClient, superadmin_token: str):
    r = await client.patch(
        "/api/v1/admin/saas/tenants/00000000-0000-0000-0000-000000000000",
        json={"status": "active"},
        headers={"Authorization": f"Bearer {superadmin_token}"},
    )
    assert r.status_code == 404
