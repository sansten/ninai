"""Tests for TrialEnforcementMiddleware."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_expired_trial_returns_402(client: AsyncClient, expired_trial_token: str):
    """Any protected endpoint with an expired trial token returns 402."""
    r = await client.get("/api/v1/memories",
                         headers={"Authorization": f"Bearer {expired_trial_token}"})
    assert r.status_code == 402
    assert r.json()["error"] == "trial_expired"


@pytest.mark.asyncio
async def test_suspended_org_returns_402(client: AsyncClient, suspended_org_token: str):
    r = await client.get("/api/v1/memories",
                         headers={"Authorization": f"Bearer {suspended_org_token}"})
    assert r.status_code == 402
    assert r.json()["error"] == "account_suspended"


@pytest.mark.asyncio
async def test_auth_routes_exempt_from_enforcement(client: AsyncClient):
    """Login endpoint must never return 402 — enforcement is bypassed for /auth."""
    r = await client.post("/api/v1/auth/login",
                          json={"email": "x@x.com", "password": "y"})
    assert r.status_code in (401, 422)  # auth failure, not 402


@pytest.mark.asyncio
async def test_active_trial_passes_through(client: AsyncClient, active_trial_token: str):
    r = await client.get("/api/v1/memories",
                         headers={"Authorization": f"Bearer {active_trial_token}"})
    assert r.status_code in (200, 404)  # 404 if no memories — not 402
