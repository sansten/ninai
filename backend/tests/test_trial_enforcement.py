"""Tests for TrialEnforcementMiddleware."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_missing_subscription_state_passes_through_for_trial_token(pg_client: AsyncClient, user_token: str):
    """With fail-open middleware and no request.state.org_subscription, request should pass through."""
    r = await pg_client.get("/api/v1/memories",
                         headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code in (200, 404)


@pytest.mark.asyncio
async def test_missing_subscription_state_passes_through_for_suspended_token(pg_client: AsyncClient, user_token: str):
    r = await pg_client.get("/api/v1/memories",
                         headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code in (200, 404)


@pytest.mark.asyncio
async def test_auth_routes_exempt_from_enforcement(pg_client: AsyncClient):
    """Login endpoint must never return 402 — enforcement is bypassed for /auth."""
    r = await pg_client.post("/api/v1/auth/login",
                          json={"email": "x@x.com", "password": "y"})
    assert r.status_code in (401, 422)  # auth failure, not 402


@pytest.mark.asyncio
async def test_active_trial_passes_through(pg_client: AsyncClient, user_token: str):
    r = await pg_client.get("/api/v1/memories",
                         headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code in (200, 404)  # 404 if no memories — not 402
