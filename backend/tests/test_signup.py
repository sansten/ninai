"""Tests for public signup flow."""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_signup_creates_org_and_user(pg_client: AsyncClient):
    r = await pg_client.post("/api/v1/signup", json={
        "email": "owner@acme.example",
        "password": "Securepass1",
        "full_name": "Acme Owner",
        "org_name": "Acme Corp",
    })
    assert r.status_code == 201
    data = r.json()
    assert "org_id" in data
    assert data["message"] == "Verification email sent. Please check your inbox."


@pytest.mark.asyncio
async def test_signup_duplicate_email_returns_409(pg_client: AsyncClient):
    payload = {
        "email": "dup@example.com",
        "password": "Securepass1",
        "full_name": "User",
        "org_name": "Org",
    }
    await pg_client.post("/api/v1/signup", json=payload)
    r = await pg_client.post("/api/v1/signup", json=payload)
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_signup_weak_password_returns_422(pg_client: AsyncClient):
    r = await pg_client.post("/api/v1/signup", json={
        "email": "test@example.com",
        "password": "weak",
        "full_name": "User",
        "org_name": "Org",
    })
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_verify_invalid_token_returns_400(pg_client: AsyncClient):
    r = await pg_client.get("/api/v1/signup/verify?token=notarealtoken")
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_verify_valid_token_returns_jwt(pg_client: AsyncClient, pg_db_session):
    """Full happy path: signup -> extract token from DB -> verify -> get JWT."""
    from sqlalchemy import select
    from app.models.organization import Organization

    r = await pg_client.post("/api/v1/signup", json={
        "email": "verify@example.com",
        "password": "Securepass1",
        "full_name": "V User",
        "org_name": "Verify Org",
    })
    assert r.status_code == 201
    org_id = r.json()["org_id"]

    result = await pg_db_session.execute(
        select(Organization).where(Organization.id == org_id)
    )
    org = result.scalar_one()
    token = org.settings["_vt"]

    verify_r = await pg_client.get(f"/api/v1/signup/verify?token={token}")
    assert verify_r.status_code == 200
    body = verify_r.json()
    assert "access_token" in body
    assert body["org_id"] == org_id


@pytest.mark.asyncio
async def test_resend_always_returns_202(pg_client: AsyncClient):
    r = await pg_client.post("/api/v1/signup/resend?email=nonexistent@example.com")
    assert r.status_code == 202
