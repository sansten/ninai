"""Tests for MFA API endpoints."""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mfa import TOTPDevice, SMSDevice, MFAEnrollment
from app.models.user import User
from app.services.mfa_service import TOTPService


@pytest_asyncio.fixture
async def mfa_test_user(db_session: AsyncSession):
    """Create a test user for MFA tests."""
    user_id = uuid4()
    user = User(
        id=user_id,
        email=f"mfa_test_{user_id}@example.com",
        hashed_password="hashed_password_placeholder",
        full_name=f"MFA Test User",
        is_active=True,
        role="user"
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.fixture
def mfa_auth_headers(mfa_test_user, test_org_id):
    """Generate proper auth headers for MFA tests."""
    from app.core.security import create_access_token

    token = create_access_token(
        user_id=str(mfa_test_user.id),
        org_id=test_org_id,
        roles=["user"],
    )
    return {
        "Authorization": f"Bearer {token}",
    }


@pytest.mark.asyncio
async def test_totp_setup_endpoint(pg_client: AsyncClient, mfa_auth_headers, db_session):
    """Test TOTP setup endpoint."""
    response = await pg_client.post("/api/v1/mfa/totp/setup", json={}, headers=mfa_auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "secret" in data
    assert "qr_code_url" in data
    assert "backup_codes" in data
    assert len(data["backup_codes"]) == 10


@pytest.mark.asyncio
async def test_totp_verify_endpoint(pg_client: AsyncClient, mfa_auth_headers, db_session, test_user_id):
    """Test TOTP verification endpoint."""
    # Setup TOTP first
    setup_response = await pg_client.post("/api/v1/mfa/totp/setup", json={}, headers=mfa_auth_headers)
    secret = setup_response.json()["secret"]
    
    # Generate valid token
    token = TOTPService.generate_token(secret)
    
    # Verify
    response = await pg_client.post(
        "/api/v1/mfa/totp/verify",
        json={"token": token},
        headers=mfa_auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "TOTP" in data["message"]


@pytest.mark.asyncio
async def test_totp_verify_invalid_token(pg_client: AsyncClient, mfa_auth_headers):
    """Test TOTP verification with invalid token."""
    # Setup TOTP
    await pg_client.post("/api/v1/mfa/totp/setup", json={}, headers=mfa_auth_headers)
    
    # Try invalid token
    response = await pg_client.post(
        "/api/v1/mfa/totp/verify",
        json={"token": "000000"},
        headers=mfa_auth_headers
    )
    
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_sms_setup_endpoint(pg_client: AsyncClient, mfa_auth_headers):
    """Test SMS setup endpoint."""
    response = await pg_client.post(
        "/api/v1/mfa/sms/setup",
        json={"phone_number": "+1234567890"},
        headers=mfa_auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["phone_number"] == "+1234567890"


@pytest.mark.asyncio
async def test_sms_send_otp_endpoint(pg_client: AsyncClient, mfa_auth_headers):
    """Test SMS send OTP endpoint."""
    # Setup SMS first
    await pg_client.post(
        "/api/v1/mfa/sms/setup",
        json={"phone_number": "+1234567890"},
        headers=mfa_auth_headers
    )
    
    # Send OTP
    response = await pg_client.post("/api/v1/mfa/sms/send-otp", json={}, headers=mfa_auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_sms_verify_otp_endpoint(pg_client: AsyncClient, mfa_auth_headers):
    """Test SMS OTP verification endpoint."""
    # Setup and send
    await pg_client.post(
        "/api/v1/mfa/sms/setup",
        json={"phone_number": "+1234567890"},
        headers=mfa_auth_headers
    )
    await pg_client.post("/api/v1/mfa/sms/send-otp", json={}, headers=mfa_auth_headers)
    
    # Verify (in test mode, any 6-digit code works)
    response = await pg_client.post(
        "/api/v1/mfa/sms/verify-otp",
        json={"otp": "123456"},
        headers=mfa_auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


@pytest.mark.asyncio
async def test_mfa_status_endpoint(pg_client: AsyncClient, mfa_auth_headers, test_user_id, db_session):
    """Test MFA status endpoint."""
    response = await pg_client.get("/api/v1/mfa/status", headers=mfa_auth_headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "totp_enabled" in data
    assert "sms_enabled" in data
    assert "webauthn_enabled" in data
    assert "mfa_required" in data
    assert "devices" in data
    
    # Initially all disabled
    assert data["totp_enabled"] is False
    assert data["sms_enabled"] is False
    assert data["webauthn_enabled"] is False


@pytest.mark.asyncio
async def test_mfa_status_after_totp_setup(pg_client: AsyncClient, mfa_auth_headers):
    """Test MFA status shows TOTP enabled after setup."""
    # Setup and verify TOTP
    setup_response = await pg_client.post("/api/v1/mfa/totp/setup", json={}, headers=mfa_auth_headers)
    secret = setup_response.json()["secret"]
    token = TOTPService.generate_token(secret)
    await pg_client.post("/api/v1/mfa/totp/verify", json={"token": token}, headers=mfa_auth_headers)
    
    # Check status
    response = await pg_client.get("/api/v1/mfa/status", headers=mfa_auth_headers)
    data = response.json()
    
    assert data["totp_enabled"] is True
    assert len(data["devices"]) == 1
    assert data["devices"][0]["device_type"] == "totp"
    assert data["devices"][0]["verified"] is True


@pytest.mark.asyncio
async def test_mfa_status_multiple_devices(pg_client: AsyncClient, mfa_auth_headers):
    """Test MFA status with multiple device types."""
    # Setup TOTP
    setup_response = await pg_client.post("/api/v1/mfa/totp/setup", json={}, headers=mfa_auth_headers)
    secret = setup_response.json()["secret"]
    token = TOTPService.generate_token(secret)
    await pg_client.post("/api/v1/mfa/totp/verify", json={"token": token}, headers=mfa_auth_headers)
    
    # Setup SMS
    await pg_client.post("/api/v1/mfa/sms/setup", json={"phone_number": "+1234567890"}, headers=mfa_auth_headers)
    await pg_client.post("/api/v1/mfa/sms/send-otp", json={}, headers=mfa_auth_headers)
    await pg_client.post("/api/v1/mfa/sms/verify-otp", json={"otp": "123456"}, headers=mfa_auth_headers)
    
    # Check status
    response = await pg_client.get("/api/v1/mfa/status", headers=mfa_auth_headers)
    data = response.json()
    
    assert data["totp_enabled"] is True
    assert data["sms_enabled"] is True
    assert len(data["devices"]) == 2


@pytest.mark.asyncio
async def test_totp_token_validation(pg_client: AsyncClient, mfa_auth_headers):
    """Test TOTP token must be 6 digits."""
    await pg_client.post("/api/v1/mfa/totp/setup", json={}, headers=mfa_auth_headers)
    
    # Too short
    response = await pg_client.post(
        "/api/v1/mfa/totp/verify",
        json={"token": "12345"},
        headers=mfa_auth_headers
    )
    assert response.status_code == 422  # Validation error
    
    # Too long
    response = await pg_client.post(
        "/api/v1/mfa/totp/verify",
        json={"token": "1234567"},
        headers=mfa_auth_headers
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_sms_otp_validation(pg_client: AsyncClient, mfa_auth_headers):
    """Test SMS OTP must be 6 digits."""
    await pg_client.post("/api/v1/mfa/sms/setup", json={"phone_number": "+1234567890"}, headers=mfa_auth_headers)
    
    # Invalid OTP
    response = await pg_client.post(
        "/api/v1/mfa/sms/verify-otp",
        json={"otp": "12345"},
        headers=mfa_auth_headers
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_unauthorized_access(pg_client: AsyncClient):
    """Test MFA endpoints require authentication."""
    # No auth headers
    response = await pg_client.get("/api/v1/mfa/status")
    assert response.status_code in [401, 403]
    
    response = await pg_client.post("/api/v1/mfa/totp/setup", json={})
    assert response.status_code in [401, 403]
