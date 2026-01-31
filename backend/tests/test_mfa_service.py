"""Tests for MFA service."""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.mfa import TOTPDevice, SMSDevice, WebAuthnDevice, MFAEnrollment
from app.models.user import User
from app.models.organization import Organization
from app.services.mfa_service import TOTPService, SMSService, WebAuthnService, MFAEnrollmentService


@pytest.fixture
def test_user_id():
    return str(uuid4())


@pytest_asyncio.fixture
async def test_user(db_session: AsyncSession, test_user_id):
    """Create a test user."""
    existing = await db_session.get(User, test_user_id)
    if existing is not None:
        return existing

    user = User(
        id=test_user_id,
        email=f"test_{test_user_id}@example.com",
        hashed_password="hashed_password_placeholder",
        full_name=f"Test User {test_user_id}",
        is_active=True,
        role="user",
    )
    db_session.add(user)
    await db_session.commit()
    return user


@pytest.mark.asyncio
async def test_totp_setup(db_session: AsyncSession, test_user_id, test_user):
    """Test TOTP device setup."""
    db_session.info["auto_commit"] = False
    
    secret, qr_url, backup_codes = await TOTPService.setup_totp(db_session, test_user_id)
    
    assert len(secret) == 32  # Base32 encoded secret
    assert "otpauth://" in qr_url
    assert len(backup_codes) == 10
    assert all(len(code) == 8 for code in backup_codes)
    
    # Verify device created
    result = await db_session.execute(
        select(TOTPDevice).where(TOTPDevice.user_id == test_user_id)
    )
    device = result.scalar_one_or_none()
    assert device is not None
    assert device.secret_key == secret
    assert device.verified is False


@pytest.mark.asyncio
async def test_totp_verify_setup(db_session: AsyncSession, test_user_id, test_user):
    """Test TOTP setup verification."""
    db_session.info["auto_commit"] = False
    
    secret, _, _ = await TOTPService.setup_totp(db_session, test_user_id)
    await db_session.commit()
    
    # Generate valid token
    token = TOTPService.generate_token(secret)
    
    # Verify
    success = await TOTPService.verify_totp_setup(db_session, test_user_id, token)
    assert success is True
    
    # Check device is verified
    result = await db_session.execute(
        select(TOTPDevice).where(TOTPDevice.user_id == test_user_id)
    )
    device = result.scalar_one()
    assert device.verified is True


@pytest.mark.asyncio
async def test_totp_verify_invalid_token(db_session: AsyncSession, test_user_id, test_user):
    """Test TOTP verification with invalid token."""
    db_session.info["auto_commit"] = False
    
    await TOTPService.setup_totp(db_session, test_user_id)
    await db_session.commit()
    
    # Try invalid token
    success = await TOTPService.verify_totp_setup(db_session, test_user_id, "000000")
    assert success is False


@pytest.mark.asyncio
async def test_sms_setup(db_session: AsyncSession, test_user_id, test_user):
    """Test SMS device setup."""
    db_session.info["auto_commit"] = False
    
    phone = "+1234567890"
    success = await SMSService.setup_sms(db_session, test_user_id, phone)
    assert success is True
    
    # Verify device created
    result = await db_session.execute(
        select(SMSDevice).where(SMSDevice.user_id == test_user_id)
    )
    device = result.scalar_one_or_none()
    assert device is not None
    assert device.phone_number == phone
    assert device.verified is False


@pytest.mark.asyncio
async def test_sms_update_phone(db_session: AsyncSession, test_user_id, test_user):
    """Test updating SMS phone number."""
    db_session.info["auto_commit"] = False
    
    # Initial setup
    await SMSService.setup_sms(db_session, test_user_id, "+1111111111")
    await db_session.commit()
    
    # Update phone
    new_phone = "+2222222222"
    await SMSService.setup_sms(db_session, test_user_id, new_phone)
    await db_session.commit()
    
    # Verify updated
    result = await db_session.execute(
        select(SMSDevice).where(SMSDevice.user_id == test_user_id)
    )
    device = result.scalar_one()
    assert device.phone_number == new_phone
    assert device.verified is False  # Should reset verification


@pytest.mark.asyncio
async def test_sms_send_otp(db_session: AsyncSession, test_user_id, test_user):
    """Test sending SMS OTP."""
    db_session.info["auto_commit"] = False
    
    await SMSService.setup_sms(db_session, test_user_id, "+1234567890")
    await db_session.commit()
    
    success = await SMSService.send_sms_otp(db_session, test_user_id)
    assert success is True
    
    # Verify last_sent_at updated
    result = await db_session.execute(
        select(SMSDevice).where(SMSDevice.user_id == test_user_id)
    )
    device = result.scalar_one()
    assert device.last_sent_at is not None


@pytest.mark.asyncio
async def test_sms_verify_otp(db_session: AsyncSession, test_user_id, test_user):
    """Test SMS OTP verification."""
    db_session.info["auto_commit"] = False
    
    await SMSService.setup_sms(db_session, test_user_id, "+1234567890")
    await db_session.commit()
    
    # In test mode, any 6-digit OTP works
    success = await SMSService.verify_sms_otp(db_session, test_user_id, "123456")
    assert success is True
    
    # Verify device is verified
    result = await db_session.execute(
        select(SMSDevice).where(SMSDevice.user_id == test_user_id)
    )
    device = result.scalar_one()
    assert device.verified is True


@pytest.mark.asyncio
async def test_sms_rate_limiting(db_session: AsyncSession, test_user_id, test_user):
    """Test SMS rate limiting after failed attempts."""
    db_session.info["auto_commit"] = False
    
    await SMSService.setup_sms(db_session, test_user_id, "+1234567890")
    await db_session.commit()
    
    # Simulate 5 failed attempts
    for _ in range(5):
        await SMSService.verify_sms_otp(db_session, test_user_id, "invalid")
        await db_session.commit()
    
    # Verify locked
    result = await db_session.execute(
        select(SMSDevice).where(SMSDevice.user_id == test_user_id)
    )
    device = result.scalar_one()
    assert device.locked_until is not None
    assert device.locked_until > datetime.utcnow()


@pytest.mark.asyncio
async def test_webauthn_register(db_session: AsyncSession, test_user_id, test_user):
    """Test WebAuthn credential registration."""
    db_session.info["auto_commit"] = False
    
    credential_id = b"test_credential_id"
    public_key = b"test_public_key"
    
    success = await WebAuthnService.register_credential(
        db_session, test_user_id, credential_id, public_key, "YubiKey"
    )
    assert success is True
    
    # Verify device created
    result = await db_session.execute(
        select(WebAuthnDevice).where(WebAuthnDevice.user_id == test_user_id)
    )
    device = result.scalar_one_or_none()
    assert device is not None
    assert device.credential_id == credential_id
    assert device.public_key == public_key
    assert device.device_name == "YubiKey"


@pytest.mark.asyncio
async def test_webauthn_get_devices(db_session: AsyncSession, test_user_id, test_user):
    """Test getting user's WebAuthn devices."""
    db_session.info["auto_commit"] = False
    
    # Register multiple devices
    await WebAuthnService.register_credential(
        db_session, test_user_id, b"key1", b"pub1", "YubiKey 1"
    )
    await WebAuthnService.register_credential(
        db_session, test_user_id, b"key2", b"pub2", "YubiKey 2"
    )
    await db_session.commit()
    
    devices = await WebAuthnService.get_user_devices(db_session, test_user_id)
    assert len(devices) == 2


@pytest.mark.asyncio
async def test_mfa_enrollment_creation(db_session: AsyncSession, test_user_id, test_user):
    """Test MFA enrollment record creation."""
    db_session.info["auto_commit"] = False
    
    enrollment = await MFAEnrollmentService.get_or_create_enrollment(db_session, test_user_id)
    assert enrollment is not None
    assert enrollment.user_id == test_user_id
    assert enrollment.totp_enabled is False
    assert enrollment.sms_enabled is False
    assert enrollment.webauthn_enabled is False


@pytest.mark.asyncio
async def test_mfa_status_update_totp(db_session: AsyncSession, test_user_id, test_user):
    """Test MFA status update when TOTP is enabled."""
    db_session.info["auto_commit"] = False
    
    # Setup and verify TOTP
    secret, _, _ = await TOTPService.setup_totp(db_session, test_user_id)
    await db_session.commit()
    
    token = TOTPService.generate_token(secret)
    await TOTPService.verify_totp_setup(db_session, test_user_id, token)
    await db_session.commit()
    
    # Update MFA status
    await MFAEnrollmentService.update_mfa_status(db_session, test_user_id)
    await db_session.commit()
    
    # Verify enrollment reflects TOTP enabled
    result = await db_session.execute(
        select(MFAEnrollment).where(MFAEnrollment.user_id == test_user_id)
    )
    enrollment = result.scalar_one()
    assert enrollment.totp_enabled is True
    assert enrollment.sms_enabled is False


@pytest.mark.asyncio
async def test_mfa_status_update_multiple_methods(db_session: AsyncSession, test_user_id, test_user):
    """Test MFA status with multiple methods enabled."""
    db_session.info["auto_commit"] = False
    
    # Setup TOTP
    secret, _, _ = await TOTPService.setup_totp(db_session, test_user_id)
    token = TOTPService.generate_token(secret)
    await TOTPService.verify_totp_setup(db_session, test_user_id, token)
    
    # Setup SMS
    await SMSService.setup_sms(db_session, test_user_id, "+1234567890")
    await SMSService.verify_sms_otp(db_session, test_user_id, "123456")
    
    await db_session.commit()
    
    # Update status
    await MFAEnrollmentService.update_mfa_status(db_session, test_user_id)
    await db_session.commit()
    
    # Verify both enabled
    result = await db_session.execute(
        select(MFAEnrollment).where(MFAEnrollment.user_id == test_user_id)
    )
    enrollment = result.scalar_one()
    assert enrollment.totp_enabled is True
    assert enrollment.sms_enabled is True


@pytest.mark.asyncio
async def test_has_any_mfa(db_session: AsyncSession, test_user_id, test_user):
    """Test checking if user has any MFA enabled."""
    db_session.info["auto_commit"] = False
    
    # Initially no MFA
    has_mfa = await MFAEnrollmentService.has_any_mfa(db_session, test_user_id)
    assert has_mfa is False
    
    # Enable TOTP
    secret, _, _ = await TOTPService.setup_totp(db_session, test_user_id)
    token = TOTPService.generate_token(secret)
    await TOTPService.verify_totp_setup(db_session, test_user_id, token)
    await MFAEnrollmentService.update_mfa_status(db_session, test_user_id)
    await db_session.commit()
    
    # Now has MFA
    has_mfa = await MFAEnrollmentService.has_any_mfa(db_session, test_user_id)
    assert has_mfa is True


@pytest.mark.asyncio
async def test_require_mfa_with_grace_period(db_session: AsyncSession, test_user_id, test_user):
    """Test requiring MFA with grace period."""
    db_session.info["auto_commit"] = False
    
    await MFAEnrollmentService.require_mfa(db_session, test_user_id, grace_period_days=7)
    await db_session.commit()
    
    result = await db_session.execute(
        select(MFAEnrollment).where(MFAEnrollment.user_id == test_user_id)
    )
    enrollment = result.scalar_one()
    assert enrollment.mfa_required is True
    assert enrollment.grace_period_until is not None
    
    # Grace period should be ~7 days from now
    expected = datetime.utcnow() + timedelta(days=7)
    assert abs((enrollment.grace_period_until - expected).total_seconds()) < 60


@pytest.mark.asyncio
async def test_totp_backup_codes_stored(db_session: AsyncSession, test_user_id, test_user):
    """Test that TOTP backup codes are stored."""
    db_session.info["auto_commit"] = False
    
    _, _, backup_codes = await TOTPService.setup_totp(db_session, test_user_id)
    await db_session.commit()
    
    result = await db_session.execute(
        select(TOTPDevice).where(TOTPDevice.user_id == test_user_id)
    )
    device = result.scalar_one()
    assert device.backup_codes is not None
    assert len(device.backup_codes) == 10
    assert set(device.backup_codes) == set(backup_codes)
