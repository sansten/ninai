"""
MFA Models - TOTP, SMS, WebAuthn devices
"""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ARRAY, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSON
from datetime import datetime
import uuid

from app.models.base import Base


class TOTPDevice(Base):
    """Time-based One-Time Password (TOTP) device"""
    __tablename__ = "mfa_totp_device"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, unique=True)
    secret_key = Column(String(32), nullable=False)  # Base32 encoded secret
    verified = Column(Boolean, default=False)
    backup_codes = Column(JSON, nullable=True)  # List of backup codes
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SMSDevice(Base):
    """SMS-based One-Time Password device"""
    __tablename__ = "mfa_sms_device"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, unique=True)
    phone_number = Column(String(20), nullable=False)
    verified = Column(Boolean, default=False)
    last_sent_at = Column(DateTime, nullable=True)
    failed_attempts = Column(Integer, default=0)  # Track failed verification attempts
    locked_until = Column(DateTime, nullable=True)  # Rate limit lockout
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WebAuthnDevice(Base):
    """WebAuthn/FIDO2 security key device"""
    __tablename__ = "mfa_webauthn_device"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    credential_id = Column(LargeBinary, nullable=False, unique=True)  # Binary credential ID
    public_key = Column(LargeBinary, nullable=False)  # Binary public key
    sign_count = Column(Integer, default=0)  # Counter for signature verification
    transports = Column(JSON, nullable=True)  # Transports used (usb, nfc, ble, etc)
    device_name = Column(String(255), nullable=True)  # User-friendly name
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MFAEnrollment(Base):
    """Track MFA enrollment status for users"""
    __tablename__ = "mfa_enrollment"

    id = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, unique=True)
    totp_enabled = Column(Boolean, default=False)
    sms_enabled = Column(Boolean, default=False)
    webauthn_enabled = Column(Boolean, default=False)
    mfa_required = Column(Boolean, default=False)  # User must use MFA
    grace_period_until = Column(DateTime, nullable=True)  # When user must enable MFA
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
