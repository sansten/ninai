"""
Pydantic schemas for MFA endpoints
"""

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class TOTPSetupRequest(BaseModel):
    """Request to setup TOTP"""
    pass


class TOTPSetupResponse(BaseModel):
    """Response for TOTP setup"""
    secret: str = Field(..., description="Base32 encoded TOTP secret")
    qr_code_url: str = Field(..., description="URL for QR code")
    backup_codes: List[str] = Field(..., description="Backup recovery codes")


class TOTPVerifyRequest(BaseModel):
    """Request to verify TOTP token"""
    token: str = Field(..., min_length=6, max_length=6, description="6-digit TOTP token")


class TOTPVerifyResponse(BaseModel):
    """Response for TOTP verification"""
    success: bool
    message: str


class SMSSetupRequest(BaseModel):
    """Request to setup SMS OTP"""
    phone_number: str = Field(..., description="Phone number for SMS OTP")


class SMSSetupResponse(BaseModel):
    """Response for SMS setup"""
    success: bool
    phone_number: str
    message: str


class SMSSendOTPRequest(BaseModel):
    """Request to send SMS OTP"""
    pass


class SMSOTPVerifyRequest(BaseModel):
    """Request to verify SMS OTP"""
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP from SMS")


class SMSOTPVerifyResponse(BaseModel):
    """Response for SMS OTP verification"""
    success: bool
    message: str


class WebAuthnSetupRequest(BaseModel):
    """Request to setup WebAuthn"""
    device_name: str = Field(..., description="Name for the security key")


class WebAuthnSetupResponse(BaseModel):
    """Response for WebAuthn setup"""
    challenge: str = Field(..., description="Attestation challenge")
    timeout: int = Field(default=60000, description="Timeout in milliseconds")


class WebAuthnVerifyRequest(BaseModel):
    """Request to verify WebAuthn registration"""
    credential_id: str = Field(..., description="Base64 encoded credential ID")
    public_key: str = Field(..., description="Base64 encoded public key")
    device_name: str = Field(..., description="Name for the security key")


class WebAuthnVerifyResponse(BaseModel):
    """Response for WebAuthn verification"""
    success: bool
    message: str


class MFADeviceResponse(BaseModel):
    """Response for MFA device info"""
    id: UUID
    device_type: str  # totp, sms, webauthn
    verified: bool
    created_at: datetime
    last_used_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class MFAStatusResponse(BaseModel):
    """Response for MFA status"""
    totp_enabled: bool
    sms_enabled: bool
    webauthn_enabled: bool
    mfa_required: bool
    grace_period_until: Optional[datetime] = None
    devices: List[MFADeviceResponse]


class MFAEnforceRequest(BaseModel):
    """Request to enforce MFA for users"""
    grace_period_days: int = Field(default=7, ge=1, le=30, description="Days before MFA is required")
    target_user_ids: Optional[List[UUID]] = Field(default=None, description="Specific users to enforce for (None = all)")
