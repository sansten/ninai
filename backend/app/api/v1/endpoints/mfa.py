"""
MFA Routes - Multi-Factor Authentication endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.mfa_schemas import (
    TOTPSetupRequest, TOTPSetupResponse, TOTPVerifyRequest, TOTPVerifyResponse,
    SMSSetupRequest, SMSSetupResponse, SMSSendOTPRequest, SMSOTPVerifyRequest, SMSOTPVerifyResponse,
    WebAuthnSetupRequest, WebAuthnSetupResponse, WebAuthnVerifyRequest, WebAuthnVerifyResponse,
    MFAStatusResponse, MFADeviceResponse, MFAEnforceRequest
)
from app.services.mfa_service import (
    TOTPService, SMSService, WebAuthnService, MFAEnrollmentService
)


router = APIRouter(prefix="/mfa", tags=["mfa"])


@router.get("/status", response_model=MFAStatusResponse)
async def get_mfa_status(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Get MFA status and enrolled devices for current user"""
    try:
        from app.models.mfa import TOTPDevice, SMSDevice, WebAuthnDevice
        from sqlalchemy import select

        await MFAEnrollmentService.update_mfa_status(db, tenant.user_id)
        enrollment = await MFAEnrollmentService.get_or_create_enrollment(db, tenant.user_id)

        result = await db.execute(select(TOTPDevice).where(TOTPDevice.user_id == tenant.user_id))
        totp_devices = result.scalars().all()
        result = await db.execute(select(SMSDevice).where(SMSDevice.user_id == tenant.user_id))
        sms_devices = result.scalars().all()
        result = await db.execute(select(WebAuthnDevice).where(WebAuthnDevice.user_id == tenant.user_id))
        webauthn_devices = result.scalars().all()

        devices: list[MFADeviceResponse] = []

        for device in totp_devices:
            devices.append(MFADeviceResponse(
                id=device.id,
                device_type="totp",
                verified=device.verified,
                created_at=device.created_at,
                last_used_at=None
            ))

        for device in sms_devices:
            devices.append(MFADeviceResponse(
                id=device.id,
                device_type="sms",
                verified=device.verified,
                created_at=device.created_at,
                last_used_at=None
            ))

        for device in webauthn_devices:
            devices.append(MFADeviceResponse(
                id=device.id,
                device_type="webauthn",
                verified=device.verified,
                created_at=device.created_at,
                last_used_at=None
            ))

        return MFAStatusResponse(
            totp_enabled=enrollment.totp_enabled,
            sms_enabled=enrollment.sms_enabled,
            webauthn_enabled=enrollment.webauthn_enabled,
            mfa_required=enrollment.mfa_required,
            grace_period_until=enrollment.grace_period_until,
            devices=devices
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# TOTP Endpoints
@router.post("/totp/setup", response_model=TOTPSetupResponse)
async def setup_totp(
    request: TOTPSetupRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Setup TOTP for current user"""
    try:
        secret, qr_url, backup_codes = await TOTPService.setup_totp(db, tenant.user_id)
        return TOTPSetupResponse(
            secret=secret,
            qr_code_url=qr_url,
            backup_codes=backup_codes
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/totp/verify", response_model=TOTPVerifyResponse)
async def verify_totp(
    request: TOTPVerifyRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Verify TOTP token to complete setup"""
    try:
        success = await TOTPService.verify_totp_setup(db, tenant.user_id, request.token)

        if success:
            await MFAEnrollmentService.update_mfa_status(db, tenant.user_id, totp_enabled=True)
            return TOTPVerifyResponse(
                success=True,
                message="TOTP setup completed successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid TOTP token"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/totp/verify-login", response_model=TOTPVerifyResponse)
async def verify_totp_login(
    request: TOTPVerifyRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Verify TOTP token during login"""
    try:
        from app.models.mfa import TOTPDevice
        from sqlalchemy import select

        result = await db.execute(
            select(TOTPDevice).where(
                TOTPDevice.user_id == tenant.user_id,
                TOTPDevice.verified == True
            )
        )
        device = result.scalar_one_or_none()

        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TOTP device not found"
            )

        if TOTPService.verify_token(device.secret_key, request.token):
            return TOTPVerifyResponse(
                success=True,
                message="TOTP verification successful"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP token"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# SMS OTP Endpoints
@router.post("/sms/setup", response_model=SMSSetupResponse)
async def setup_sms(
    request: SMSSetupRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Setup SMS OTP for current user"""
    try:
        success = await SMSService.setup_sms(db, tenant.user_id, request.phone_number)

        if success:
            return SMSSetupResponse(
                success=True,
                phone_number=request.phone_number,
                message="SMS setup initiated. Check your phone for OTP."
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to setup SMS"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/sms/send-otp")
async def send_sms_otp(
    request: SMSSendOTPRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Send OTP via SMS"""
    try:
        success = await SMSService.send_sms_otp(db, tenant.user_id)

        if success:
            return {
                "success": True,
                "message": "OTP sent to your phone"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to send OTP"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/sms/verify-otp", response_model=SMSOTPVerifyResponse)
async def verify_sms_otp(
    request: SMSOTPVerifyRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db)
):
    """Verify SMS OTP"""
    try:
        success = await SMSService.verify_sms_otp(db, tenant.user_id, request.otp)

        if success:
            await MFAEnrollmentService.update_mfa_status(db, tenant.user_id, sms_enabled=True)
            return SMSOTPVerifyResponse(
                success=True,
                message="SMS OTP verified successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# Note: Remaining endpoints are placeholders for future implementation
# They require async/await refactoring for the service layer

