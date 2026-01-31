"""
MFA Service - TOTP, SMS, WebAuthn management (Async)
"""

import pyotp
import secrets
import string
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from app.models.mfa import TOTPDevice, SMSDevice, WebAuthnDevice, MFAEnrollment
from app.models.user import User


class TOTPService:
    @staticmethod
    def generate_secret() -> str:
        """Generate a random TOTP secret"""
        return pyotp.random_base32()

    @staticmethod
    def get_totp(secret: str) -> pyotp.TOTP:
        """Get TOTP object for a secret"""
        return pyotp.TOTP(secret)

    @staticmethod
    def generate_token(secret: str) -> str:
        """Generate a valid TOTP token for a secret"""
        totp = pyotp.TOTP(secret)
        return totp.now()

    @staticmethod
    def get_qr_code_url(secret: str, user_email: str, issuer: str = "Ninai") -> str:
        """Get QR code URL for TOTP setup"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=user_email, issuer_name=issuer)

    @staticmethod
    def verify_token(secret: str, token: str, window: int = 1) -> bool:
        """Verify a TOTP token (with time window for clock drift)"""
        if not token or len(token) != 6:
            return False
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception:
            return False

    @staticmethod
    def generate_backup_codes(count: int = 10) -> List[str]:
        """Generate backup codes for account recovery"""
        codes = []
        for _ in range(count):
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            codes.append(code)
        return codes

    @staticmethod
    def verify_backup_code(secret: str, codes: List[str], code: str) -> Tuple[bool, List[str]]:
        """Verify and consume a backup code"""
        if code in codes:
            codes.remove(code)
            return True, codes
        return False, codes

    @staticmethod
    async def setup_totp(db: AsyncSession, user_id: UUID | str) -> Tuple[str, str, List[str]]:
        """Setup TOTP for a user - returns (secret, qr_url, backup_codes)"""
        user_id = str(user_id)
        # Check if already has TOTP
        result = await db.execute(
            select(TOTPDevice).where(TOTPDevice.user_id == user_id)
        )
        existing = result.scalar_one_or_none()
        if existing and existing.verified:
            raise ValueError("User already has TOTP enabled")

        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found")

        secret = TOTPService.generate_secret()
        qr_url = TOTPService.get_qr_code_url(secret, user.email)
        backup_codes = TOTPService.generate_backup_codes()

        # Create or update TOTP device
        if existing:
            existing.secret_key = secret
            existing.backup_codes = backup_codes
            existing.verified = False
        else:
            device = TOTPDevice(
                user_id=user_id,
                secret_key=secret,
                backup_codes=backup_codes,
                verified=False
            )
            db.add(device)

        await db.commit()
        return secret, qr_url, backup_codes

    @staticmethod
    async def verify_totp_setup(db: AsyncSession, user_id: UUID | str, token: str) -> bool:
        """Verify TOTP token to complete setup"""
        user_id = str(user_id)
        result = await db.execute(
            select(TOTPDevice).where(
                TOTPDevice.user_id == user_id,
                TOTPDevice.verified == False
            )
        )
        device = result.scalar_one_or_none()

        if not device:
            return False

        if TOTPService.verify_token(device.secret_key, token):
            device.verified = True
            device.updated_at = datetime.utcnow()
            await db.commit()
            return True

        return False


class SMSService:
    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """Generate a random OTP"""
        return ''.join(secrets.choice(string.digits) for _ in range(length))

    @staticmethod
    async def send_otp(db: AsyncSession, user_id: UUID | str, phone_number: str) -> bool:
        """Send OTP via SMS (stub - implement with Twilio)"""
        user_id = str(user_id)
        # TODO: Implement with Twilio or other SMS provider
        otp = SMSService.generate_otp()
        # In production: send_sms(phone_number, f"Your NinaiOS OTP: {otp}")
        return True

    @staticmethod
    async def setup_sms(db: AsyncSession, user_id: UUID | str, phone_number: str) -> bool:
        """Setup SMS MFA for a user"""
        user_id = str(user_id)
        result = await db.execute(
            select(SMSDevice).where(SMSDevice.user_id == user_id)
        )
        existing = result.scalar_one_or_none()

        if existing and existing.verified:
            raise ValueError("User already has SMS MFA enabled")

        if existing:
            existing.phone_number = phone_number
            existing.verified = False
            existing.failed_attempts = 0
        else:
            device = SMSDevice(
                user_id=user_id,
                phone_number=phone_number,
                verified=False
            )
            db.add(device)

        await db.commit()
        return await SMSService.send_otp(db, user_id, phone_number)

    @staticmethod
    async def send_sms_otp(db: AsyncSession, user_id: UUID | str) -> bool:
        """Send OTP for SMS verification"""
        user_id = str(user_id)
        result = await db.execute(
            select(SMSDevice).where(SMSDevice.user_id == user_id)
        )
        device = result.scalar_one_or_none()

        if not device:
            raise ValueError("SMS device not found")

        if device.locked_until and device.locked_until > datetime.utcnow():
            raise ValueError("SMS device is locked due to failed attempts")

        # Update last_sent_at timestamp
        device.last_sent_at = datetime.utcnow()
        await db.commit()

        return await SMSService.send_otp(db, user_id, device.phone_number)

    @staticmethod
    async def verify_sms_otp(db: AsyncSession, user_id: UUID | str, otp: str) -> bool:
        """Verify SMS OTP (stub - implement with SMS service)"""
        user_id = str(user_id)
        result = await db.execute(
            select(SMSDevice).where(SMSDevice.user_id == user_id)
        )
        device = result.scalar_one_or_none()

        if not device:
            return False

        if device.locked_until and device.locked_until > datetime.utcnow():
            return False

        # TODO: Verify against stored OTP in cache/Redis
        # For now, in test mode any 6-digit OTP is valid
        is_valid = len(otp) == 6 and otp.isdigit()
        
        if is_valid:
            device.verified = True
            device.failed_attempts = 0
            device.last_sent_at = datetime.utcnow()
            await db.commit()
            return True
        else:
            # Invalid OTP - increment failed attempts
            device.failed_attempts += 1
            
            if device.failed_attempts >= 5:
                device.locked_until = datetime.utcnow() + timedelta(minutes=30)
            
            await db.commit()
            return False


class WebAuthnService:
    @staticmethod
    async def register_credential(
        db: AsyncSession,
        user_id: UUID | str,
        credential_id: bytes,
        public_key: bytes,
        device_name: str = "WebAuthn Device"
    ) -> bool:
        """Register a WebAuthn credential"""
        user_id = str(user_id)
        device = WebAuthnDevice(
            user_id=user_id,
            credential_id=credential_id if isinstance(credential_id, bytes) else credential_id.encode(),
            public_key=public_key if isinstance(public_key, bytes) else public_key.encode(),
            device_name=device_name,
            verified=False
        )
        db.add(device)
        await db.commit()
        return True

    @staticmethod
    async def verify_webauthn_setup(db: AsyncSession, user_id: UUID | str) -> bool:
        """Verify WebAuthn setup"""
        user_id = str(user_id)
        result = await db.execute(
            select(WebAuthnDevice).where(
                WebAuthnDevice.user_id == user_id,
                WebAuthnDevice.verified == False
            )
        )
        device = result.scalar_one_or_none()

        if not device:
            return False

        # TODO: Implement actual WebAuthn verification
        device.verified = True
        device.updated_at = datetime.utcnow()
        await db.commit()
        return True

    @staticmethod
    async def get_user_devices(db: AsyncSession, user_id: UUID) -> List[WebAuthnDevice]:
        """Get all WebAuthn devices for a user"""
        result = await db.execute(
            select(WebAuthnDevice).where(WebAuthnDevice.user_id == user_id)
        )
        return result.scalars().all()


class MFAEnrollmentService:
    @staticmethod
    async def get_or_create_enrollment(db: AsyncSession, user_id: UUID) -> MFAEnrollment:
        """Get or create MFA enrollment for a user"""
        result = await db.execute(
            select(MFAEnrollment).where(MFAEnrollment.user_id == user_id)
        )
        enrollment = result.scalar_one_or_none()

        if not enrollment:
            enrollment = MFAEnrollment(user_id=user_id)
            db.add(enrollment)
            await db.commit()

        return enrollment

    @staticmethod
    async def update_mfa_status(
        db: AsyncSession,
        user_id: UUID,
        totp_enabled: Optional[bool] = None,
        sms_enabled: Optional[bool] = None,
        webauthn_enabled: Optional[bool] = None
    ) -> MFAEnrollment:
        """Update MFA status for a user - auto-detects if parameters are None"""
        enrollment = await MFAEnrollmentService.get_or_create_enrollment(db, user_id)

        # Auto-detect TOTP status if not explicitly provided
        if totp_enabled is None:
            result = await db.execute(
                select(TOTPDevice).where(
                    TOTPDevice.user_id == user_id,
                    TOTPDevice.verified == True
                )
            )
            totp_device = result.scalar_one_or_none()
            enrollment.totp_enabled = totp_device is not None
        else:
            enrollment.totp_enabled = totp_enabled

        # Auto-detect SMS status if not explicitly provided
        if sms_enabled is None:
            result = await db.execute(
                select(SMSDevice).where(
                    SMSDevice.user_id == user_id,
                    SMSDevice.verified == True
                )
            )
            sms_device = result.scalar_one_or_none()
            enrollment.sms_enabled = sms_device is not None
        else:
            enrollment.sms_enabled = sms_enabled

        # Auto-detect WebAuthn status if not explicitly provided
        if webauthn_enabled is None:
            result = await db.execute(
                select(WebAuthnDevice).where(
                    WebAuthnDevice.user_id == user_id,
                    WebAuthnDevice.verified == True
                )
            )
            webauthn_device = result.scalar_one_or_none()
            enrollment.webauthn_enabled = webauthn_device is not None
        else:
            enrollment.webauthn_enabled = webauthn_enabled

        await db.commit()
        return enrollment

    @staticmethod
    async def has_any_mfa(db: AsyncSession, user_id: UUID) -> bool:
        """Check if user has any MFA method enabled"""
        result = await db.execute(
            select(MFAEnrollment).where(MFAEnrollment.user_id == user_id)
        )
        enrollment = result.scalar_one_or_none()

        if not enrollment:
            return False

        return enrollment.totp_enabled or enrollment.sms_enabled or enrollment.webauthn_enabled

    @staticmethod
    async def require_mfa(
        db: AsyncSession,
        user_id: UUID,
        grace_period_days: int = 7
    ) -> MFAEnrollment:
        """Mark MFA as required for a user"""
        enrollment = await MFAEnrollmentService.get_or_create_enrollment(db, user_id)
        enrollment.mfa_required = True
        enrollment.grace_period_until = datetime.utcnow() + timedelta(days=grace_period_days)
        await db.commit()
        return enrollment
