"""GDPR / Compliance Service.

Handles data retention policy management, deletion request lifecycle,
data portability exports, and consent record management.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.compliance import ConsentRecord, DataRetentionPolicy, UserDataDeletion

logger = logging.getLogger(__name__)

_VALID_SCOPES = {"account", "org_data", "all"}
_VALID_CONSENT_TYPES = {"data_processing", "analytics", "marketing"}
_VALID_DELETION_STATUSES = {"pending", "processing", "completed", "failed"}


class ComplianceService:
    """Business logic for GDPR / compliance operations."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Data Retention Policy
    # ------------------------------------------------------------------

    async def get_retention_policy(self, org_id: str) -> Optional[DataRetentionPolicy]:
        """Return the retention policy for an org, or None if not configured."""
        stmt = select(DataRetentionPolicy).where(
            DataRetentionPolicy.organization_id == org_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_retention_policy(
        self,
        org_id: str,
        *,
        retention_days_memory: int = 365,
        retention_days_audit: int = 730,
        auto_archive_after_days: int = 90,
        created_by_user_id: Optional[str] = None,
    ) -> DataRetentionPolicy:
        """Create or update the retention policy for an org."""
        existing = await self.get_retention_policy(org_id)
        if existing is not None:
            existing.retention_days_memory = max(1, retention_days_memory)
            existing.retention_days_audit = max(1, retention_days_audit)
            existing.auto_archive_after_days = max(1, auto_archive_after_days)
            existing.is_active = True
            await self.db.commit()
            await self.db.refresh(existing)
            return existing

        policy = DataRetentionPolicy(
            organization_id=org_id,
            retention_days_memory=max(1, retention_days_memory),
            retention_days_audit=max(1, retention_days_audit),
            auto_archive_after_days=max(1, auto_archive_after_days),
            created_by_user_id=created_by_user_id,
            is_active=True,
        )
        self.db.add(policy)
        await self.db.commit()
        await self.db.refresh(policy)
        return policy

    # ------------------------------------------------------------------
    # Deletion Requests
    # ------------------------------------------------------------------

    async def create_deletion_request(
        self,
        user_id: str,
        org_id: str,
        *,
        requested_by_user_id: Optional[str] = None,
        reason: Optional[str] = None,
        deletion_scope: str = "account",
    ) -> UserDataDeletion:
        """Create a new deletion request for a user."""
        if deletion_scope not in _VALID_SCOPES:
            raise ValueError(f"deletion_scope must be one of {_VALID_SCOPES}")

        request = UserDataDeletion(
            user_id=user_id,
            organization_id=org_id,
            requested_by_user_id=requested_by_user_id,
            reason=reason,
            deletion_scope=deletion_scope,
            status="pending",
        )
        self.db.add(request)
        await self.db.commit()
        await self.db.refresh(request)
        logger.info(
            "Deletion request created",
            extra={"user_id": user_id, "org_id": org_id, "scope": deletion_scope},
        )
        return request

    async def get_deletion_request(self, request_id: str) -> Optional[UserDataDeletion]:
        """Fetch a deletion request by ID."""
        stmt = select(UserDataDeletion).where(
            UserDataDeletion.id == request_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_deletion_requests(
        self, org_id: str, *, status: Optional[str] = None, limit: int = 50
    ) -> list[UserDataDeletion]:
        """List deletion requests for an org, optionally filtered by status."""
        conditions = [UserDataDeletion.organization_id == org_id]
        if status and status in _VALID_DELETION_STATUSES:
            conditions.append(UserDataDeletion.status == status)
        stmt = (
            select(UserDataDeletion)
            .where(and_(*conditions))
            .order_by(UserDataDeletion.created_at.desc())
            .limit(min(limit, 200))
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def mark_deletion_processing(self, request_id: str) -> Optional[UserDataDeletion]:
        """Transition a deletion request to processing state."""
        req = await self.get_deletion_request(request_id)
        if req is None:
            return None
        req.status = "processing"
        await self.db.commit()
        await self.db.refresh(req)
        return req

    async def mark_deletion_completed(self, request_id: str) -> Optional[UserDataDeletion]:
        """Mark a deletion request as completed."""
        req = await self.get_deletion_request(request_id)
        if req is None:
            return None
        req.status = "completed"
        req.deleted_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(req)
        return req

    async def mark_deletion_failed(
        self, request_id: str, error_detail: str
    ) -> Optional[UserDataDeletion]:
        """Mark a deletion request as failed with an error message."""
        req = await self.get_deletion_request(request_id)
        if req is None:
            return None
        req.status = "failed"
        req.error_detail = str(error_detail)[:2000]
        await self.db.commit()
        await self.db.refresh(req)
        return req

    # ------------------------------------------------------------------
    # Consent Records
    # ------------------------------------------------------------------

    async def record_consent(
        self,
        user_id: str,
        org_id: str,
        *,
        consent_type: str,
        granted: bool,
        ip_address: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> ConsentRecord:
        """Record a consent grant or revoke for a user."""
        if consent_type not in _VALID_CONSENT_TYPES:
            raise ValueError(f"consent_type must be one of {_VALID_CONSENT_TYPES}")

        now = datetime.now(timezone.utc)
        record = ConsentRecord(
            user_id=user_id,
            organization_id=org_id,
            consent_type=consent_type,
            granted=granted,
            granted_at=now,
            expires_at=expires_at,
            revoked_at=None if granted else now,
            ip_address=ip_address,
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def list_user_consents(
        self, user_id: str, org_id: str
    ) -> list[ConsentRecord]:
        """List all consent records for a user within an org."""
        stmt = (
            select(ConsentRecord)
            .where(
                ConsentRecord.user_id == user_id,
                ConsentRecord.organization_id == org_id,
            )
            .order_by(ConsentRecord.granted_at.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_consent(
        self, user_id: str, org_id: str, consent_type: str
    ) -> Optional[ConsentRecord]:
        """Return the most recent consent record for a user + type."""
        stmt = (
            select(ConsentRecord)
            .where(
                ConsentRecord.user_id == user_id,
                ConsentRecord.organization_id == org_id,
                ConsentRecord.consent_type == consent_type,
            )
            .order_by(ConsentRecord.granted_at.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()


def anonymize_user_id(user_id: str) -> str:
    """Return a SHA-256 hash of the user_id for anonymization in audit trails."""
    return "anon_" + hashlib.sha256(user_id.encode()).hexdigest()[:16]
