"""
Capability Token Management Service - Issue, Revoke, and Manage Tokens

Admins use this to grant agents/users capability tokens for memory syscalls.
"""

import uuid
import secrets
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, String
from sqlalchemy.sql import cast
import logging

from app.models.capability_token import CapabilityToken, CapabilityScope
from app.models.user import User
from app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class CapabilityTokenService:
    """
    Manage capability tokens for scoped memory access.
    
    Only org admins can:
    - Issue tokens
    - Revoke tokens
    - List tokens
    - Update quotas
    """

    def __init__(self, db: AsyncSession, organization_id: uuid.UUID):
        self.db = db
        self.organization_id = organization_id

    async def issue_token(
        self,
        name: str,
        scopes: list[str],  # ["read", "search", "append"]
        session_id: Optional[uuid.UUID] = None,
        agent_name: Optional[str] = None,
        issued_to_user_id: Optional[uuid.UUID] = None,
        ttl_seconds: int = 86400,  # 24 hours
        max_tokens_per_month: Optional[int] = None,
        max_storage_bytes: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
        created_by_user_id: Optional[uuid.UUID] = None
    ) -> CapabilityToken:
        """
        Issue a new capability token.
        
        Returns: CapabilityToken with .token (the actual Bearer token)
        """
        # Generate secure token
        # Must fit DB schema (VARCHAR(60)). token_urlsafe(42) yields 56 chars;
        # with 'cap_' prefix that's exactly 60.
        token_value = f"cap_{secrets.token_urlsafe(42)}"

        # Validate scopes
        valid_scopes = {scope.value for scope in CapabilityScope}
        for scope in scopes:
            if scope not in valid_scopes:
                raise ValueError(f"Invalid scope: {scope}")

        # Create token
        now = datetime.utcnow()
        token = CapabilityToken(
            id=str(uuid.uuid4()),
            token=token_value,
            organization_id=str(self.organization_id),
            session_id=str(session_id) if session_id else None,
            scopes=",".join(scopes),
            quota_tokens_per_month=max_tokens_per_month or 1_000_000,
            quota_storage_bytes=max_storage_bytes or 104_857_600,
            quota_requests_per_minute=max_requests_per_minute or 100,
            expires_at=now + timedelta(seconds=ttl_seconds),
            created_by_user_id=str(created_by_user_id) if created_by_user_id else None,
            token_metadata={
                "name": name,
                "agent_name": agent_name,
                "issued_to_user_id": str(issued_to_user_id) if issued_to_user_id else None,
                "session_id": str(session_id) if session_id else None,
            },
        )
        self.db.add(token)
        await self.db.flush()

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="capability.issued",
            actor_id=str(created_by_user_id) if created_by_user_id else None,
            organization_id=str(self.organization_id),
            resource_type="capability_token",
            resource_id=str(token.id),
            success=True,
            details={
                "token_name": name,
                "scopes": scopes,
                "agent_name": agent_name,
                "ttl_seconds": ttl_seconds
            }
        )

        logger.info(f"Issued capability token '{name}' with scopes {scopes} for org {self.organization_id}")
        return token

    async def revoke_token(
        self,
        token_id: uuid.UUID,
        reason: str,
        revoked_by_user_id: Optional[uuid.UUID] = None
    ) -> None:
        """Revoke a capability token."""
        stmt = select(CapabilityToken).where(
            CapabilityToken.id == token_id,
            CapabilityToken.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        token = result.scalar_one_or_none()

        if not token:
            raise ValueError("Token not found")

        if token.revoked_at is not None:
            raise ValueError("Token already revoked")

        # Revoke
        token.revoked_at = datetime.utcnow()
        token.revocation_reason = reason
        await self.db.flush()

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="capability.revoked",
            actor_id=str(revoked_by_user_id) if revoked_by_user_id else None,
            organization_id=str(self.organization_id),
            resource_type="capability_token",
            resource_id=str(token.id),
            success=True,
            details={
                "token_name": token.name,
                "revocation_reason": reason
            }
        )

        logger.info(f"Revoked capability token '{token.name}': {reason}")

    async def get_token(self, token_id: uuid.UUID) -> Optional[CapabilityToken]:
        """Get token by ID (org-scoped)."""
        stmt = select(CapabilityToken).where(
            CapabilityToken.id == token_id,
            CapabilityToken.organization_id == self.organization_id
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_tokens(
        self,
        agent_name: Optional[str] = None,
        include_revoked: bool = False
    ) -> list[CapabilityToken]:
        """List organization's capability tokens."""
        stmt = select(CapabilityToken).where(
            CapabilityToken.organization_id == self.organization_id
        )

        if agent_name:
            stmt = stmt.where(
                cast(CapabilityToken.token_metadata["agent_name"].astext, String) == agent_name
            )

        if not include_revoked:
            stmt = stmt.where(CapabilityToken.revoked_at.is_(None))

        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def update_quota(
        self,
        token_id: uuid.UUID,
        max_tokens_per_month: Optional[int] = None,
        max_storage_bytes: Optional[int] = None,
        max_requests_per_minute: Optional[int] = None,
        updated_by_user_id: Optional[uuid.UUID] = None
    ) -> CapabilityToken:
        """Update token quotas."""
        token = await self.get_token(token_id)
        if not token:
            raise ValueError("Token not found")

        old_quotas = {
            "max_tokens_per_month": token.max_tokens_per_month,
            "max_storage_bytes": token.max_storage_bytes,
            "max_requests_per_minute": token.max_requests_per_minute
        }

        if max_tokens_per_month is not None:
            token.max_tokens_per_month = max_tokens_per_month
        if max_storage_bytes is not None:
            token.max_storage_bytes = max_storage_bytes
        if max_requests_per_minute is not None:
            token.max_requests_per_minute = max_requests_per_minute

        await self.db.flush()

        # Audit log
        audit_svc = AuditService(self.db)
        await audit_svc.log_event(
            event_type="capability.updated",
            actor_id=str(updated_by_user_id) if updated_by_user_id else None,
            organization_id=str(self.organization_id),
            resource_type="capability_token",
            resource_id=str(token.id),
            success=True,
            details={
                "old_quotas": old_quotas,
                "new_quotas": {
                    "max_tokens_per_month": max_tokens_per_month,
                    "max_storage_bytes": max_storage_bytes,
                    "max_requests_per_minute": max_requests_per_minute
                }
            }
        )

        return token

    async def validate_token(self, token_value: str) -> Optional[CapabilityToken]:
        """
        Validate a capability token.
        
        Checks:
        - Token exists
        - Not expired
        - Not revoked
        - Within quota limits
        
        Returns token if valid, None if invalid.
        """
        from datetime import datetime
        
        stmt = select(CapabilityToken).where(
            CapabilityToken.token == token_value
        )
        result = await self.db.execute(stmt)
        token = result.scalar_one_or_none()
        
        if not token:
            logger.warning(f"Token validation failed: token not found")
            return None
        
        # Check if revoked
        if token.revoked_at:
            logger.warning(f"Token validation failed: token {token.id} was revoked at {token.revoked_at}")
            return None
        
        # Check expiration
        if token.expires_at and token.expires_at < datetime.utcnow():
            logger.warning(f"Token validation failed: token {token.id} expired at {token.expires_at}")
            return None
        
        # Check token quota
        if token.max_tokens_per_month and token.tokens_used_this_month >= token.max_tokens_per_month:
            logger.warning(f"Token validation failed: token {token.id} exceeded monthly token quota")
            return None
        
        # Check storage quota
        if token.max_storage_bytes and token.storage_used_bytes >= token.max_storage_bytes:
            logger.warning(f"Token validation failed: token {token.id} exceeded storage quota")
            return None
        
        # Check rate limit (requests per minute)
        # Note: This is a simple check. For production, use Redis for accurate rate limiting
        if token.max_requests_per_minute and token.requests_this_minute >= token.max_requests_per_minute:
            logger.warning(f"Token validation failed: token {token.id} exceeded rate limit")
            return None
        
        # Token is valid
        return token
