"""
Audit Service
=============

Service for creating audit events and access logs.
All security-relevant actions should be logged through this service.
"""

from typing import Optional
from unittest.mock import Mock
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.audit import AuditEvent, MemoryAccessLog


class AuditService:
    """
    Audit logging service.
    
    Creates immutable audit records for security events
    and memory access attempts.
    
    Event Categories:
    - auth: Authentication events (login, logout, failed attempts)
    - user: User management (create, update, delete)
    - role: Role changes (grant, revoke)
    - memory: Memory operations (create, read, update, delete, share, export)
    - admin: Administrative actions
    - policy: Policy changes
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize audit service.
        
        Args:
            session: Database session for writing audit records
        """
        self.session = session
    
    # =========================================================================
    # Audit Events
    # =========================================================================
    
    async def log_event(
        self,
        event_type: str,
        actor_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        details: Optional[dict] = None,
        changes: Optional[dict] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        severity: str = "info",
        actor_type: str = "user",
    ) -> AuditEvent:
        """
        Create an audit event.
        
        Args:
            event_type: Event type in category.action format
            actor_id: User who performed the action
            organization_id: Organization context
            resource_type: Type of affected resource
            resource_id: ID of affected resource
            success: Whether the action succeeded
            error_message: Error message if failed
            details: Event-specific details
            changes: Before/after values for updates
            request_id: Request ID for correlation
            ip_address: Client IP address
            user_agent: Client user agent
            severity: Event severity (debug, info, warning, error, critical)
            actor_type: Type of actor (user, system, agent)
        
        Returns:
            Created AuditEvent
        """
        # Extract category from event_type
        category = event_type.split(".")[0] if "." in event_type else "unknown"
        
        event = AuditEvent(
            timestamp=datetime.utcnow(),  # Use timezone-naive for TIMESTAMP WITHOUT TIMEZONE
            event_type=event_type,
            event_category=category,
            severity=severity,
            actor_id=actor_id,
            actor_type=actor_type,
            organization_id=organization_id if organization_id else None,  # Convert empty string to None
            resource_type=resource_type,
            resource_id=resource_id if resource_id else None,  # Convert empty string to None
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            details=details or {},
            changes=changes,
        )
        
        self.session.add(event)
        await self.session.flush()

        # Emit webhook events for org-scoped audit events.
        if organization_id and isinstance(self.session, AsyncSession) and not isinstance(self.session, Mock):
            try:
                from app.services.webhook_service import WebhookService

                webhook_service = WebhookService(self.session)
                await webhook_service.emit_event(
                    organization_id=organization_id,
                    event_type=event_type,
                    payload={
                        "audit_event_id": event.id,
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "actor_id": actor_id,
                        "success": success,
                        "severity": severity,
                        "details": details or {},
                        "changes": changes,
                        "request_id": request_id,
                    },
                )
            except Exception:
                # Webhook emission must never break request paths.
                pass
        
        return event
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    async def log_auth_success(
        self,
        user_id: str,
        organization_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:
        """Log successful authentication."""
        return await self.log_event(
            event_type="auth.login",
            actor_id=user_id,
            organization_id=organization_id,
            success=True,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    
    async def log_auth_failure(
        self,
        email: str,
        reason: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:
        """Log failed authentication."""
        return await self.log_event(
            event_type="auth.failed",
            success=False,
            error_message=reason,
            details={"email": email},
            ip_address=ip_address,
            user_agent=user_agent,
            severity="warning",
        )
    
    async def log_role_grant(
        self,
        actor_id: str,
        user_id: str,
        role_name: str,
        organization_id: str,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        reason: Optional[str] = None,
    ) -> AuditEvent:
        """Log role assignment."""
        return await self.log_event(
            event_type="role.grant",
            actor_id=actor_id,
            organization_id=organization_id,
            resource_type="user",
            resource_id=user_id,
            success=True,
            details={
                "role_name": role_name,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "reason": reason,
            },
        )
    
    async def log_role_revoke(
        self,
        actor_id: str,
        user_id: str,
        role_name: str,
        organization_id: str,
        reason: Optional[str] = None,
    ) -> AuditEvent:
        """Log role revocation."""
        return await self.log_event(
            event_type="role.revoke",
            actor_id=actor_id,
            organization_id=organization_id,
            resource_type="user",
            resource_id=user_id,
            success=True,
            details={
                "role_name": role_name,
                "reason": reason,
            },
        )
    
    async def log_memory_operation(
        self,
        actor_id: str,
        organization_id: str,
        memory_id: str,
        operation: str,
        success: bool = True,
        error_message: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> AuditEvent:
        """Log memory operation (create, update, delete)."""
        return await self.log_event(
            event_type=f"memory.{operation}",
            actor_id=actor_id,
            organization_id=organization_id,
            resource_type="memory",
            resource_id=memory_id,
            success=success,
            error_message=error_message,
            details=details,
        )
    
    async def log_memory_share(
        self,
        actor_id: str,
        organization_id: str,
        memory_id: str,
        share_type: str,
        target_id: str,
        permission: str,
        expires_at: Optional[datetime] = None,
    ) -> AuditEvent:
        """Log memory sharing."""
        return await self.log_event(
            event_type="memory.share",
            actor_id=actor_id,
            organization_id=organization_id,
            resource_type="memory",
            resource_id=memory_id,
            success=True,
            details={
                "share_type": share_type,
                "target_id": target_id,
                "permission": permission,
                "expires_at": expires_at.isoformat() if expires_at else None,
            },
        )
    
    # =========================================================================
    # Memory Access Logging
    # =========================================================================
    
    async def log_memory_access(
        self,
        user_id: str,
        organization_id: str,
        memory_id: str,
        action: str,
        authorized: bool,
        authorization_method: str,
        denial_reason: Optional[str] = None,
        access_context: Optional[dict] = None,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        justification: Optional[str] = None,
        case_id: Optional[str] = None,
    ) -> MemoryAccessLog:
        """
        Log a memory access attempt.
        
        This creates a detailed record of every memory access
        for compliance and auditing.
        
        Args:
            user_id: User attempting access
            organization_id: Organization context
            memory_id: Memory being accessed
            action: Action type (read, write, delete, share, export)
            authorized: Whether access was granted
            authorization_method: How auth was determined
            denial_reason: Reason if denied
            access_context: Additional context (search query, etc.)
            request_id: Request ID for correlation
            ip_address: Client IP
            user_agent: Client user agent
            justification: Justification for sensitive access
            case_id: Case/ticket ID for need-to-know
        
        Returns:
            Created MemoryAccessLog
        """
        log = MemoryAccessLog(
            timestamp=datetime.now(timezone.utc),
            memory_id=memory_id,
            user_id=user_id,
            organization_id=organization_id,
            action=action,
            authorized=authorized,
            authorization_method=authorization_method,
            denial_reason=denial_reason,
            access_context=access_context or {},
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            justification=justification,
            case_id=case_id,
        )
        
        self.session.add(log)
        await self.session.flush()
        
        return log

    async def list_events(
        self,
        limit: int = 50,
        organization_id: Optional[str] = None,
    ) -> list[AuditEvent]:
        """List recent audit events, optionally filtered by organization."""

        stmt = select(AuditEvent).order_by(AuditEvent.timestamp.desc()).limit(limit)
        if organization_id:
            stmt = stmt.where(AuditEvent.organization_id == organization_id)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())
