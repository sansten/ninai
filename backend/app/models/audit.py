"""
Audit Models
============

Models for audit events and memory access logs.
These tables are designed for compliance and are append-only.
"""

from typing import Optional
from datetime import datetime

from sqlalchemy import DateTime, String, Text, Boolean, Integer, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET

from app.models.base import Base, UUIDMixin, TimestampMixin


class AuditEvent(Base, UUIDMixin):
    """
    Audit event model (append-only).
    
    Records all security-relevant events for compliance.
    This table should be partitioned by month in production.
    
    Event types:
    - auth.login, auth.logout, auth.failed
    - user.create, user.update, user.delete
    - role.grant, role.revoke
    - memory.create, memory.read, memory.update, memory.delete
    - memory.share, memory.export
    - admin.* (various admin actions)
    
    Attributes:
        event_type: Category.action format
        actor_id: User who performed the action
        organization_id: Org context
        resource_type: Type of affected resource
        resource_id: ID of affected resource
        details: JSON payload with event-specific data
    """
    
    __tablename__ = "audit_events"
    
    # Timestamp (no updated_at for audit events - immutable)
    timestamp: Mapped[datetime] = mapped_column(
        nullable=False,
        index=True,
        doc="When the event occurred (UTC)",
    )
    
    # Event classification
    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Event type (category.action format)",
    )
    event_category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        doc="Event category (auth, user, memory, admin, etc.)",
    )
    severity: Mapped[str] = mapped_column(
        String(20),
        default="info",
        nullable=False,
        doc="Severity: debug, info, warning, error, critical",
    )
    
    # Actor (who)
    actor_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
        doc="User who performed the action",
    )
    actor_type: Mapped[str] = mapped_column(
        String(50),
        default="user",
        nullable=False,
        doc="Type of actor: user, system, agent",
    )
    
    # Organization context
    organization_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        index=True,
        doc="Organization context (if applicable)",
    )
    
    # Resource (what)
    resource_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Type of resource affected",
    )
    resource_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="ID of resource affected",
    )
    
    # Request context
    request_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Request ID for correlation",
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        INET,
        nullable=True,
        doc="Client IP address",
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Client user agent",
    )
    
    # Outcome
    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Whether the action succeeded",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Error message if action failed",
    )
    
    # Details (JSON)
    details: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Event-specific details (JSON)",
    )
    
    # Changes (for update events)
    changes: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        doc="Before/after values for changes",
    )
    
    __table_args__ = (
        # Composite index for common query patterns
        Index("ix_audit_org_time", "organization_id", "timestamp"),
        Index("ix_audit_actor_time", "actor_id", "timestamp"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        # GIN index for details searches
        Index("ix_audit_details", "details", postgresql_using="gin"),
    )
    
    def __repr__(self) -> str:
        return f"<AuditEvent {self.event_type} at {self.timestamp}>"


class MemoryAccessLog(Base, UUIDMixin):
    """
    Memory access log (append-only).
    
    Detailed log of every memory access attempt for compliance.
    This table should be partitioned by month in production.
    
    Attributes:
        memory_id: Memory that was accessed
        user_id: User who attempted access
        action: Action attempted (read, write, delete, share, export)
        authorized: Whether access was granted
        denial_reason: Reason if access was denied
    """
    
    __tablename__ = "memory_access_log"
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        doc="When access was attempted (UTC)",
    )
    
    # Memory reference
    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Memory that was accessed",
    )
    
    # User
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="User who attempted access",
    )
    
    # Organization context
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Organization context",
    )
    
    # Action
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Action: read, write, update, delete, share, export",
    )
    
    # Authorization result
    authorized: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        doc="Whether access was granted",
    )
    authorization_method: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="How authorization was determined (own, team, share, policy)",
    )
    denial_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Reason for denial (if not authorized)",
    )
    
    # Context
    access_context: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Additional context (search query, filters, etc.)",
    )
    
    # Request metadata
    request_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Request ID for correlation",
    )
    ip_address: Mapped[Optional[str]] = mapped_column(
        INET,
        nullable=True,
        doc="Client IP address",
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Client user agent",
    )
    
    # Justification (for sensitive access)
    justification: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Justification provided for sensitive access",
    )
    case_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        doc="Case/ticket ID for need-to-know access",
    )
    
    __table_args__ = (
        # Composite index for memory access history
        Index("ix_access_memory_time", "memory_id", "timestamp"),
        # Index for user access history
        Index("ix_access_user_time", "user_id", "timestamp"),
        # Index for org-wide access reports
        Index("ix_access_org_time", "organization_id", "timestamp"),
        # Index for audit queries on denied access
        Index("ix_access_denied", "authorized", "timestamp"),
    )
    
    def __repr__(self) -> str:
        status = "granted" if self.authorized else "denied"
        return f"<MemoryAccessLog {self.action} {status} at {self.timestamp}>"
