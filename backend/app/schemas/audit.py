"""
Audit Event Schemas and Types
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid
import enum
from datetime import datetime


class AuditEventType(str, enum.Enum):
    """Types of audit events."""
    # Knowledge events
    KNOWLEDGE_CREATED = "knowledge_created"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    KNOWLEDGE_DELETED = "knowledge_deleted"
    KNOWLEDGE_PUBLISHED = "knowledge_published"
    KNOWLEDGE_READ = "knowledge_read"
    KNOWLEDGE_SEARCHED = "knowledge_searched"
    KNOWLEDGE_CONSOLIDATED = "knowledge_consolidated"
    
    # Memory events
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    
    # Capability events
    CAPABILITY_ISSUED = "capability_issued"
    CAPABILITY_REVOKED = "capability_revoked"
    CAPABILITY_UPDATED = "capability_updated"
    CAPABILITY_USED = "capability_used"
    CAPABILITY_DENIED = "capability_denied"
    
    # User & Auth events
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_PASSWORD_CHANGED = "user_password_changed"
    
    # Organization events
    ORG_CREATED = "org_created"
    ORG_UPDATED = "org_updated"
    ORG_DELETED = "org_deleted"
    
    # Admin events
    ADMIN_ACTION = "admin_action"
    ADMIN_POLICY_CHANGED = "admin_policy_changed"
    
    # System events
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"


class AuditEventCreate(BaseModel):
    """Request to create an audit event."""
    event_type: AuditEventType = Field(..., description="Type of event")
    resource_type: str = Field(..., description="Resource type (knowledge, user, org, etc)")
    resource_id: Optional[uuid.UUID] = Field(None, description="Resource ID")
    action: str = Field(..., description="Specific action (create, delete, deny, etc)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    initiator_user_id: Optional[uuid.UUID] = Field(None, description="User who initiated action")
    ip_address: Optional[str] = Field(None, description="IP address of requester")
    user_agent: Optional[str] = Field(None, description="User agent string")


class AuditEventResponse(BaseModel):
    """Audit event response."""
    id: str
    event_type: str
    resource_type: str
    resource_id: Optional[str]
    action: str
    details: Optional[Dict[str, Any]]
    initiator_user_id: Optional[str]
    organization_id: str
    created_at: str
    ip_address: Optional[str]
