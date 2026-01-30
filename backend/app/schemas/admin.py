"""Admin UI Pydantic schemas for request/response validation"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, validator


# ==================== ADMIN ROLE SCHEMAS ====================

class PermissionBase(BaseModel):
    """Base permission model"""
    resource: str = Field(..., description="Resource type (users, roles, settings, etc)")
    actions: List[str] = Field(..., description="Actions on resource (read, write, delete)")

    @validator("actions")
    def validate_actions(cls, v):
        valid_actions = {"read", "write", "delete", "manage"}
        if not all(action in valid_actions for action in v):
            raise ValueError(f"Invalid actions. Must be one of {valid_actions}")
        return v


class AdminRoleCreate(BaseModel):
    """Create admin role"""
    name: str = Field(..., min_length=1, max_length=100, description="Role name")
    description: Optional[str] = Field(None, description="Role description")
    permissions: List[str] = Field([], description="List of permissions (e.g., 'users:read', 'users:write')")

    @validator("permissions")
    def validate_permissions(cls, v):
        valid_permissions = {
            "users:read", "users:write", "users:delete",
            "roles:read", "roles:write", "roles:delete",
            "settings:read", "settings:write",
            "audit:read",
            "system:read",
            "mfa:manage",
            "webhooks:manage",
            "backups:manage",
            "incidents:manage",
        }
        for perm in v:
            if perm not in valid_permissions:
                raise ValueError(f"Invalid permission: {perm}. Valid permissions: {valid_permissions}")
        return v


class AdminRoleUpdate(BaseModel):
    """Update admin role"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


class AdminRoleResponse(BaseModel):
    """Admin role response"""
    id: UUID
    name: str
    description: Optional[str]
    permissions: List[str]
    is_system: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[UUID]

    class Config:
        from_attributes = True


# ==================== ADMIN SETTING SCHEMAS ====================

class AdminSettingCreate(BaseModel):
    """Create admin setting"""
    category: str = Field(..., min_length=1, max_length=50)
    key: str = Field(..., min_length=1, max_length=255)
    value: Any = Field(..., description="Setting value (any JSON type)")
    type: Optional[str] = Field(None, description="Setting type: string, number, boolean, json")
    description: Optional[str] = None
    is_secret: bool = Field(False, description="Whether value should be hidden")


class AdminSettingUpdate(BaseModel):
    """Update admin setting"""
    value: Optional[Any] = None
    description: Optional[str] = None
    is_secret: Optional[bool] = None


class AdminSettingResponse(BaseModel):
    """Admin setting response"""
    id: UUID
    category: str
    key: str
    value: Any = Field(None, description="Omitted if is_secret=True")
    type: Optional[str]
    description: Optional[str]
    is_secret: bool
    updated_at: datetime
    updated_by: Optional[UUID]

    class Config:
        from_attributes = True

    def dict(self, **kwargs):
        """Override to hide secret values"""
        data = super().dict(**kwargs)
        if self.is_secret:
            data["value"] = "***REDACTED***"
        return data


class AdminSettingListResponse(BaseModel):
    """List of admin settings"""
    items: List[AdminSettingResponse]
    total: int
    category: Optional[str]


# ==================== ADMIN AUDIT LOG SCHEMAS ====================

class AdminAuditLogResponse(BaseModel):
    """Admin audit log response"""
    id: UUID
    admin_id: UUID
    action: str
    resource_type: str
    resource_id: Optional[str]
    old_values: Optional[Dict[str, Any]]
    new_values: Optional[Dict[str, Any]]
    ip_address: Optional[str]
    user_agent: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class AdminAuditLogListResponse(BaseModel):
    """List of admin audit logs"""
    items: List[AdminAuditLogResponse]
    total: int
    page: int
    page_size: int


class AdminAuditLogFilter(BaseModel):
    """Filter for audit logs"""
    admin_id: Optional[str] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


# ==================== ADMIN SESSION SCHEMAS ====================

class AdminSessionResponse(BaseModel):
    """Admin session response"""
    id: UUID
    admin_id: UUID
    ip_address: str
    user_agent: Optional[str]
    expires_at: datetime
    last_activity: datetime
    created_at: datetime
    is_expired: bool

    class Config:
        from_attributes = True

    def dict(self, **kwargs):
        data = super().dict(**kwargs)
        data["is_expired"] = self.expires_at <= datetime.utcnow()
        return data


# ==================== ADMIN IP WHITELIST SCHEMAS ====================

class AdminIPWhitelistCreate(BaseModel):
    """Create IP whitelist entry"""
    ip_address: str = Field(..., description="IP address (IPv4 or IPv6 CIDR)")
    description: Optional[str] = None


class AdminIPWhitelistResponse(BaseModel):
    """IP whitelist response"""
    id: UUID
    ip_address: str
    description: Optional[str]
    created_by: Optional[UUID]
    created_at: datetime

    class Config:
        from_attributes = True


# ==================== ADMIN DASHBOARD SCHEMAS ====================

class DashboardKPI(BaseModel):
    """KPI card for dashboard"""
    label: str
    value: str
    unit: Optional[str] = None
    trend: Optional[str] = None  # up, down, stable
    change_percent: Optional[float] = None


class ServiceHealthStatus(BaseModel):
    """Service health status"""
    name: str
    status: str  # healthy, degraded, unhealthy
    message: Optional[str] = None
    last_check: datetime


class DashboardResponse(BaseModel):
    """Admin dashboard response"""
    timestamp: datetime
    kpis: List[DashboardKPI]
    services: List[ServiceHealthStatus]
    alerts_count: int
    recent_activities: List[AdminAuditLogResponse]


# ==================== USER MANAGEMENT SCHEMAS ====================

class UserCreateByAdmin(BaseModel):
    """Admin creating a new user"""
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=255)
    admin_role_id: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8, description="If not provided, temporary password sent via email")
    admin_notes: Optional[str] = None


class UserUpdateByAdmin(BaseModel):
    """Admin updating user"""
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)
    admin_role_id: Optional[str] = None
    admin_notes: Optional[str] = None


class UserAdminResponse(BaseModel):
    """User response for admin"""
    id: str
    email: str
    full_name: str
    is_active: bool
    is_admin: bool
    admin_role_id: Optional[str]
    admin_notes: Optional[str]
    last_login: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    last_admin_action_at: Optional[datetime]
    last_admin_action_by: Optional[str]

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """List of users"""
    items: List[UserAdminResponse]
    total: int
    page: int
    page_size: int


# ==================== PERMISSION SCHEMA ====================

class PermissionInfo(BaseModel):
    """Permission information"""
    permission: str = Field(..., description="Permission code (e.g., 'users:read')")
    description: str
    category: str  # users, roles, settings, audit, system, mfa, webhooks, backups, incidents


class PermissionsResponse(BaseModel):
    """List of available permissions"""
    permissions: List[PermissionInfo]


# ==================== DASHBOARD SCHEMAS ====================

class DashboardKPIResponse(BaseModel):
    """Dashboard KPI metric"""
    label: str
    value: str
    unit: Optional[str] = None
    trend: Optional[str] = Field(None, description="up, down, or stable")
    change_percent: Optional[float] = None


class ServiceHealthResponse(BaseModel):
    """Service health status"""
    name: str
    status: str = Field(..., description="healthy, degraded, or unhealthy")
    message: Optional[str] = None
    last_check: datetime


class DashboardActivityResponse(BaseModel):
    """Recent admin activity"""
    id: str
    admin_id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    timestamp: datetime
    description: str
    status: str


class DashboardKPIsResponse(BaseModel):
    """All KPIs grouped by category"""
    users: List[DashboardKPIResponse]
    memories: List[DashboardKPIResponse]
    system: List[DashboardKPIResponse]


class DashboardResponse(BaseModel):
    """Complete dashboard data"""
    timestamp: datetime
    kpis: DashboardKPIsResponse
    services: List[ServiceHealthResponse]
    recent_activities: List[DashboardActivityResponse]
    alerts: List[Dict[str, Any]] = []

    class Config:
        from_attributes = True


# ==================== ERROR RESPONSES ====================

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    status_code: int
