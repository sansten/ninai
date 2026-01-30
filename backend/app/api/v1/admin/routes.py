"""Admin API routes"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.api.v1.admin.dependencies import (
    get_admin_user, AdminUser, require_admin_permission, PermissionMatrix
)
from app.services.admin import (
    AdminRoleService, AdminSettingService, AdminAuditService,
    AdminUserService, AdminIPWhitelistService
)
from app.services.dashboard import DashboardService
from app.schemas.admin import (
    AdminRoleCreate, AdminRoleUpdate, AdminRoleResponse,
    AdminSettingCreate, AdminSettingUpdate, AdminSettingResponse, AdminSettingListResponse,
    AdminAuditLogResponse, AdminAuditLogListResponse, AdminAuditLogFilter,
    UserCreateByAdmin, UserUpdateByAdmin, UserAdminResponse, UserListResponse,
    AdminIPWhitelistCreate, AdminIPWhitelistResponse,
    PermissionsResponse, PermissionInfo, DashboardResponse, DashboardKPI,
    ServiceHealthStatus
)

router = APIRouter(prefix="/admin", tags=["admin"])


# ==================== DASHBOARD ENDPOINTS ====================

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """
    Get admin dashboard with KPIs, service health, and recent activities.
    Requires: system:read permission
    """
    if not admin.has_permission("system:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: system:read"
        )
    
    from datetime import datetime
    
    # Get KPI data using dashboard service
    user_kpis = DashboardService.get_user_kpis(db)
    memory_kpis = DashboardService.get_memory_kpis(db)
    system_kpis = DashboardService.get_system_kpis(db)
    
    # Convert KPI objects to response format
    kpis_response = {
        "users": [
            {
                "label": kpi.label,
                "value": kpi.value,
                "unit": kpi.unit,
                "trend": kpi.trend,
                "change_percent": kpi.change_percent
            }
            for kpi in user_kpis
        ],
        "memories": [
            {
                "label": kpi.label,
                "value": kpi.value,
                "unit": kpi.unit,
                "trend": kpi.trend,
                "change_percent": kpi.change_percent
            }
            for kpi in memory_kpis
        ],
        "system": [
            {
                "label": kpi.label,
                "value": kpi.value,
                "unit": kpi.unit,
                "trend": kpi.trend,
                "change_percent": kpi.change_percent
            }
            for kpi in system_kpis
        ]
    }
    
    # Get service health
    services = DashboardService.get_service_health(db)
    services_response = [
        {
            "name": svc.name,
            "status": svc.status,
            "message": svc.message,
            "last_check": svc.last_check
        }
        for svc in services
    ]
    
    # Get recent activities
    activities = DashboardService.get_recent_activities(db, limit=10)
    
    return DashboardResponse(
        timestamp=datetime.utcnow(),
        kpis=kpis_response,
        services=services_response,
        recent_activities=activities,
        alerts=[]
    )


# ==================== ROLE ENDPOINTS ====================

@router.post("/roles", response_model=AdminRoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_create: AdminRoleCreate,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Create new admin role"""
    if not admin.has_permission("roles:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: roles:write"
        )
    
    try:
        role = AdminRoleService.create_role(db, role_create, admin.user.id)
        
        # Log action
        AdminAuditService.log_action(
            db,
            admin_id=admin.user.id,
            action="create",
            resource_type="role",
            resource_id=role.id,
            new_values={"name": role.name, "permissions": role.permissions}
        )
        
        return AdminRoleResponse.from_orm(role)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/roles", response_model=List[AdminRoleResponse])
async def list_roles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """List all admin roles"""
    if not admin.has_permission("roles:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: roles:read"
        )
    
    roles, _ = AdminRoleService.list_roles(db, skip, limit)
    return [AdminRoleResponse.from_orm(role) for role in roles]


@router.get("/roles/{role_id}", response_model=AdminRoleResponse)
async def get_role(
    role_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Get role by ID"""
    if not admin.has_permission("roles:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: roles:read"
        )
    
    role = AdminRoleService.get_role(db, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found"
        )
    
    return AdminRoleResponse.from_orm(role)


@router.put("/roles/{role_id}", response_model=AdminRoleResponse)
async def update_role(
    role_id: str,
    role_update: AdminRoleUpdate,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Update admin role"""
    if not admin.has_permission("roles:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: roles:write"
        )
    
    role = AdminRoleService.get_role(db, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found"
        )
    
    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify system role"
        )
    
    old_values = {
        "name": role.name,
        "permissions": role.permissions
    }
    
    updated_role = AdminRoleService.update_role(db, role_id, role_update)
    
    # Log action
    AdminAuditService.log_action(
        db,
        admin_id=admin.user.id,
        action="update",
        resource_type="role",
        resource_id=role_id,
        old_values=old_values,
        new_values={
            "name": updated_role.name,
            "permissions": updated_role.permissions
        }
    )
    
    return AdminRoleResponse.from_orm(updated_role)


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    role_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Delete admin role"""
    if not admin.has_permission("roles:delete"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: roles:delete"
        )
    
    role = AdminRoleService.get_role(db, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found"
        )
    
    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete system role"
        )
    
    try:
        AdminRoleService.delete_role(db, role_id)
        
        # Log action
        AdminAuditService.log_action(
            db,
            admin_id=admin.user.id,
            action="delete",
            resource_type="role",
            resource_id=role_id,
            old_values={"name": role.name}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )


# ==================== SETTINGS ENDPOINTS ====================

@router.post("/settings", response_model=AdminSettingResponse, status_code=status.HTTP_201_CREATED)
async def create_setting(
    setting_create: AdminSettingCreate,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Create new setting"""
    if not admin.has_permission("settings:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: settings:write"
        )
    
    try:
        setting = AdminSettingService.create_setting(db, setting_create, admin.user.id)
        
        AdminAuditService.log_action(
            db,
            admin_id=admin.user.id,
            action="create",
            resource_type="setting",
            resource_id=f"{setting.category}:{setting.key}",
            new_values={"value": setting.value if not setting.is_secret else "***REDACTED***"}
        )
        
        return AdminSettingResponse.from_orm(setting)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/settings", response_model=AdminSettingListResponse)
async def list_settings(
    category: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """List settings"""
    if not admin.has_permission("settings:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: settings:read"
        )
    
    settings, total = AdminSettingService.list_settings(db, category, skip, limit)
    
    return AdminSettingListResponse(
        items=[AdminSettingResponse.from_orm(s) for s in settings],
        total=total,
        category=category
    )


@router.get("/settings/{setting_id}", response_model=AdminSettingResponse)
async def get_setting(
    setting_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Get setting by ID"""
    if not admin.has_permission("settings:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: settings:read"
        )
    
    setting = AdminSettingService.get_setting_by_id(db, setting_id)
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Setting not found"
        )
    
    return AdminSettingResponse.from_orm(setting)


@router.put("/settings/{setting_id}", response_model=AdminSettingResponse)
async def update_setting(
    setting_id: str,
    setting_update: AdminSettingUpdate,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Update setting"""
    if not admin.has_permission("settings:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: settings:write"
        )
    
    setting = AdminSettingService.get_setting_by_id(db, setting_id)
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Setting not found"
        )
    
    old_value = setting.value
    
    updated_setting = AdminSettingService.update_setting(db, setting_id, setting_update, admin.user.id)
    
    AdminAuditService.log_action(
        db,
        admin_id=admin.user.id,
        action="update",
        resource_type="setting",
        resource_id=f"{setting.category}:{setting.key}",
        old_values={"value": old_value if not setting.is_secret else "***REDACTED***"},
        new_values={"value": updated_setting.value if not updated_setting.is_secret else "***REDACTED***"}
    )
    
    return AdminSettingResponse.from_orm(updated_setting)


@router.delete("/settings/{setting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_setting(
    setting_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Delete setting"""
    if not admin.has_permission("settings:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: settings:write"
        )
    
    setting = AdminSettingService.get_setting_by_id(db, setting_id)
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Setting not found"
        )
    
    AdminAuditService.log_action(
        db,
        admin_id=admin.user.id,
        action="delete",
        resource_type="setting",
        resource_id=f"{setting.category}:{setting.key}"
    )
    
    AdminSettingService.delete_setting(db, setting_id)


# ==================== AUDIT LOG ENDPOINTS ====================

@router.get("/audit-logs", response_model=AdminAuditLogListResponse)
async def list_audit_logs(
    admin_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """List audit logs"""
    if not admin.has_permission("audit:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: audit:read"
        )
    
    filter_params = AdminAuditLogFilter(
        admin_id=admin_id,
        action=action,
        resource_type=resource_type,
    )
    
    logs, total = AdminAuditService.list_audit_logs(db, filter_params, skip, limit)
    
    return AdminAuditLogListResponse(
        items=[AdminAuditLogResponse.from_orm(log) for log in logs],
        total=total,
        page=skip // limit + 1,
        page_size=limit
    )


@router.get("/audit-logs/{log_id}", response_model=AdminAuditLogResponse)
async def get_audit_log(
    log_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Get audit log by ID"""
    if not admin.has_permission("audit:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: audit:read"
        )
    
    log = AdminAuditService.get_audit_log(db, log_id)
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit log not found"
        )
    
    return AdminAuditLogResponse.from_orm(log)


# ==================== PERMISSIONS ENDPOINT ====================

@router.get("/permissions", response_model=PermissionsResponse)
async def get_permissions(
    admin: AdminUser = Depends(get_admin_user),
):
    """Get available permissions"""
    
    permissions_list = []
    matrix = PermissionMatrix.get_permissions_by_category()
    
    descriptions = {
        "users:read": "View user accounts",
        "users:write": "Create and edit user accounts",
        "users:delete": "Delete user accounts",
        "roles:read": "View admin roles",
        "roles:write": "Create and edit admin roles",
        "roles:delete": "Delete admin roles",
        "settings:read": "View system settings",
        "settings:write": "Edit system settings",
        "audit:read": "View audit logs",
        "system:read": "View system health and status",
        "mfa:manage": "Manage MFA settings",
        "webhooks:manage": "Manage webhooks",
        "backups:manage": "Manage backups",
        "incidents:manage": "Manage incidents",
    }
    
    for category, actions in matrix.items():
        for action in actions:
            perm = f"{category}:{action}"
            permissions_list.append(PermissionInfo(
                permission=perm,
                description=descriptions.get(perm, f"{action} {category}"),
                category=category
            ))
    
    return PermissionsResponse(permissions=permissions_list)


# ==================== USER MANAGEMENT ENDPOINTS ====================

@router.get("/users", response_model=UserListResponse)
async def list_users(
    search: Optional[str] = Query(None),
    role_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """List users"""
    if not admin.has_permission("users:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: users:read"
        )
    
    users, total = AdminUserService.list_users(db, search, role_id, skip, limit)
    
    return UserListResponse(
        items=[UserAdminResponse.from_orm(u) for u in users],
        total=total,
        page=skip // limit + 1,
        page_size=limit
    )


@router.get("/users/{user_id}", response_model=UserAdminResponse)
async def get_user(
    user_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Get user by ID"""
    if not admin.has_permission("users:read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: users:read"
        )
    
    user = AdminUserService.get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserAdminResponse.from_orm(user)


@router.post("/users/{user_id}/disable", response_model=UserAdminResponse)
async def disable_user(
    user_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Disable user account"""
    if not admin.has_permission("users:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: users:write"
        )
    
    user = AdminUserService.disable_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    AdminAuditService.log_action(
        db,
        admin_id=admin.user.id,
        action="disable",
        resource_type="user",
        resource_id=user_id
    )
    
    return UserAdminResponse.from_orm(user)


@router.post("/users/{user_id}/enable", response_model=UserAdminResponse)
async def enable_user(
    user_id: str,
    admin: AdminUser = Depends(get_admin_user),
    db: Session = Depends(get_db),
):
    """Enable user account"""
    if not admin.has_permission("users:write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing permission: users:write"
        )
    
    user = AdminUserService.enable_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    AdminAuditService.log_action(
        db,
        admin_id=admin.user.id,
        action="enable",
        resource_type="user",
        resource_id=user_id
    )
    
    return UserAdminResponse.from_orm(user)


from datetime import datetime
