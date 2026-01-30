"""Admin authentication and permission checking dependencies"""
from typing import List, Optional, Set
from datetime import datetime, timedelta
from functools import wraps
from uuid import uuid4
import hashlib
import secrets

from fastapi import Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.admin import AdminRole, AdminSession, AdminIPWhitelist
from app.core.security import verify_token
from app.core.config import settings


class AdminUser:
    """Admin user with permissions"""
    
    def __init__(self, user: User, role: Optional[AdminRole] = None):
        self.user = user
        self.role = role
        self.permissions = set(role.permissions) if role else set()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        if self.user.is_admin and not self.permissions:
            return True
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if user has any of the permissions"""
        if self.user.is_admin and not self.permissions:
            return True
        return any(p in self.permissions for p in permissions)
    
    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if user has all permissions"""
        if self.user.is_admin and not self.permissions:
            return True
        return all(p in self.permissions for p in permissions)


async def get_admin_user(
    request: Request,
    db: Session = Depends(get_db),
) -> AdminUser:
    """
    Dependency to get authenticated admin user with permissions
    
    Raises:
        HTTPException: If user not authenticated or not admin
    """
    # Get token from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = auth_header.split(" ")[1]
    
    # Verify token and get user
    try:
        payload = verify_token(token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        # TokenData is a pydantic model; use attributes instead of .get
        user_id = getattr(payload, "user_id", None)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
        )
    
    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Check if user is admin
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have admin privileges",
        )
    
    # Get admin role and permissions
    role = None
    if user.admin_role_id:
        role = db.query(AdminRole).filter(
            AdminRole.id == user.admin_role_id
        ).first()
    
    # Check IP whitelist if enabled
    client_ip = request.client.host if request.client else None
    if client_ip:
        await check_admin_ip_whitelist(client_ip, db)
    
    # Check admin session
    await verify_admin_session(user_id, token, client_ip, db)
    
    return AdminUser(user, role)


async def check_admin_ip_whitelist(ip_address: str, db: Session) -> None:
    """Check if IP is in admin whitelist (if enabled)"""
    # Get whitelist setting
    from app.models.admin import AdminSetting
    
    whitelist_enabled = db.query(AdminSetting).filter(
        AdminSetting.category == "security",
        AdminSetting.key == "admin_ip_whitelist_enabled"
    ).first()
    
    if not whitelist_enabled or not whitelist_enabled.value:
        return
    
    # Check if IP is in whitelist
    ip_allowed = db.query(AdminIPWhitelist).filter(
        AdminIPWhitelist.ip_address == ip_address
    ).first()
    
    if not ip_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"IP address {ip_address} is not whitelisted for admin access",
        )


async def verify_admin_session(
    user_id: str,
    token: str,
    ip_address: Optional[str],
    db: Session
) -> None:
    """Verify admin session is valid"""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    
    session = db.query(AdminSession).filter(
        AdminSession.admin_id == user_id,
        AdminSession.token_hash == token_hash
    ).first()
    
    if not session:
        # Create new session
        timeout_hours = getattr(settings, "ADMIN_SESSION_TIMEOUT_HOURS", 12)
        session = AdminSession(
            id=str(uuid4()),
            admin_id=user_id,
            token_hash=token_hash,
            ip_address=ip_address or "0.0.0.0",
            expires_at=datetime.utcnow() + timedelta(hours=timeout_hours)
        )
        db.add(session)
        db.commit()
    elif session.is_expired():
        db.delete(session)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin session expired",
        )
    else:
        # Update last activity
        session.last_activity = datetime.utcnow()
        db.commit()


def require_admin_permission(permission: str):
    """
    Decorator to require specific admin permission
    
    Usage:
        @require_admin_permission("users:write")
        async def create_user(admin: AdminUser = Depends(get_admin_user)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, admin: AdminUser = Depends(get_admin_user), **kwargs):
            if not admin.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission}",
                )
            return await func(*args, admin=admin, **kwargs)
        return wrapper
    return decorator


def require_admin_any_permission(*permissions: str):
    """
    Decorator to require any of the specified permissions
    
    Usage:
        @require_admin_any_permission("users:write", "users:delete")
        async def manage_user(admin: AdminUser = Depends(get_admin_user)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, admin: AdminUser = Depends(get_admin_user), **kwargs):
            if not admin.has_any_permission(list(permissions)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing any of required permissions: {permissions}",
                )
            return await func(*args, admin=admin, **kwargs)
        return wrapper
    return decorator


def require_admin_all_permissions(*permissions: str):
    """
    Decorator to require all specified permissions
    
    Usage:
        @require_admin_all_permissions("users:write", "roles:write")
        async def modify_user_role(admin: AdminUser = Depends(get_admin_user)):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, admin: AdminUser = Depends(get_admin_user), **kwargs):
            if not admin.has_all_permissions(list(permissions)):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing all required permissions: {permissions}",
                )
            return await func(*args, admin=admin, **kwargs)
        return wrapper
    return decorator


class PermissionMatrix:
    """Admin permission matrix"""
    
    VALID_PERMISSIONS = {
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
    
    # Default role permissions
    DEFAULT_ROLES = {
        "super_admin": [
            "users:read", "users:write", "users:delete",
            "roles:read", "roles:write", "roles:delete",
            "settings:read", "settings:write",
            "audit:read",
            "system:read",
            "mfa:manage",
            "webhooks:manage",
            "backups:manage",
            "incidents:manage",
        ],
        "admin": [
            "users:read", "users:write",
            "settings:read",
            "audit:read",
            "system:read",
        ],
        "operator": [
            "users:read",
            "system:read",
            "audit:read",
        ],
    }
    
    @classmethod
    def get_permissions_by_category(cls) -> dict:
        """Get permissions grouped by category"""
        categories = {
            "users": ["read", "write", "delete"],
            "roles": ["read", "write", "delete"],
            "settings": ["read", "write"],
            "audit": ["read"],
            "system": ["read"],
            "mfa": ["manage"],
            "webhooks": ["manage"],
            "backups": ["manage"],
            "incidents": ["manage"],
        }
        return categories
    
    @classmethod
    def validate_permissions(cls, permissions: List[str]) -> bool:
        """Validate permissions against valid set"""
        return all(p in cls.VALID_PERMISSIONS for p in permissions)
