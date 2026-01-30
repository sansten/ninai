"""Role-Based Access Control (RBAC) utilities and decorators.

Provides capability-based access control for API endpoints.
"""

from functools import wraps
from typing import List, Optional

from fastapi import HTTPException, status
from app.models.user import User


# Standard capability definitions
class Capabilities:
    """Standard capability identifiers."""

    # Pipeline management
    PIPELINES_VIEW = "pipelines.view"
    PIPELINES_MANAGE = "pipelines.manage"
    PIPELINES_ADMIN = "pipelines.admin"  # Cancel, retry, priority changes

    # Queue operations
    QUEUES_VIEW = "queues.view"
    QUEUES_MANAGE = "queues.manage"

    # Resource budgets
    RESOURCES_VIEW = "resources.view"
    RESOURCES_MANAGE = "resources.manage"

    # Alerts
    ALERTS_VIEW = "alerts.view"
    ALERTS_MANAGE = "alerts.manage"

    # Observability
    OBSERVABILITY_VIEW = "observability.view"
    OBSERVABILITY_MANAGE = "observability.manage"

    # Dead Letter Queue
    DLQ_VIEW = "dlq.view"
    DLQ_MANAGE = "dlq.manage"  # Requeue, discard

    # System administration
    SYSTEM_ADMIN = "system.admin"
    USERS_MANAGE = "users.manage"
    ORGANIZATIONS_MANAGE = "organizations.manage"


# Role to capability mappings
ROLE_CAPABILITIES = {
    "admin": [
        # Admins have all capabilities
        Capabilities.PIPELINES_VIEW,
        Capabilities.PIPELINES_MANAGE,
        Capabilities.PIPELINES_ADMIN,
        Capabilities.QUEUES_VIEW,
        Capabilities.QUEUES_MANAGE,
        Capabilities.RESOURCES_VIEW,
        Capabilities.RESOURCES_MANAGE,
        Capabilities.ALERTS_VIEW,
        Capabilities.ALERTS_MANAGE,
        Capabilities.OBSERVABILITY_VIEW,
        Capabilities.OBSERVABILITY_MANAGE,
        Capabilities.DLQ_VIEW,
        Capabilities.DLQ_MANAGE,
        Capabilities.SYSTEM_ADMIN,
        Capabilities.USERS_MANAGE,
        Capabilities.ORGANIZATIONS_MANAGE,
    ],
    "operator": [
        # Operators can view and manage pipelines/queues
        Capabilities.PIPELINES_VIEW,
        Capabilities.PIPELINES_MANAGE,
        Capabilities.QUEUES_VIEW,
        Capabilities.QUEUES_MANAGE,
        Capabilities.RESOURCES_VIEW,
        Capabilities.ALERTS_VIEW,
        Capabilities.ALERTS_MANAGE,
        Capabilities.OBSERVABILITY_VIEW,
        Capabilities.DLQ_VIEW,
        Capabilities.DLQ_MANAGE,
    ],
    "viewer": [
        # Viewers can only view
        Capabilities.PIPELINES_VIEW,
        Capabilities.QUEUES_VIEW,
        Capabilities.RESOURCES_VIEW,
        Capabilities.ALERTS_VIEW,
        Capabilities.OBSERVABILITY_VIEW,
        Capabilities.DLQ_VIEW,
    ],
    "user": [
        # Regular users have minimal access
        Capabilities.PIPELINES_VIEW,
    ],
}


def get_user_capabilities(user: User) -> List[str]:
    """
    Get all capabilities for a user based on their role.
    
    Args:
        user: User instance
        
    Returns:
        List of capability strings
    """
    # For now, use role field. In future, could be many-to-many with roles table
    role = getattr(user, "role", "user")
    return ROLE_CAPABILITIES.get(role, ROLE_CAPABILITIES["user"])


def has_capability(user: User, capability: str) -> bool:
    """
    Check if user has a specific capability.
    
    Args:
        user: User instance
        capability: Capability string to check
        
    Returns:
        True if user has capability
    """
    capabilities = get_user_capabilities(user)
    
    # System admins have all capabilities
    if Capabilities.SYSTEM_ADMIN in capabilities:
        return True
    
    return capability in capabilities


def has_any_capability(user: User, capabilities: List[str]) -> bool:
    """
    Check if user has any of the specified capabilities.
    
    Args:
        user: User instance
        capabilities: List of capability strings
        
    Returns:
        True if user has at least one capability
    """
    user_capabilities = get_user_capabilities(user)
    
    # System admins have all capabilities
    if Capabilities.SYSTEM_ADMIN in user_capabilities:
        return True
    
    return any(cap in user_capabilities for cap in capabilities)


def has_all_capabilities(user: User, capabilities: List[str]) -> bool:
    """
    Check if user has all of the specified capabilities.
    
    Args:
        user: User instance
        capabilities: List of capability strings
        
    Returns:
        True if user has all capabilities
    """
    user_capabilities = get_user_capabilities(user)
    
    # System admins have all capabilities
    if Capabilities.SYSTEM_ADMIN in user_capabilities:
        return True
    
    return all(cap in user_capabilities for cap in capabilities)


def require_capability(capability: str):
    """
    Decorator to enforce capability requirement on endpoint.
    
    Usage:
        @router.get("/admin/pipelines")
        @require_capability(Capabilities.PIPELINES_VIEW)
        async def list_pipelines(current_user: User = Depends(get_current_user)):
            ...
    
    Args:
        capability: Required capability string
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get("current_user")
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                )
            
            if not has_capability(current_user, capability):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required capability: {capability}",
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def require_any_capability(capabilities: List[str]):
    """
    Decorator to enforce at least one capability requirement.
    
    Args:
        capabilities: List of acceptable capabilities
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                )
            
            if not has_any_capability(current_user, capabilities):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required capabilities. Need one of: {', '.join(capabilities)}",
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def require_all_capabilities(capabilities: List[str]):
    """
    Decorator to enforce all capabilities requirement.
    
    Args:
        capabilities: List of required capabilities
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated",
                )
            
            if not has_all_capabilities(current_user, capabilities):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required capabilities: {', '.join(capabilities)}",
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator
