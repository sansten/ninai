"""
Tenant Context Dependency
=========================

FastAPI dependency for extracting and validating tenant context
from JWT tokens and setting up database session variables.
"""

from typing import Optional
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import verify_token
from app.core.database import get_db, set_tenant_context
from app.services.api_key_service import ApiKeyService


# Security scheme for Bearer tokens
security = HTTPBearer(auto_error=False)


@dataclass
class TenantContext:
    """
    Tenant context for the current request.
    
    Contains all information needed for authorization decisions
    and database session configuration.
    
    Attributes:
        user_id: UUID of the authenticated user
        org_id: UUID of the current organization
        roles: List of role names for the user in this org
        clearance_level: User's security clearance (0-4)
        is_authenticated: Whether user has valid auth
        capabilities: Set of capabilities granted to this user
    """
    user_id: str
    org_id: str
    roles: list[str] = None
    clearance_level: int = 0
    is_authenticated: bool = True
    capabilities: set[str] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.capabilities is None:
            # Default capabilities based on roles
            self.capabilities = self._derive_capabilities_from_roles()
    
    def _derive_capabilities_from_roles(self) -> set[str]:
        """
        Derive capabilities from user's roles.
        
        Maps roles to their allowed capabilities for fine-grained access control.
        System admin and org admin get all capabilities by default.
        """
        capabilities = set()
        
        # System admins get all capabilities
        if self.has_role("system_admin"):
            return {
                "canManageQueues",
                "canViewLogs",
                "canManageWebhooks",
                "canToggleMaintenance",
                "canManageAlerts",
                "canManagePolicies",
                "canManageBackups",
                "canViewMetrics",
            }
        
        # Org admins get all capabilities
        if self.has_role("org_admin"):
            return {
                "canManageQueues",
                "canViewLogs",
                "canManageWebhooks",
                "canToggleMaintenance",
                "canManageAlerts",
                "canManagePolicies",
                "canManageBackups",
                "canViewMetrics",
            }
        
        # Knowledge reviewers can view logs and metrics
        if self.has_role("knowledge_reviewer"):
            capabilities.add("canViewLogs")
            capabilities.add("canViewMetrics")
        
        return capabilities
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_any_role(self, *roles: str) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)
    
    def has_capability(self, capability: str) -> bool:
        """Check if user has a specific capability."""
        return capability in self.capabilities
    
    def has_any_capability(self, *capabilities: str) -> bool:
        """Check if user has any of the specified capabilities."""
        return any(cap in self.capabilities for cap in capabilities)
    
    @property
    def is_system_admin(self) -> bool:
        """Check if user is a system administrator."""
        return self.has_role("system_admin")
    
    @property
    def is_org_admin(self) -> bool:
        """Check if user is an organization administrator."""
        return self.has_any_role("org_admin", "system_admin")
    
    @property
    def roles_string(self) -> str:
        """Get roles as comma-separated string for DB session."""
        return ",".join(self.roles)


class AnonymousContext(TenantContext):
    """Context for unauthenticated requests."""
    
    def __init__(self):
        super().__init__(
            user_id="",
            org_id="",
            roles=[],
            is_authenticated=False,
        )


async def get_optional_tenant_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> TenantContext:
    """
    Get tenant context without requiring authentication.
    
    Use this for endpoints that work differently for authenticated
    vs anonymous users.
    
    Returns:
        TenantContext or AnonymousContext
    """
    if x_api_key:
        user_id, org_id, roles, clearance = await ApiKeyService.authenticate_api_key(db, x_api_key)
        return TenantContext(user_id=user_id, org_id=org_id, roles=roles, clearance_level=clearance)

    if not credentials:
        return AnonymousContext()
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        return AnonymousContext()
    
    return TenantContext(
        user_id=token_data.user_id,
        org_id=token_data.org_id,
        roles=token_data.roles,
    )


async def get_tenant_context(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> TenantContext:
    """
    Get tenant context from JWT token (required authentication).
    
    This is the primary dependency for protected endpoints.
    Extracts user_id, org_id, and roles from the JWT token.
    
    Returns:
        TenantContext with authenticated user data
    
    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if x_api_key:
        user_id, org_id, roles, clearance = await ApiKeyService.authenticate_api_key(db, x_api_key)
        return TenantContext(user_id=user_id, org_id=org_id, roles=roles, clearance_level=clearance)

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return TenantContext(
        user_id=token_data.user_id,
        org_id=token_data.org_id,
        roles=token_data.roles,
    )


async def get_db_with_tenant(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> AsyncSession:
    """
    Get database session with tenant context variables set.
    
    This dependency combines authentication with database session
    setup, ensuring RLS policies have the correct context.
    
    IMPORTANT: Must be used within a transaction block!
    
    Returns:
        AsyncSession with tenant context configured
    """
    await set_tenant_context(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        roles=tenant.roles_string,
        clearance_level=tenant.clearance_level,
    )
    return db


def require_roles(*required_roles: str):
    """
    Dependency factory for role-based access control.
    
    Creates a dependency that verifies the user has at least
    one of the specified roles.
    
    Args:
        required_roles: Role names that grant access
    
    Returns:
        Dependency function
    
    Example:
        @router.get("/admin")
        async def admin_endpoint(
            tenant: TenantContext = Depends(require_roles("admin", "super_admin"))
        ):
            ...
    """
    async def role_checker(
        tenant: TenantContext = Depends(get_tenant_context),
    ) -> TenantContext:
        if not tenant.has_any_role(*required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(required_roles)}. User has: {', '.join(tenant.roles) if tenant.roles else 'none'}",
            )
        return tenant
    
    return role_checker


def require_system_admin():
    """Dependency that requires system admin role."""
    return require_roles("system_admin")


def require_org_admin():
    """Dependency that requires org admin role."""
    return require_roles("org_admin", "system_admin")


def require_knowledge_reviewer():
    """Dependency that grants access to knowledge review queue.

    Knowledge reviewers are non-admin information workers who can approve/reject
    knowledge submissions without having access to admin settings.
    """
    return require_roles("knowledge_reviewer", "org_admin", "system_admin")


def require_capability(*required_capabilities: str):
    """
    Dependency factory for capability-based access control.
    
    Creates a dependency that verifies the user has at least
    one of the specified capabilities.
    
    Capabilities provide fine-grained access control and can be
    granted independently of roles for specialized use cases.
    
    Args:
        required_capabilities: Capability names that grant access
    
    Returns:
        Dependency function
    
    Example:
        @router.post("/admin/queues/pause")
        async def pause_queue(
            tenant: TenantContext = Depends(require_capability("canManageQueues"))
        ):
            ...
    
    Built-in Capabilities:
        - canManageQueues: Queue pause, resume, drain operations
        - canViewLogs: Access to application logs
        - canManageWebhooks: Create/update/delete webhooks
        - canToggleMaintenance: Enable/disable maintenance mode
        - canManageAlerts: Create/update/disable alert rules
        - canManagePolicies: Deploy, promote, activate policies
        - canManageBackups: Create snapshots, restore, verify
        - canViewMetrics: Access metrics and prometheus endpoints
    """
    async def capability_checker(
        tenant: TenantContext = Depends(get_tenant_context),
    ) -> TenantContext:
        if not tenant.has_any_capability(*required_capabilities):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of capabilities: {', '.join(required_capabilities)}. User has: {', '.join(tenant.capabilities) if tenant.capabilities else 'none'}",
            )
        return tenant
    
    return capability_checker
