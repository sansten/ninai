"""
User Endpoints
==============

User management and role assignment operations.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.core.security import get_password_hash
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    require_roles,
)
from app.models.user import User, Role, UserRole
from app.schemas.base import PaginatedResponse
from app.schemas.base import BaseSchema


router = APIRouter()


# =============================================================================
# User Schemas (local to this module)
# =============================================================================

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    """Create a new user."""
    
    email: EmailStr
    external_id: Optional[str] = None
    display_name: str = Field(..., min_length=1, max_length=255)
    default_organization_id: str
    


class UserUpdate(BaseModel):
    """Update user fields."""
    
    display_name: Optional[str] = Field(None, min_length=1, max_length=255)
    avatar_url: Optional[str] = None
    preferences: Optional[dict] = None
    is_active: Optional[bool] = None


class UserRoleAssign(BaseModel):
    """Assign a role to a user."""
    
    role_id: str
    scope_type: str = Field(..., pattern="^(global|organization|department|team)$")
    scope_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    granted_reason: Optional[str] = None


class UserRoleResponse(BaseModel):
    """User role response."""
    
    id: str
    user_id: str
    role_id: str
    role_name: str
    scope_type: str
    scope_id: Optional[str]
    granted_by: Optional[str]
    granted_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    
    model_config = {"from_attributes": True}


class RoleResponse(BaseModel):
    """Role details."""
    
    id: str
    name: str
    description: Optional[str]
    permissions: List[str]
    is_system_role: bool
    
    model_config = {"from_attributes": True}


class UserAdminResponse(BaseSchema):
    """Admin-facing user response."""

    id: str
    email: EmailStr
    display_name: str
    avatar_url: Optional[str] = None
    is_active: bool
    created_at: datetime


class UserAdminListItemResponse(UserAdminResponse):
    """Admin-facing user list item with role summary."""

    role_names: List[str] = []
    role_assignment_ids: dict[str, str] = {}


class UserWithRolesResponse(UserAdminResponse):
    """Admin-facing user with role assignments."""

    roles: List[UserRoleResponse] = []


# =============================================================================
# User Endpoints
# =============================================================================

@router.get("", response_model=PaginatedResponse[UserAdminListItemResponse])
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    is_active: Optional[bool] = None,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    List users in the organization.
    
    **Required role:** org_admin or system_admin
    """
    # Build query
    query = select(User).where(User.default_organization_id == tenant.org_id)
    count_query = select(func.count(User.id)).where(
        User.default_organization_id == tenant.org_id
    )
    
    # Apply filters
    if search:
        search_filter = or_(
            User.email.ilike(f"%{search}%"),
            User.display_name.ilike(f"%{search}%"),
        )
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)
    
    if is_active is not None:
        query = query.where(User.is_active == is_active)
        count_query = count_query.where(User.is_active == is_active)
    
    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size).order_by(User.created_at.desc())
    
    result = await db.execute(query)
    users = result.scalars().all()

    user_ids = [str(u.id) for u in users]
    roles_by_user: dict[str, list[str]] = {uid: [] for uid in user_ids}
    assignment_ids_by_user: dict[str, dict[str, str]] = {uid: {} for uid in user_ids}

    if user_ids:
        roles_result = await db.execute(
            select(UserRole.user_id, UserRole.id, Role.name)
            .join(Role, UserRole.role_id == Role.id)
            .where(
                UserRole.user_id.in_(user_ids),
                UserRole.organization_id == tenant.org_id,
                UserRole.is_active == True,
            )
        )
        for user_id, assignment_id, role_name in roles_result.all():
            uid = str(user_id)
            roles_by_user.setdefault(uid, []).append(role_name)
            assignment_ids_by_user.setdefault(uid, {})[role_name] = str(assignment_id)
    
    return PaginatedResponse(
        items=[
            UserAdminListItemResponse(
                id=str(u.id),
                email=u.email,
                display_name=u.display_name,
                avatar_url=u.avatar_url,
                is_active=u.is_active,
                created_at=u.created_at,
                role_names=roles_by_user.get(str(u.id), []),
                role_assignment_ids=assignment_ids_by_user.get(str(u.id), {}),
            )
            for u in users
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )


@router.post("", response_model=UserAdminResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new user.
    
    **Required role:** org_admin or system_admin
    
    Note: For SSO users, use external_id instead of password.
    """
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    user = User(
        email=body.email,
        external_id=body.external_id,
        display_name=body.display_name,
        default_organization_id=body.default_organization_id,
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return UserAdminResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/{user_id}", response_model=UserWithRolesResponse)
async def get_user(
    user_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user details with roles.
    
    **Required role:** org_admin or system_admin
    """
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.default_organization_id == tenant.org_id,
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Get user's roles
    roles_result = await db.execute(
        select(UserRole, Role).join(Role, UserRole.role_id == Role.id).where(
            UserRole.user_id == user_id,
            UserRole.is_active == True,
        )
    )
    user_roles = []
    for user_role, role in roles_result.all():
        user_roles.append(UserRoleResponse(
            id=str(user_role.id),
            user_id=str(user_role.user_id),
            role_id=str(user_role.role_id),
            role_name=role.name,
            scope_type=user_role.scope_type,
            scope_id=str(user_role.scope_id) if user_role.scope_id else None,
            granted_by=str(user_role.granted_by) if user_role.granted_by else None,
            granted_at=user_role.granted_at,
            expires_at=user_role.expires_at,
            is_active=user_role.is_active,
        ))
    
    return UserWithRolesResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        is_active=user.is_active,
        created_at=user.created_at,
        roles=user_roles,
    )


@router.patch("/{user_id}", response_model=UserAdminResponse)
async def update_user(
    user_id: str,
    body: UserUpdate,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a user.
    
    **Required role:** org_admin or system_admin
    """
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.default_organization_id == tenant.org_id,
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Update fields
    if body.display_name is not None:
        user.display_name = body.display_name
    if body.avatar_url is not None:
        user.avatar_url = body.avatar_url
    if body.preferences is not None:
        user.preferences = body.preferences
    if body.is_active is not None:
        user.is_active = body.is_active
    
    await db.commit()
    await db.refresh(user)
    
    return UserAdminResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    user_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Deactivate a user (soft delete).
    
    **Required role:** org_admin or system_admin
    """
    if user_id == tenant.user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate yourself",
        )
    
    result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.default_organization_id == tenant.org_id,
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    user.is_active = False
    await db.commit()


# =============================================================================
# Role Endpoints
# =============================================================================

@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    List available roles.
    
    **Required role:** org_admin or system_admin
    """
    result = await db.execute(
        select(Role).where(
            or_(
                Role.organization_id == tenant.org_id,
                Role.is_system_role == True,
            )
        ).order_by(Role.is_system_role.desc(), Role.name)
    )
    roles = result.scalars().all()
    
    return [RoleResponse.model_validate(r) for r in roles]


@router.post("/{user_id}/roles", response_model=UserRoleResponse, status_code=status.HTTP_201_CREATED)
async def assign_role(
    user_id: str,
    body: UserRoleAssign,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Assign a role to a user.
    
    **Required role:** org_admin or system_admin
    """
    # Check user exists
    user_result = await db.execute(
        select(User).where(
            User.id == user_id,
            User.default_organization_id == tenant.org_id,
        )
    )
    if not user_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Check role exists
    role_result = await db.execute(select(Role).where(Role.id == body.role_id))
    role = role_result.scalar_one_or_none()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found",
        )
    
    # Check for existing assignment
    existing = await db.execute(
        select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == body.role_id,
            UserRole.scope_type == body.scope_type,
            UserRole.scope_id == body.scope_id if body.scope_id else UserRole.scope_id.is_(None),
            UserRole.is_active == True,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Role already assigned to user with same scope",
        )
    
    user_role = UserRole(
        user_id=user_id,
        role_id=body.role_id,
        organization_id=tenant.org_id,
        scope_type=body.scope_type,
        scope_id=body.scope_id,
        granted_by=tenant.user_id,
        granted_at=datetime.now(timezone.utc),
        expires_at=body.expires_at,
        granted_reason=body.granted_reason,
    )
    
    db.add(user_role)
    await db.commit()
    await db.refresh(user_role)
    
    return UserRoleResponse(
        id=str(user_role.id),
        user_id=str(user_role.user_id),
        role_id=str(user_role.role_id),
        role_name=role.name,
        scope_type=user_role.scope_type,
        scope_id=str(user_role.scope_id) if user_role.scope_id else None,
        granted_by=str(user_role.granted_by) if user_role.granted_by else None,
        granted_at=user_role.granted_at,
        expires_at=user_role.expires_at,
        is_active=user_role.is_active,
    )


@router.delete("/{user_id}/roles/{role_assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_role(
    user_id: str,
    role_assignment_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke a role assignment.
    
    **Required role:** org_admin or system_admin
    """
    result = await db.execute(
        select(UserRole).where(
            UserRole.id == role_assignment_id,
            UserRole.user_id == user_id,
            UserRole.organization_id == tenant.org_id,
        )
    )
    user_role = result.scalar_one_or_none()
    
    if not user_role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role assignment not found",
        )
    
    user_role.is_active = False
    user_role.revoked_at = datetime.now(timezone.utc)
    user_role.revoked_by = tenant.user_id
    
    await db.commit()
