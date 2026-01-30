"""
Authentication Endpoints
========================

Login, logout, token refresh, and user profile endpoints.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.oidc import (
    OidcError,
    extract_email_and_name,
    parse_group_to_role_mapping,
    verify_id_token,
)
from app.core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_password_hash,
)
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    get_optional_tenant_context,
)
from app.models.user import User, UserRole, Role
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    AuthMethodsResponse,
    OidcExchangeRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
    OrgSwitchRequest,
)
from app.services.audit_service import AuditService
from app.services.app_settings_service import get_effective_auth_config


router = APIRouter()


def _auth_mode_from(cfg: dict) -> str:
    mode = (cfg.get("auth_mode") or "password").strip().lower()
    if mode not in {"password", "oidc", "both"}:
        return "password"
    return mode


def _password_enabled(cfg: dict) -> bool:
    return _auth_mode_from(cfg) in {"password", "both"}


def _oidc_enabled(cfg: dict) -> bool:
    return _auth_mode_from(cfg) in {"oidc", "both"}


async def _resolve_default_org_id(db: AsyncSession, cfg: dict) -> Optional[str]:
    if cfg.get("oidc_default_org_id"):
        return cfg.get("oidc_default_org_id")
    if cfg.get("oidc_default_org_slug"):
        from app.models.organization import Organization

        result = await db.execute(
            select(Organization.id).where(Organization.slug == cfg.get("oidc_default_org_slug"))
        )
        return result.scalar_one_or_none()
    return None


async def _ensure_user_role(
    *,
    db: AsyncSession,
    user_id: str,
    org_id: str,
    role_name: str,
) -> None:
    role_result = await db.execute(
        select(Role).where(Role.name == role_name)
    )
    role = role_result.scalar_one_or_none()
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configured role '{role_name}' does not exist",
        )

    existing_result = await db.execute(
        select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.organization_id == org_id,
            UserRole.role_id == role.id,
        )
    )
    existing = existing_result.scalar_one_or_none()
    if existing:
        return

    db.add(
        UserRole(
            id=str(uuid4()),
            user_id=user_id,
            role_id=role.id,
            organization_id=org_id,
            scope_type="organization",
        )
    )


async def _build_login_response(
    *,
    db: AsyncSession,
    user: User,
    org_id: Optional[str],
    roles: list[str],
) -> LoginResponse:
    # Fetch the organization details
    organization = None
    if org_id:
        from app.models.organization import Organization

        org_result = await db.execute(
            select(Organization).where(Organization.id == org_id)
        )
        organization = org_result.scalar_one_or_none()

    access_token = create_access_token(
        user_id=user.id,
        org_id=org_id,
        roles=roles,
    )
    refresh_token = create_refresh_token(
        user_id=user.id,
        org_id=org_id,
    )

    from app.schemas.organization import OrganizationResponse

    org_response = None
    if organization:
        org_response = OrganizationResponse(
            id=organization.id,
            name=organization.name,
            slug=organization.slug,
            description=organization.description,
            settings=organization.settings,
            is_active=organization.is_active,
            parent_org_id=organization.parent_org_id,
            created_at=organization.created_at,
            updated_at=organization.updated_at,
        )

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=1800,
        user=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            avatar_url=user.avatar_url,
            is_active=user.is_active,
            clearance_level=user.clearance_level,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            organization_id=org_id if org_id else None,
            roles=roles,
        ),
        organization=org_response,
    )


@router.get("/methods", response_model=AuthMethodsResponse)
async def auth_methods(db: AsyncSession = Depends(get_db)):
    """Return which auth methods are enabled (public endpoint)."""
    effective, _ = await get_effective_auth_config(db)
    return AuthMethodsResponse(
        password_enabled=_password_enabled(effective),
        oidc_enabled=_oidc_enabled(effective),
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user and return tokens.
    
    Returns access and refresh tokens on successful authentication.
    """
    effective, _ = await get_effective_auth_config(db)
    if not _password_enabled(effective):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password login is disabled",
        )
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == body.email)
    )
    user = result.scalar_one_or_none()
    
    # Get request metadata for audit
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    
    audit_service = AuditService(db)
    
    if not user:
        await audit_service.log_auth_failure(
            email=body.email,
            reason="User not found",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    if not verify_password(body.password, user.hashed_password):
        await audit_service.log_auth_failure(
            email=body.email,
            reason="Invalid password",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    
    if not user.is_active:
        await audit_service.log_auth_failure(
            email=body.email,
            reason="Account inactive",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is not active",
        )
    
    # Get user's organizations and roles
    user_roles_query = await db.execute(
        select(UserRole).where(UserRole.user_id == user.id).order_by(UserRole.created_at)
    )
    user_roles_list = user_roles_query.scalars().all()
    
    if not user_roles_list:
        # User has no roles, cannot log in
        await audit_service.log_auth_failure(
            email=body.email,
            reason="No roles assigned",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User has no assigned roles",
        )
    
    # Use the first organization the user belongs to
    org_id = user_roles_list[0].organization_id
    
    # Get role names for this organization
    role_ids = [ur.role_id for ur in user_roles_list if ur.organization_id == org_id]
    if role_ids:
        roles_query = await db.execute(
            select(Role.name).where(Role.id.in_(role_ids))
        )
        roles = roles_query.scalars().all()
    else:
        roles = []
    
    # Update last login (use timezone-naive for TIMESTAMP WITHOUT TIME ZONE)
    user.last_login_at = datetime.utcnow()
    
    # Audit log
    await audit_service.log_auth_success(
        user_id=user.id,
        organization_id=org_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    await db.commit()

    return await _build_login_response(db=db, user=user, org_id=org_id, roles=roles)


@router.post("/oidc/exchange", response_model=LoginResponse)
async def oidc_exchange(
    request: Request,
    body: OidcExchangeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Exchange an OIDC ID token for Ninai tokens."""
    effective, _ = await get_effective_auth_config(db)
    if not _oidc_enabled(effective):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OIDC login is disabled",
        )

    issuer = effective.get("oidc_issuer")
    client_id = effective.get("oidc_client_id")
    audience = effective.get("oidc_audience")
    if not issuer or not client_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OIDC is enabled but not configured (OIDC_ISSUER / OIDC_CLIENT_ID)",
        )

    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("User-Agent")
    audit_service = AuditService(db)

    try:
        claims = await verify_id_token(
            id_token=body.id_token,
            issuer=issuer,
            client_id=client_id,
            audience=audience,
        )
    except OidcError as e:
        await audit_service.log_auth_failure(
            email="(oidc)",
            reason=f"OIDC verification failed: {e}",
            ip_address=ip_address,
            user_agent=user_agent,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid SSO token",
        ) from e

    email, full_name = extract_email_and_name(claims)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="SSO token missing email claim",
        )

    allowed_domains = effective.get("oidc_allowed_email_domains")
    if allowed_domains:
        domain = email.split("@")[-1].lower() if "@" in email else ""
        allowed = {d.lower() for d in allowed_domains}
        if domain not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email domain not allowed",
            )

    # Upsert local user
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            id=str(uuid4()),
            email=email,
            hashed_password=get_password_hash(uuid4().hex),
            full_name=full_name or email,
            is_active=True,
        )
        db.add(user)
        await db.flush()

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active",
        )

    # Determine org context
    org_id = await _resolve_default_org_id(db, effective)

    # Determine roles from group mapping (optional), otherwise default role
    role_name = (effective.get("oidc_default_role") or "member").strip()
    groups_claim = (effective.get("oidc_groups_claim") or "groups").strip()
    group_to_role = parse_group_to_role_mapping(effective.get("oidc_group_to_role_json"))

    roles: list[str] = []
    token_groups = claims.get(groups_claim)
    if isinstance(token_groups, list) and group_to_role:
        for g in token_groups:
            if isinstance(g, str) and g in group_to_role:
                roles.append(group_to_role[g])

    if not roles:
        roles = [role_name]

    if not org_id:
        # Without an org we cannot set tenant context; require corporate to configure defaults.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OIDC user has no organization mapping (configure OIDC_DEFAULT_ORG_ID or OIDC_DEFAULT_ORG_SLUG)",
        )

    # Ensure user has at least one role in the default org
    await _ensure_user_role(db=db, user_id=user.id, org_id=org_id, role_name=roles[0])

    # Refresh roles list from DB for the org
    user_roles_query = await db.execute(
        select(UserRole).where(UserRole.user_id == user.id, UserRole.organization_id == org_id)
    )
    user_roles_list = user_roles_query.scalars().all()
    role_ids = [ur.role_id for ur in user_roles_list]
    roles_query = await db.execute(select(Role.name).where(Role.id.in_(role_ids)))
    roles = roles_query.scalars().all()

    user.last_login_at = datetime.utcnow()

    await audit_service.log_auth_success(
        user_id=user.id,
        organization_id=org_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    await db.commit()

    return await _build_login_response(db=db, user=user, org_id=org_id, roles=roles)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Refresh access token using refresh token.
    """
    token_data = verify_token(body.refresh_token, token_type="refresh")
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    # Verify user still exists and is active
    result = await db.execute(
        select(User).where(User.id == token_data.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Create new access token
    access_token = create_access_token(
        user_id=token_data.user_id,
        org_id=token_data.org_id,
        roles=token_data.roles,
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=1800,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current authenticated user's profile.
    """
    result = await db.execute(
        select(User).where(User.id == tenant.user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        is_active=user.is_active,
        clearance_level=user.clearance_level,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        organization_id=tenant.org_id if tenant.org_id else None,
        roles=tenant.roles,
    )


@router.post("/switch-org", response_model=TokenResponse)
async def switch_organization(
    body: OrgSwitchRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Switch to a different organization context.
    
    Returns a new access token scoped to the requested organization.
    User must have a role in the target organization.
    """
    # TODO: Verify user has role in target organization
    # For now, just create a new token with the new org
    
    # Get user's roles in the new org
    roles = []  # Placeholder - implement role lookup for new org
    
    access_token = create_access_token(
        user_id=tenant.user_id,
        org_id=body.organization_id,
        roles=roles,
    )
    
    return TokenResponse(
        access_token=access_token,
        expires_in=1800,
    )


@router.post("/logout")
async def logout(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Log out current user.
    
    In a full implementation, this would invalidate the refresh token.
    """
    audit_service = AuditService(db)
    
    await audit_service.log_event(
        event_type="auth.logout",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        success=True,
    )
    
    await db.commit()
    
    return {"message": "Successfully logged out"}
