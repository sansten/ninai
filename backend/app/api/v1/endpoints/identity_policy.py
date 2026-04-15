"""
Identity Policy Endpoints
=========================

Admin endpoints for managing the org-level identity attribution policy.
Requires org_admin role.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.org_identity_policy import OrgIdentityPolicy
from app.schemas.admin import OrgIdentityPolicyResponse, OrgIdentityPolicyUpdate

router = APIRouter()


def _require_org_admin(tenant: TenantContext) -> None:
    """Raise 403 if the caller is not an org_admin or system_admin."""
    if not (tenant.has_role("org_admin") or tenant.has_role("system_admin")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="org_admin role required",
        )


@router.get(
    "/admin/identity-policy",
    response_model=OrgIdentityPolicyResponse,
    tags=["Admin - Identity Policy"],
)
async def get_identity_policy(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> OrgIdentityPolicyResponse:
    """Retrieve the current org identity attribution policy."""
    _require_org_admin(tenant)
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    row = await db.scalar(
        select(OrgIdentityPolicy).where(OrgIdentityPolicy.org_id == tenant.org_id)
    )

    if row is None:
        # Return default policy values when no row exists yet
        return OrgIdentityPolicyResponse(
            org_id=tenant.org_id,
            mandate_actor_identity=False,
            allowed_modes=["full", "role_only", "anonymous"],
            enrich_from_directory=True,
            audit_trail_always=True,
        )

    return OrgIdentityPolicyResponse(
        org_id=row.org_id,
        mandate_actor_identity=row.mandate_actor_identity,
        allowed_modes=list(row.allowed_modes or []),
        enrich_from_directory=row.enrich_from_directory,
        audit_trail_always=row.audit_trail_always,
    )


@router.patch(
    "/admin/identity-policy",
    response_model=OrgIdentityPolicyResponse,
    tags=["Admin - Identity Policy"],
)
async def update_identity_policy(
    body: OrgIdentityPolicyUpdate,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> OrgIdentityPolicyResponse:
    """Update the org identity attribution policy. Creates a default row if none exists."""
    _require_org_admin(tenant)
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    row = await db.scalar(
        select(OrgIdentityPolicy).where(OrgIdentityPolicy.org_id == tenant.org_id)
    )

    if row is None:
        row = OrgIdentityPolicy(
            org_id=tenant.org_id,
            mandate_actor_identity=False,
            allowed_modes=["full", "role_only", "anonymous"],
            enrich_from_directory=True,
            audit_trail_always=True,
        )
        db.add(row)

    if body.mandate_actor_identity is not None:
        row.mandate_actor_identity = body.mandate_actor_identity
    if body.allowed_modes is not None:
        row.allowed_modes = list(body.allowed_modes)
    if body.enrich_from_directory is not None:
        row.enrich_from_directory = body.enrich_from_directory
    if body.audit_trail_always is not None:
        row.audit_trail_always = body.audit_trail_always

    row.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(row)

    return OrgIdentityPolicyResponse(
        org_id=row.org_id,
        mandate_actor_identity=row.mandate_actor_identity,
        allowed_modes=list(row.allowed_modes or []),
        enrich_from_directory=row.enrich_from_directory,
        audit_trail_always=row.audit_trail_always,
    )


@router.delete(
    "/admin/identity-cache/{user_id}",
    status_code=status.HTTP_200_OK,
    tags=["Admin - Identity Policy"],
)
async def invalidate_identity_cache(
    user_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
) -> dict:
    """
    Bust the Redis identity cache for a specific user.

    Call this after a termination, role change, or AD record update
    when immediate propagation is required without waiting for the 15-min TTL.
    """
    _require_org_admin(tenant)

    from app.core.redis import RedisClient

    try:
        client = await RedisClient.get_client()
        await client.delete(f"identity:{tenant.org_id}:{user_id}")
    except Exception:
        # Cache invalidation is best-effort
        pass

    return {"invalidated": True, "user_id": user_id}
