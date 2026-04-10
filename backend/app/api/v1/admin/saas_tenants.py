"""
Super-admin SaaS tenant management.  All routes require system_admin role.

GET    /admin/saas/tenants                      list all orgs with subscription state
GET    /admin/saas/tenants/{org_id}             single org detail
PATCH  /admin/saas/tenants/{org_id}             update plan / status / version pin
DELETE /admin/saas/tenants/{org_id}             soft-delete (status -> deleted)
GET    /admin/saas/migrations                   list orgs pinned to an old API version
POST   /admin/saas/migrations/{org_id}/notify   send deprecation notice email
POST   /admin/saas/migrations/{org_id}/force    clear pinned version (force upgrade)
POST   /admin/saas/tenants/{org_id}/impersonate issue 15-min support token
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.middleware.tenant_context import TenantContext, require_system_admin
from app.models.org_subscription import OrgSubscription
from app.models.organization import Organization
from app.models.user import User
from app.services.email_service import email_service

router = APIRouter(prefix="/admin/saas", tags=["Super-Admin SaaS"])


class TenantPatch(BaseModel):
    status: Optional[str] = None               # active | suspended | deleted
    plan: Optional[str] = None                 # trial | starter | pro | enterprise
    seat_limit: Optional[int] = None
    pinned_api_version: Optional[str] = None   # set to "" to clear the pin


@router.get("/tenants")
async def list_tenants(
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(Organization, OrgSubscription)
            .outerjoin(OrgSubscription, OrgSubscription.organization_id == Organization.id)
            .order_by(Organization.created_at.desc())
        )
    ).all()

    return [
        {
            "id": org.id,
            "name": org.name,
            "slug": org.slug,
            "status": org.status,
            "pinned_api_version": org.pinned_api_version,
            "signup_ref": org.signup_ref,
            "created_at": org.created_at,
            "subscription": {
                "plan": sub.plan if sub else None,
                "status": sub.status if sub else None,
                "trial_ends_at": sub.trial_ends_at if sub else None,
                "seat_count": sub.seat_count if sub else 0,
                "seat_limit": sub.seat_limit if sub else 0,
                "stripe_customer_id": sub.stripe_customer_id if sub else None,
            },
        }
        for org, sub in rows
    ]


@router.get("/tenants/{org_id}")
async def get_tenant(
    org_id: str,
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    sub = (
        await db.execute(
            select(OrgSubscription).where(OrgSubscription.organization_id == org_id)
        )
    ).scalar_one_or_none()
    return {"org": org, "subscription": sub}


@router.patch("/tenants/{org_id}")
async def patch_tenant(
    org_id: str,
    body: TenantPatch,
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    if body.status is not None:
        org.status = body.status
    if body.pinned_api_version is not None:
        # Empty string clears the pin; non-empty sets it
        org.pinned_api_version = body.pinned_api_version or None

    if body.plan is not None or body.seat_limit is not None:
        sub = (
            await db.execute(
                select(OrgSubscription).where(OrgSubscription.organization_id == org_id)
            )
        ).scalar_one_or_none()
        if sub:
            if body.plan is not None:
                sub.plan = body.plan
            if body.seat_limit is not None:
                sub.seat_limit = body.seat_limit

    await db.commit()
    return {"ok": True}


@router.delete("/tenants/{org_id}")
async def delete_tenant(
    org_id: str,
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    org.status = "deleted"
    org.is_active = False
    await db.commit()
    return {"ok": True}


@router.get("/migrations")
async def list_migration_status(
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    """List all orgs still pinned to an old API version."""
    result = await db.execute(
        select(Organization).where(Organization.pinned_api_version.isnot(None))
    )
    orgs = result.scalars().all()
    return [
        {
            "id": o.id,
            "slug": o.slug,
            "name": o.name,
            "pinned_api_version": o.pinned_api_version,
            "status": o.status,
            "created_at": o.created_at,
        }
        for o in orgs
    ]


@router.post("/migrations/{org_id}/notify")
async def notify_migration(
    org_id: str,
    deadline: str,  # ISO date e.g. "2026-12-31"
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Send version deprecation notice to the org's admin user."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    admin = (
        await db.execute(select(User).where(User.role == "org_admin").limit(1))
    ).scalar_one_or_none()

    if admin:
        await email_service.send_version_deprecation_notice(
            to=admin.email,
            version=org.pinned_api_version or "v1",
            deadline=date.fromisoformat(deadline),
        )
    return {"ok": True, "notified": admin.email if admin else None}


@router.post("/migrations/{org_id}/force")
async def force_migration(
    org_id: str,
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Clear the version pin, forcing the org onto the current API version."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")
    org.pinned_api_version = None
    await db.commit()
    return {"ok": True}


@router.post("/tenants/{org_id}/impersonate")
async def impersonate_tenant(
    org_id: str,
    tenant: TenantContext = Depends(require_system_admin()),
    db: AsyncSession = Depends(get_db),
):
    """
    Issue a 15-minute access token scoped to the target org for support use.
    The token carries the impersonating super-admin's user ID but the target org's
    org_id, so all audit log entries are attributed correctly.
    """
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(404, "Organization not found")

    # Use the support engineer's own user ID (tenant.user_id) but target org
    token = create_access_token(
        user_id=tenant.user_id,
        org_id=org_id,
        roles=["org_admin"],
        expires_delta=timedelta(minutes=15),
    )
    return {
        "access_token": token,
        "expires_in": 900,
        "target_org": org.slug,
        "warning": "Impersonation token — do not share, expires in 15 minutes",
    }
