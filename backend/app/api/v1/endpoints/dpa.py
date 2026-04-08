"""DPA acceptance endpoints required for EU-region organizations."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.dpa_acceptance import DpaAcceptance

CURRENT_DPA_VERSION = "2026-04-01"

router = APIRouter(prefix="/admin/dpa", tags=["dpa"])


@router.get("/status")
async def get_dpa_status(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    res = await db.execute(
        select(DpaAcceptance).where(
            DpaAcceptance.organization_id == tenant.org_id,
            DpaAcceptance.dpa_version == CURRENT_DPA_VERSION,
        )
    )
    row = res.scalar_one_or_none()

    return {
        "current_version": CURRENT_DPA_VERSION,
        "accepted": bool(row and row.accepted),
        "accepted_at": row.accepted_at if row else None,
        "accepted_by": row.accepted_by_user_id if row else None,
    }


@router.post("/accept")
async def accept_dpa(
    request: Request,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    res = await db.execute(
        select(DpaAcceptance).where(
            DpaAcceptance.organization_id == tenant.org_id,
            DpaAcceptance.dpa_version == CURRENT_DPA_VERSION,
        )
    )
    row = res.scalar_one_or_none()

    if not row:
        row = DpaAcceptance(
            id=str(uuid4()),
            organization_id=tenant.org_id,
            dpa_version=CURRENT_DPA_VERSION,
        )
        db.add(row)

    row.accepted = True
    row.accepted_by_user_id = tenant.user_id
    row.accepted_at = datetime.now(timezone.utc).isoformat()
    row.ip_address = request.client.host if request.client else None

    await db.commit()
    return {
        "accepted": True,
        "version": CURRENT_DPA_VERSION,
        "accepted_at": row.accepted_at,
    }
