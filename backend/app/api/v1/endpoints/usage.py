"""Admin usage endpoints for org-level usage dashboards."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.usage_service import UsageService

router = APIRouter(prefix="/admin/usage", tags=["usage"])


@router.get("/summary")
async def usage_summary(
    days: int = Query(30, ge=1, le=365),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    svc = UsageService(db, tenant.org_id)
    summary = await svc.get_summary(days=days)
    return {"days": days, "summary": summary}


@router.get("/daily")
async def usage_daily(
    metric: str = Query(..., min_length=1),
    days: int = Query(30, ge=1, le=365),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    svc = UsageService(db, tenant.org_id)
    data = await svc.get_daily(metric=metric, days=days)
    return {"metric": metric, "days": days, "points": data}
