"""Admin feature flag endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.feature_flag_service import FeatureFlagService

router = APIRouter(prefix="/admin", tags=["feature-flags"])


class FeatureFlagUpsertRequest(BaseModel):
    enabled: bool
    rollout_pct: int = Field(default=100, ge=0, le=100)


class FeatureFlagBatchItem(BaseModel):
    flag_name: str
    enabled: bool
    rollout_pct: int = Field(default=100, ge=0, le=100)


class FeatureFlagBatchUpsertRequest(BaseModel):
    flags: list[FeatureFlagBatchItem]


@router.get("/feature-flags")
async def list_feature_flags(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    svc = FeatureFlagService(db, tenant.org_id)
    return {"flags": await svc.list_flags()}


@router.put("/feature-flags")
async def upsert_feature_flags(
    body: FeatureFlagBatchUpsertRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    svc = FeatureFlagService(db, tenant.org_id)
    for item in body.flags:
        await svc.set_flag(flag_name=item.flag_name, enabled=item.enabled, rollout_pct=item.rollout_pct)
    await db.commit()
    return {"flags": await svc.list_flags()}


@router.put("/feature-flags/{flag_name}")
async def upsert_feature_flag(
    flag_name: str,
    body: FeatureFlagUpsertRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    svc = FeatureFlagService(db, tenant.org_id)
    row = await svc.set_flag(flag_name=flag_name, enabled=body.enabled, rollout_pct=body.rollout_pct)
    await db.commit()
    return {"flag_name": row.flag_name, "enabled": row.enabled, "rollout_pct": row.rollout_pct}
