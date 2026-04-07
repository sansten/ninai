"""Data residency settings endpoints."""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.org_data_residency import OrgDataResidency, VALID_REGIONS

router = APIRouter(prefix="/admin/data-residency", tags=["data-residency"])

REGION_TO_GCP = {
    "us": "us-central1",
    "eu": "europe-west1",
    "apac": "asia-southeast1",
    "ca": "northamerica-northeast1",
}


class ResidencyRequest(BaseModel):
    region: str
    backup_region: str | None = None


@router.get("")
async def get_residency(
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(select(OrgDataResidency).where(OrgDataResidency.organization_id == tenant.org_id))
    row = res.scalar_one_or_none()
    if not row:
        return {"region": "us", "gdpr_required": False, "gcp_region": "us-central1"}

    return {
        "region": row.region,
        "gdpr_required": row.gdpr_required,
        "gcp_region": row.gcp_region,
        "backup_region": row.backup_region,
    }


@router.put("")
async def set_residency(
    body: ResidencyRequest,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    if body.region not in VALID_REGIONS:
        raise HTTPException(status_code=400, detail=f"Region must be one of: {sorted(VALID_REGIONS)}")

    res = await db.execute(select(OrgDataResidency).where(OrgDataResidency.organization_id == tenant.org_id))
    row = res.scalar_one_or_none()
    if not row:
        row = OrgDataResidency(id=str(uuid4()), organization_id=tenant.org_id)
        db.add(row)

    row.region = body.region
    row.gcp_region = REGION_TO_GCP[body.region]
    row.gdpr_required = body.region == "eu"
    row.backup_region = body.backup_region
    row.declared_at = datetime.now(timezone.utc).isoformat()
    await db.commit()

    return {"region": row.region, "gcp_region": row.gcp_region, "gdpr_required": row.gdpr_required}
