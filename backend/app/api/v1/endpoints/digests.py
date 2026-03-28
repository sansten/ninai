"""Intelligence Digest Endpoints — P46.

Routes:
  GET /digests/latest          — most recent digest for caller's org
  GET /digests/history         — paginated digest history
  POST /digests/trigger        — manually trigger a digest build (admin)
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.digest_service import DigestService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Intelligence Digest"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class DigestReportResponse(BaseModel):
    id: str
    organization_id: str
    run_date: date
    digest_type: str
    memories_total: int
    memories_flagged: int
    anomalies_count: int
    goals_active: int
    goals_completed: int
    top_insights: Optional[list] = None
    quality_trend: Optional[dict] = None
    narrative: Optional[str] = None
    delivered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/latest", response_model=DigestReportResponse)
async def get_latest_digest(
    digest_type: str = Query("daily", pattern="^(daily|weekly)$"),
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> DigestReportResponse:
    """Return the most recent intelligence digest for the caller's organization."""
    svc = DigestService(db)
    report = await svc.get_latest(ctx.organization_id, digest_type)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No digest found. Run the digest task or wait for the next scheduled run.",
        )
    return DigestReportResponse.model_validate(report)


@router.get("/history", response_model=List[DigestReportResponse])
async def list_digest_history(
    digest_type: str = Query("daily", pattern="^(daily|weekly)$"),
    limit: int = Query(30, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> List[DigestReportResponse]:
    """Return paginated digest history for the caller's organization."""
    svc = DigestService(db)
    reports = await svc.list_history(ctx.organization_id, digest_type, limit=limit)
    return [DigestReportResponse.model_validate(r) for r in reports]


@router.post("/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_digest(
    digest_type: str = Query("daily", pattern="^(daily|weekly)$"),
    db: AsyncSession = Depends(get_db),
    ctx: TenantContext = Depends(get_tenant_context),
) -> dict:
    """Manually build and save a digest for the caller's org. Admin only."""
    roles = set((ctx.roles or "").split(","))
    if "admin" not in roles and "org_admin" not in roles and "system_admin" not in roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")

    svc = DigestService(db)
    try:
        report = await svc.build_digest(ctx.organization_id, digest_type)
        saved = await svc.save_report(report)
    except Exception as exc:
        logger.exception("Manual digest trigger failed", extra={"org_id": ctx.organization_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Digest build failed",
        )

    return {
        "report_id": str(saved.id),
        "run_date": saved.run_date.isoformat(),
        "digest_type": saved.digest_type,
        "status": "completed",
    }
