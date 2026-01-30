"""Simulation reports API endpoints.

These endpoints provide read-only access to persisted SimulationAgent reports.
Access is controlled via RBAC permissions and Postgres RLS.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.simulation_report import SimulationReport
from app.schemas.simulation import SimulationReportResponse
from app.services.permission_checker import PermissionChecker


router = APIRouter()


async def _require_permission(*, db: AsyncSession, tenant: TenantContext, permission: str) -> None:
    checker = PermissionChecker(db)
    decision = await checker.check_permission(tenant.user_id, tenant.org_id, permission)
    if not decision.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason)


@router.get("", response_model=list[SimulationReportResponse])
async def list_simulation_reports(
    session_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        await _require_permission(db=db, tenant=tenant, permission="simulation:read:reports")

        stmt = (
            select(SimulationReport)
            .where(SimulationReport.organization_id == tenant.org_id)
            .order_by(SimulationReport.created_at.desc())
        )
        if session_id:
            stmt = stmt.where(SimulationReport.session_id == session_id)

        stmt = stmt.limit(min(max(int(limit or 100), 1), 500)).offset(max(int(offset or 0), 0))

        res = await db.execute(stmt)
        rows = res.scalars().all()

        return [
            SimulationReportResponse(
                id=str(r.id),
                organization_id=str(r.organization_id),
                session_id=str(r.session_id) if r.session_id else None,
                memory_id=str(r.memory_id) if r.memory_id else None,
                report=r.report or {},
                created_at=r.created_at,
            )
            for r in rows
        ]
