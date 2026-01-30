"""Agent Process inspection endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_process import AgentProcess
from app.schemas.agent_process import AgentProcessListResponse, AgentProcessResponse

router = APIRouter()


@router.get("/processes", response_model=AgentProcessListResponse)
async def list_agent_processes(
    status_filter: Optional[str] = Query(None, description="Filter by status (queued|running|blocked|succeeded|failed)"),
    agent_name_filter: Optional[str] = Query(None, description="Filter by agent_name (partial match)"),
    session_id_filter: Optional[str] = Query(None, description="Filter by session_id"),
    limit: int = Query(100, ge=1, le=500, description="Limit results"),
    offset: int = Query(0, ge=0, description="Offset results"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """List agent processes with optional filtering."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = select(AgentProcess).where(AgentProcess.organization_id == tenant.org_id)

    if status_filter:
        stmt = stmt.where(AgentProcess.status == status_filter)

    if agent_name_filter:
        stmt = stmt.where(AgentProcess.agent_name.ilike(f"%{agent_name_filter}%"))

    if session_id_filter:
        stmt = stmt.where(AgentProcess.session_id == session_id_filter)

    count_stmt = select(func.count()).select_from(AgentProcess).where(AgentProcess.organization_id == tenant.org_id)
    if status_filter:
        count_stmt = count_stmt.where(AgentProcess.status == status_filter)
    if agent_name_filter:
        count_stmt = count_stmt.where(AgentProcess.agent_name.ilike(f"%{agent_name_filter}%"))
    if session_id_filter:
        count_stmt = count_stmt.where(AgentProcess.session_id == session_id_filter)

    total_res = await db.execute(count_stmt)
    total = int(total_res.scalar_one() or 0)

    stmt = stmt.order_by(AgentProcess.created_at.desc()).limit(limit).offset(offset)

    res = await db.execute(stmt)
    processes = res.scalars().all()

    status_summary: dict[str, int] = {}
    summary_stmt = (
        select(AgentProcess.status, func.count().label("cnt"))
        .where(AgentProcess.organization_id == tenant.org_id)
        .group_by(AgentProcess.status)
    )
    if status_filter:
        summary_stmt = summary_stmt.where(AgentProcess.status == status_filter)
    if agent_name_filter:
        summary_stmt = summary_stmt.where(AgentProcess.agent_name.ilike(f"%{agent_name_filter}%"))
    if session_id_filter:
        summary_stmt = summary_stmt.where(AgentProcess.session_id == session_id_filter)

    summary_res = await db.execute(summary_stmt)
    for status_val, cnt in summary_res.all():
        status_summary[status_val] = int(cnt)

    return AgentProcessListResponse(
        total=total,
        items=[AgentProcessResponse.model_validate(p) for p in processes],
        status_summary=status_summary,
    )


@router.get("/processes/{process_id}", response_model=AgentProcessResponse)
async def get_agent_process(
    process_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Inspect a single agent process."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(
        select(AgentProcess).where(
            AgentProcess.id == process_id,
            AgentProcess.organization_id == tenant.org_id,
        )
    )
    proc = res.scalar_one_or_none()

    if proc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Process not found")

    return AgentProcessResponse.model_validate(proc)
