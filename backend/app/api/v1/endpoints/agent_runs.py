"""Agent run endpoints.

Provides API access to persisted agent execution outcomes (AgentRun).
This is the first building block for trajectory/procedural memory:
- list runs for an org (optionally filtered by memory_id / agent_name / trace_id)
- fetch a single run with outputs + provenance
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun
from app.models.agent_run_event import AgentRunEvent
from app.models.base import generate_uuid
from app.schemas.agent_run import AgentRunDetailResponse, AgentRunSummaryResponse
from app.schemas.agent_run_event import AgentRunEventCreateRequest, AgentRunEventResponse


router = APIRouter()


@router.get("/agent-runs", response_model=list[AgentRunSummaryResponse])
async def list_agent_runs(
    memory_id: str | None = Query(default=None, description="Filter by memory_id"),
    agent_name: str | None = Query(default=None, description="Filter by agent_name"),
    trace_id: str | None = Query(default=None, description="Filter by trace_id"),
    status_filter: str | None = Query(default=None, alias="status", description="Filter by status"),
    started_after: datetime | None = Query(default=None, description="started_at >= started_after"),
    started_before: datetime | None = Query(default=None, description="started_at <= started_before"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = select(AgentRun).where(AgentRun.organization_id == tenant.org_id)

    if memory_id:
        stmt = stmt.where(AgentRun.memory_id == memory_id)
    if agent_name:
        stmt = stmt.where(AgentRun.agent_name == agent_name)
    if trace_id:
        stmt = stmt.where(AgentRun.trace_id == trace_id)
    if status_filter:
        stmt = stmt.where(AgentRun.status == status_filter)
    if started_after:
        stmt = stmt.where(AgentRun.started_at >= started_after)
    if started_before:
        stmt = stmt.where(AgentRun.started_at <= started_before)

    stmt = stmt.order_by(AgentRun.started_at.desc()).limit(limit).offset(offset)

    res = await db.execute(stmt)
    runs = res.scalars().all()

    return [
        AgentRunSummaryResponse(
            id=r.id,
            organization_id=r.organization_id,
            memory_id=r.memory_id,
            agent_name=r.agent_name,
            agent_version=r.agent_version,
            status=r.status,
            confidence=r.confidence,
            started_at=r.started_at,
            finished_at=r.finished_at,
            trace_id=r.trace_id,
        )
        for r in runs
    ]


@router.get("/agent-runs/{agent_run_id}", response_model=AgentRunDetailResponse)
async def get_agent_run(
    agent_run_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(
        select(AgentRun).where(
            AgentRun.id == agent_run_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    row = res.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found")

    return AgentRunDetailResponse(
        id=row.id,
        organization_id=row.organization_id,
        memory_id=row.memory_id,
        agent_name=row.agent_name,
        agent_version=row.agent_version,
        inputs_hash=row.inputs_hash,
        status=row.status,
        confidence=row.confidence,
        outputs=row.outputs,
        warnings=row.warnings,
        errors=row.errors,
        started_at=row.started_at,
        finished_at=row.finished_at,
        trace_id=row.trace_id,
        provenance=row.provenance,
    )


@router.get("/agent-runs/{agent_run_id}/events", response_model=list[AgentRunEventResponse])
async def list_agent_run_events(
    agent_run_id: str,
    event_type: str | None = Query(default=None, description="Filter by event_type"),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    # Ensure the run exists in this org (also avoids leaking existence across tenants)
    run_res = await db.execute(
        select(AgentRun).where(
            AgentRun.id == agent_run_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    run = run_res.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found")

    stmt = select(AgentRunEvent).where(
        AgentRunEvent.organization_id == tenant.org_id,
        AgentRunEvent.agent_run_id == agent_run_id,
    )
    if event_type:
        stmt = stmt.where(AgentRunEvent.event_type == event_type)

    stmt = stmt.order_by(AgentRunEvent.step_index.asc(), AgentRunEvent.created_at.asc()).limit(limit).offset(offset)
    res = await db.execute(stmt)
    events = res.scalars().all()

    return [
        AgentRunEventResponse(
            id=e.id,
            organization_id=e.organization_id,
            agent_run_id=e.agent_run_id,
            memory_id=e.memory_id,
            event_type=e.event_type,
            step_index=e.step_index,
            payload=e.payload,
            summary_text=getattr(e, "summary_text", "") or "",
            created_at=e.created_at,
            trace_id=e.trace_id,
        )
        for e in events
    ]


@router.post(
    "/agent-runs/{agent_run_id}/events",
    response_model=AgentRunEventResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_agent_run_event(
    agent_run_id: str,
    body: AgentRunEventCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    run_res = await db.execute(
        select(AgentRun).where(
            AgentRun.id == agent_run_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    run = run_res.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent run not found")

    event = AgentRunEvent(
        organization_id=tenant.org_id,
        agent_run_id=agent_run_id,
        memory_id=run.memory_id,
        event_type=body.event_type,
        step_index=body.step_index,
        payload=body.payload,
        summary_text=body.summary_text or "",
        created_at=body.created_at or datetime.now(timezone.utc),
        trace_id=body.trace_id,
    )
    # Ensure id is present even before flush (helps API response + AsyncMock tests).
    if not getattr(event, "id", None):
        event.id = generate_uuid()

    db.add(event)
    await db.commit()

    return AgentRunEventResponse(
        id=event.id,
        organization_id=event.organization_id,
        agent_run_id=event.agent_run_id,
        memory_id=event.memory_id,
        event_type=event.event_type,
        step_index=event.step_index,
        payload=event.payload,
        summary_text=getattr(event, "summary_text", "") or "",
        created_at=event.created_at,
        trace_id=event.trace_id,
    )


@router.get("/agent-run-events/search", response_model=list[AgentRunEventResponse])
async def search_agent_run_events(
    q: str = Query(..., min_length=1, max_length=5000, description="Search query"),
    agent_run_id: str | None = Query(default=None, description="Filter by agent_run_id"),
    memory_id: str | None = Query(default=None, description="Filter by memory_id"),
    event_type: str | None = Query(default=None, description="Filter by event_type"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Search trajectory events by summary text.

    Uses Postgres FTS (simple + plainto_tsquery) for robustness.
    """

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    tsv = func.to_tsvector("simple", func.coalesce(AgentRunEvent.summary_text, ""))
    tsq = func.plainto_tsquery("simple", q)
    rank = func.ts_rank_cd(tsv, tsq)

    stmt = select(AgentRunEvent).where(
        AgentRunEvent.organization_id == tenant.org_id,
        tsv.op("@@")(tsq),
    )
    if agent_run_id:
        stmt = stmt.where(AgentRunEvent.agent_run_id == agent_run_id)
    if memory_id:
        stmt = stmt.where(AgentRunEvent.memory_id == memory_id)
    if event_type:
        stmt = stmt.where(AgentRunEvent.event_type == event_type)

    stmt = stmt.order_by(rank.desc(), AgentRunEvent.created_at.desc()).limit(limit).offset(offset)
    res = await db.execute(stmt)
    events = res.scalars().all()

    return [
        AgentRunEventResponse(
            id=e.id,
            organization_id=e.organization_id,
            agent_run_id=e.agent_run_id,
            memory_id=e.memory_id,
            event_type=e.event_type,
            step_index=e.step_index,
            payload=e.payload,
            summary_text=getattr(e, "summary_text", "") or "",
            created_at=e.created_at,
            trace_id=e.trace_id,
        )
        for e in events
    ]
