"""PR-2 Memory Consolidation (Sleep Cycle) endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.consolidation_pr2 import (
    ConsolidationSessionResponse,
    ConsolidationSessionsResponse,
    ConsolidationStartRequest,
    MemoryArcResponse,
    PinMemoryResponse,
)
from app.services.memory_consolidation_service import MemoryConsolidationService


router = APIRouter(tags=["consolidation-pr2"])


@router.post("/v1/consolidation/start", response_model=ConsolidationSessionResponse)
async def start_consolidation_session(
    request: ConsolidationStartRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ConsolidationSessionResponse:
    """Trigger a full PR-2 memory consolidation cycle."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = MemoryConsolidationService(db, str(tenant.user_id), str(tenant.org_id))
    session = await svc.run_full_consolidation_cycle(session_type=request.session_type)
    return ConsolidationSessionResponse(
        id=str(session.id),
        session_type=session.session_type,
        started_at=session.started_at,
        completed_at=session.completed_at,
        duration_seconds=session.duration_seconds,
        status=session.status,
        operations=dict(session.operations or {}),
        memory_quality_before=session.memory_quality_before,
        memory_quality_after=session.memory_quality_after,
    )


@router.get("/v1/consolidation/sessions", response_model=ConsolidationSessionsResponse)
async def list_consolidation_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ConsolidationSessionsResponse:
    """List recent consolidation sessions."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = MemoryConsolidationService(db, str(tenant.user_id), str(tenant.org_id))
    sessions = await svc.list_sessions(limit=limit)

    items = [
        ConsolidationSessionResponse(
            id=str(s.id),
            session_type=s.session_type,
            started_at=s.started_at,
            completed_at=s.completed_at,
            duration_seconds=s.duration_seconds,
            status=s.status,
            operations=dict(s.operations or {}),
            memory_quality_before=s.memory_quality_before,
            memory_quality_after=s.memory_quality_after,
        )
        for s in sessions
    ]
    return ConsolidationSessionsResponse(items=items, total=len(items))


@router.get("/v1/consolidation/{session_id}/report", response_model=ConsolidationSessionResponse)
async def get_consolidation_report(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ConsolidationSessionResponse:
    """Get detailed report for one consolidation session."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = MemoryConsolidationService(db, str(tenant.user_id), str(tenant.org_id))
    session = await svc.get_session_report(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Consolidation session not found")

    return ConsolidationSessionResponse(
        id=str(session.id),
        session_type=session.session_type,
        started_at=session.started_at,
        completed_at=session.completed_at,
        duration_seconds=session.duration_seconds,
        status=session.status,
        operations=dict(session.operations or {}),
        memory_quality_before=session.memory_quality_before,
        memory_quality_after=session.memory_quality_after,
    )


@router.get("/v1/memory/{memory_id}/arc", response_model=MemoryArcResponse)
async def get_memory_arc(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MemoryArcResponse:
    """Return trajectory analysis for a single memory."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = MemoryConsolidationService(db, str(tenant.user_id), str(tenant.org_id))
    arc = await svc.get_memory_arc(memory_id)
    if arc is None:
        raise HTTPException(status_code=404, detail="Memory arc not found")

    return MemoryArcResponse(
        memory_id=str(arc.memory_id),
        measurements=list(arc.measurements or []),
        trend=arc.trend,
        trajectory_type=arc.trajectory_type,
        prediction_next_access=arc.prediction_next_access,
        last_computed_at=arc.last_computed_at,
    )


@router.post("/v1/consolidation/pin/{memory_id}", response_model=PinMemoryResponse)
async def pin_memory(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> PinMemoryResponse:
    """Mark memory as pinned (do not prune during forgetting curve)."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = MemoryConsolidationService(db, str(tenant.user_id), str(tenant.org_id))
    ok = await svc.pin_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")

    return PinMemoryResponse(memory_id=memory_id, pinned=True, message="Memory pinned")


@router.post("/v1/consolidation/unpin/{memory_id}", response_model=PinMemoryResponse)
async def unpin_memory(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> PinMemoryResponse:
    """Remove pinned flag for memory pruning eligibility."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    svc = MemoryConsolidationService(db, str(tenant.user_id), str(tenant.org_id))
    ok = await svc.unpin_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")

    return PinMemoryResponse(memory_id=memory_id, pinned=False, message="Memory unpinned")
