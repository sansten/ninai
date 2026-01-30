"""Cognitive Loop endpoints (sessions, iterations, evaluation reports)."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.cognitive_session import CognitiveSession
from app.models.cognitive_iteration import CognitiveIteration
from app.models.tool_call_log import ToolCallLog
from app.models.goal import Goal
from app.services.cognitive_loop.evaluation_report_service import EvaluationReportService
from app.services.agent_scheduler_service import AgentSchedulerService
from app.schemas.cognitive import (
    CognitiveSessionCreateRequest,
    CognitiveSessionResponse,
    CognitiveIterationResponse,
    ToolCallLogResponse,
    EvaluationReportResponse,
)
from app.tasks.cognitive_loop import cognitive_loop_task


router = APIRouter()


def _require_session_access(*, tenant: TenantContext, sess: CognitiveSession) -> None:
    if getattr(sess, "organization_id", None) != tenant.org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if getattr(sess, "user_id", None) == tenant.user_id:
        return
    if tenant.is_org_admin:
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


@router.post("/sessions", response_model=CognitiveSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_cognitive_session(
    body: CognitiveSessionCreateRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    goal_id: str | None = None
    if getattr(body, "goal_id", None):
        # Validate goal exists and is visible under RLS for this user/org.
        res = await db.execute(select(Goal).where(Goal.id == body.goal_id))
        g = res.scalar_one_or_none()
        if g is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Goal not found")
        goal_id = str(getattr(g, "id"))

    row = CognitiveSession(
        organization_id=tenant.org_id,
        user_id=tenant.user_id,
        agent_id=body.agent_id,
        status="running",
        goal_id=goal_id,
        goal=body.goal,
        context_snapshot=body.context_snapshot or {},
        trace_id=x_trace_id,
    )
    db.add(row)
    await db.flush()
    await db.commit()

    now = datetime.now(timezone.utc)

    return CognitiveSessionResponse(
        id=row.id,
        organization_id=row.organization_id,
        user_id=row.user_id,
        agent_id=row.agent_id,
        status=row.status,
        goal=row.goal,
        goal_id=getattr(row, "goal_id", None),
        context_snapshot=row.context_snapshot or {},
        created_at=row.created_at or now,
        updated_at=row.updated_at or now,
        trace_id=row.trace_id,
    )


@router.post("/sessions/{session_id}/run", status_code=status.HTTP_202_ACCEPTED)
async def run_cognitive_session(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
    sess = res.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _require_session_access(tenant=tenant, sess=sess)

    scheduler = AgentSchedulerService(db)
    proc = await scheduler.enqueue(
        organization_id=tenant.org_id,
        agent_name=str(getattr(sess, "agent_id", "") or "cognitive_loop"),
        priority=0,
        session_id=session_id,
        trace_id=str(getattr(sess, "trace_id", "") or None),
        scopes={"scheduler.enqueue"},
    )

    cognitive_loop_task.delay(
        org_id=tenant.org_id,
        session_id=session_id,
        initiator_user_id=tenant.user_id,
        process_id=str(proc.id),
        roles=tenant.roles_string,
        clearance_level=tenant.clearance_level,
        justification="cognitive_loop_api",
    )

    return {"queued": True, "session_id": session_id, "process_id": str(proc.id)}


@router.get("/sessions/{session_id}", response_model=CognitiveSessionResponse)
async def get_cognitive_session(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    res = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
    sess = res.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _require_session_access(tenant=tenant, sess=sess)

    return CognitiveSessionResponse(
        id=sess.id,
        organization_id=sess.organization_id,
        user_id=sess.user_id,
        agent_id=sess.agent_id,
        status=sess.status,
        goal=sess.goal,
        goal_id=getattr(sess, "goal_id", None),
        context_snapshot=sess.context_snapshot or {},
        created_at=sess.created_at,
        updated_at=sess.updated_at,
        trace_id=sess.trace_id,
    )


@router.get("/sessions/{session_id}/iterations", response_model=list[CognitiveIterationResponse])
async def list_cognitive_iterations(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    sres = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
    sess = sres.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _require_session_access(tenant=tenant, sess=sess)

    res = await db.execute(
        select(CognitiveIteration)
        .where(CognitiveIteration.session_id == session_id)
        .order_by(CognitiveIteration.iteration_num.asc())
    )
    rows = res.scalars().all()

    return [
        CognitiveIterationResponse(
            id=r.id,
            session_id=r.session_id,
            iteration_num=r.iteration_num,
            plan_json=r.plan_json or {},
            execution_json=r.execution_json or {},
            critique_json=r.critique_json or {},
            evaluation=r.evaluation,
            started_at=r.started_at,
            finished_at=r.finished_at,
            metrics=r.metrics or {},
        )
        for r in rows
    ]


@router.get("/sessions/{session_id}/tool-calls", response_model=list[ToolCallLogResponse])
async def list_tool_calls(
    session_id: str,
    status_filter: str | None = None,
    tool_name: str | None = None,
    iteration_id: str | None = None,
    limit: int = 200,
    offset: int = 0,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    sres = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
    sess = sres.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _require_session_access(tenant=tenant, sess=sess)

    stmt = select(ToolCallLog).where(ToolCallLog.session_id == session_id)
    if status_filter:
        stmt = stmt.where(ToolCallLog.status == status_filter)
    if tool_name:
        stmt = stmt.where(ToolCallLog.tool_name == tool_name)
    if iteration_id:
        stmt = stmt.where(ToolCallLog.iteration_id == iteration_id)

    stmt = stmt.order_by(ToolCallLog.started_at.asc()).limit(min(max(int(limit or 200), 1), 500)).offset(max(int(offset or 0), 0))

    res = await db.execute(stmt)
    rows = res.scalars().all()

    return [
        ToolCallLogResponse(
            id=r.id,
            session_id=r.session_id,
            iteration_id=r.iteration_id,
            tool_name=r.tool_name,
            tool_input=r.tool_input or {},
            tool_output_summary=r.tool_output_summary or {},
            status=r.status,
            denial_reason=r.denial_reason,
            started_at=r.started_at,
            finished_at=r.finished_at,
        )
        for r in rows
    ]


@router.get("/sessions/{session_id}/report", response_model=EvaluationReportResponse)
async def get_evaluation_report(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    sres = await db.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
    sess = sres.scalar_one_or_none()
    if sess is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    _require_session_access(tenant=tenant, sess=sess)

    svc = EvaluationReportService(db)
    existing = await svc.get_latest_for_session(session_id=session_id)
    if existing is None and str(getattr(sess, "status", "")) in {"succeeded", "failed", "aborted"}:
        existing = await svc.generate_for_session(session_id=session_id)
        await db.commit()

    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation report not found")

    return EvaluationReportResponse(
        id=existing.id,
        session_id=existing.session_id,
        report=existing.report or {},
        final_decision=existing.final_decision,
        created_at=existing.created_at,
    )
