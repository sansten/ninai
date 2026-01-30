"""Meta Agent Supervision & Calibration endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.meta_agent import CalibrationProfile, MetaAgentRun, MetaConflictRegistry
from app.schemas.meta_agent import (
    CalibrationProfileOut,
    CalibrationProfileUpdateIn,
    MetaAgentRunOut,
    MetaConflictOut,
)
from app.services.meta_agent.calibration_service import CalibrationService
from app.tasks.meta_agent import (
    calibration_update_task,
    meta_review_cognitive_session_task,
    meta_review_memory_task,
)


router = APIRouter()


@router.post("/review/memories/{memory_id}", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_meta_review_memory(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    job = meta_review_memory_task.delay(
        org_id=tenant.org_id,
        memory_id=memory_id,
        initiator_user_id=tenant.user_id,
        roles=tenant.roles_string,
        clearance_level=int(tenant.clearance_level or 0),
        trace_id=x_trace_id,
    )
    return {"task_id": job.id, "status": "queued"}


@router.post("/review/cognitive-sessions/{session_id}", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_meta_review_cognitive_session(
    session_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    job = meta_review_cognitive_session_task.delay(
        org_id=tenant.org_id,
        session_id=session_id,
        initiator_user_id=tenant.user_id,
        roles=tenant.roles_string,
        clearance_level=int(tenant.clearance_level or 0),
        trace_id=x_trace_id,
    )
    return {"task_id": job.id, "status": "queued"}


@router.get("/runs", response_model=list[MetaAgentRunOut])
async def list_meta_runs(
    resource_type: str | None = None,
    resource_id: str | None = None,
    limit: int = 50,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = select(MetaAgentRun).where(MetaAgentRun.organization_id == tenant.org_id)
    if resource_type:
        stmt = stmt.where(MetaAgentRun.resource_type == resource_type)
    if resource_id:
        stmt = stmt.where(MetaAgentRun.resource_id == resource_id)
    stmt = stmt.order_by(MetaAgentRun.created_at.desc()).limit(int(limit or 50))

    res = await db.execute(stmt)
    now = datetime.now(timezone.utc)

    out: list[MetaAgentRunOut] = []
    for r in res.scalars().all():
        out.append(
            MetaAgentRunOut(
                id=r.id,
                organization_id=r.organization_id,
                resource_type=r.resource_type,
                resource_id=r.resource_id,
                supervision_type=r.supervision_type,
                status=r.status,
                final_confidence=r.final_confidence,
                risk_score=r.risk_score,
                reasoning_summary=r.reasoning_summary,
                evidence=r.evidence or {},
                created_at=getattr(r, "created_at", None) or now,
            )
        )
    return out


@router.get("/conflicts", response_model=list[MetaConflictOut])
async def list_meta_conflicts(
    status_filter: str | None = None,
    limit: int = 50,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = select(MetaConflictRegistry).where(MetaConflictRegistry.organization_id == tenant.org_id)
    if status_filter:
        stmt = stmt.where(MetaConflictRegistry.status == status_filter)
    stmt = stmt.order_by(MetaConflictRegistry.created_at.desc()).limit(int(limit or 50))

    res = await db.execute(stmt)
    now = datetime.now(timezone.utc)

    out: list[MetaConflictOut] = []
    for c in res.scalars().all():
        out.append(
            MetaConflictOut(
                id=c.id,
                organization_id=c.organization_id,
                resource_type=c.resource_type,
                resource_id=c.resource_id,
                conflict_type=c.conflict_type,
                candidates=c.candidates or {},
                resolution=c.resolution or {},
                resolved_by=c.resolved_by,
                status=c.status,
                resolved_at=c.resolved_at,
                created_at=getattr(c, "created_at", None) or now,
            )
        )
    return out


@router.get("/calibration-profile", response_model=CalibrationProfileOut)
async def get_calibration_profile(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    profile = await CalibrationService().get_profile_for_read(db, org_id=tenant.org_id)
    await db.commit()

    now = datetime.now(timezone.utc)
    return CalibrationProfileOut(
        organization_id=profile.organization_id,
        promotion_threshold=float(profile.promotion_threshold or 0.75),
        conflict_escalation_threshold=float(profile.conflict_escalation_threshold or 0.60),
        drift_threshold=float(profile.drift_threshold or 0.20),
        signal_weights=profile.signal_weights or {},
        learning_rate=float(profile.learning_rate or 0.05),
        updated_at=getattr(profile, "updated_at", None) or now,
    )


@router.put("/calibration-profile", status_code=status.HTTP_202_ACCEPTED)
async def update_calibration_profile(
    body: CalibrationProfileUpdateIn,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    if not tenant.is_org_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    # Enqueue update (async) per spec; do not block on DB writes.
    job = calibration_update_task.delay(
        org_id=tenant.org_id,
        initiator_user_id=tenant.user_id,
        signal_weights=body.signal_weights,
        learning_rate=body.learning_rate,
        promotion_threshold=body.promotion_threshold,
        conflict_escalation_threshold=body.conflict_escalation_threshold,
        drift_threshold=body.drift_threshold,
        roles=tenant.roles_string,
        clearance_level=int(tenant.clearance_level or 0),
    )
    return {"task_id": job.id, "status": "queued"}


@router.get("/metrics")
async def meta_metrics(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    open_conflicts = await db.execute(
        select(func.count()).select_from(MetaConflictRegistry).where(
            MetaConflictRegistry.organization_id == tenant.org_id,
            MetaConflictRegistry.status == "open",
        )
    )
    runs_total = await db.execute(
        select(func.count()).select_from(MetaAgentRun).where(MetaAgentRun.organization_id == tenant.org_id)
    )
    escalations = await db.execute(
        select(func.count()).select_from(MetaAgentRun).where(
            MetaAgentRun.organization_id == tenant.org_id,
            MetaAgentRun.status == "escalated",
        )
    )
    avg_conf = await db.execute(
        select(func.avg(MetaAgentRun.final_confidence)).where(MetaAgentRun.organization_id == tenant.org_id)
    )

    runs_total_n = int(runs_total.scalar_one() or 0)
    open_conflicts_n = int(open_conflicts.scalar_one() or 0)
    escalations_n = int(escalations.scalar_one() or 0)
    avg_conf_n = float(avg_conf.scalar_one() or 0.0)

    conflict_rate = float(open_conflicts_n / runs_total_n) if runs_total_n else 0.0
    escalation_rate = float(escalations_n / runs_total_n) if runs_total_n else 0.0

    return {
        "org_id": tenant.org_id,
        "avg_confidence_by_org": avg_conf_n,
        "conflict_rate": conflict_rate,
        "escalation_rate": escalation_rate,
        "feedback_improvement_rate": 0.0,
    }
