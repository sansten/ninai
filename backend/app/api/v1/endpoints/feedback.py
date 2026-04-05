"""Active Learning Feedback Loop API (Feature 18)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun
from app.models.audit import AuditEvent
from app.models.contradiction import Contradiction
from app.models.memory import MemoryMetadata
from app.models.meta_agent import MetaConflictRegistry
from app.services.audit_service import AuditService
from app.services.memory_feedback_service import MemoryFeedbackService
from app.tasks.memory_pipeline import enqueue_feedback_learning

router = APIRouter()


class MemoryFeedbackRequest(BaseModel):
    relevant: bool
    comment: str | None = Field(default=None, max_length=2000)
    used_in: str | None = Field(default=None, max_length=500)
    target_agent: str | None = Field(default=None, max_length=255)
    trace_id: str | None = Field(default=None, max_length=255)


class DecisionFeedbackRequest(BaseModel):
    correct: bool
    actual_outcome: str | None = Field(default=None, max_length=2000)
    comment: str | None = Field(default=None, max_length=2000)
    target_agent: str | None = Field(default=None, max_length=255)


class ConflictFeedbackRequest(BaseModel):
    false_positive: bool
    comment: str | None = Field(default=None, max_length=2000)
    target_agent: str | None = Field(default=None, max_length=255)


class AnomalyFeedbackRequest(BaseModel):
    genuine: bool
    comment: str | None = Field(default=None, max_length=2000)
    target_agent: str | None = Field(default=None, max_length=255)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes"}:
            return True
        if lower in {"false", "0", "no"}:
            return False
    return default


@router.post("/memory/{memory_id}", status_code=status.HTTP_201_CREATED)
async def post_memory_feedback(
    request: Request,
    memory_id: str,
    body: MemoryFeedbackRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Record thumbs up/down style feedback for a memory."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    memory_res = await db.execute(
        select(MemoryMetadata).where(
            MemoryMetadata.id == memory_id,
            MemoryMetadata.organization_id == tenant.org_id,
            MemoryMetadata.is_active.is_(True),
        )
    )
    memory = memory_res.scalar_one_or_none()
    if memory is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

    payload = {
        "relevant": bool(body.relevant),
        "value": 1 if body.relevant else -1,
        "comment": body.comment,
        "used_in": body.used_in,
        "trace_id": body.trace_id or getattr(request.state, "request_id", None),
    }

    row = await MemoryFeedbackService(db, user_id=tenant.user_id, org_id=tenant.org_id).create_feedback(
        memory_id=memory_id,
        feedback_type="relevance",
        payload=payload,
        target_agent=body.target_agent,
    )

    await AuditService(db).log_event(
        event_type="feedback.memory",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        resource_type="memory",
        resource_id=memory_id,
        success=True,
        details={
            "relevant": bool(body.relevant),
            "comment": body.comment,
            "used_in": body.used_in,
            "target_agent": body.target_agent,
        },
    )

    await db.commit()

    enqueue_feedback_learning(
        org_id=tenant.org_id,
        memory_id=memory_id,
        initiator_user_id=tenant.user_id,
        trace_id=payload.get("trace_id"),
        storage="long_term",
    )

    return {
        "feedback_id": str(row.id),
        "feedback_type": "memory",
        "memory_id": memory_id,
        "relevant": bool(body.relevant),
        "target_agent": body.target_agent,
        "created_at": getattr(row, "created_at", _utcnow()),
    }


@router.post("/decision/{decision_id}", status_code=status.HTTP_201_CREATED)
async def post_decision_feedback(
    decision_id: str,
    body: DecisionFeedbackRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Record whether a decision outcome was correct."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    run_res = await db.execute(
        select(AgentRun).where(
            AgentRun.id == decision_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    run = run_res.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision not found")

    event = await AuditService(db).log_event(
        event_type="feedback.decision",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        resource_type="agent_run",
        resource_id=decision_id,
        success=True,
        details={
            "correct": bool(body.correct),
            "actual_outcome": body.actual_outcome,
            "comment": body.comment,
            "target_agent": body.target_agent or run.agent_name,
        },
    )
    await db.commit()

    return {
        "feedback_id": str(event.id),
        "feedback_type": "decision",
        "decision_id": decision_id,
        "correct": bool(body.correct),
        "target_agent": body.target_agent or run.agent_name,
        "created_at": event.timestamp,
    }


@router.post("/conflict/{conflict_id}", status_code=status.HTTP_201_CREATED)
async def post_conflict_feedback(
    conflict_id: str,
    body: ConflictFeedbackRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Record whether a conflict was a false positive."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    contradiction_res = await db.execute(
        select(Contradiction).where(
            Contradiction.id == conflict_id,
            Contradiction.organization_id == tenant.org_id,
        )
    )
    contradiction = contradiction_res.scalar_one_or_none()

    meta_res = None
    if contradiction is None:
        meta_query = await db.execute(
            select(MetaConflictRegistry).where(
                MetaConflictRegistry.id == conflict_id,
                MetaConflictRegistry.organization_id == tenant.org_id,
            )
        )
        meta_res = meta_query.scalar_one_or_none()

    if contradiction is None and meta_res is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conflict not found")

    conflict_type = "fact_contradiction" if contradiction is not None else str(meta_res.conflict_type)
    event = await AuditService(db).log_event(
        event_type="feedback.conflict",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        resource_type="conflict",
        resource_id=conflict_id,
        success=True,
        details={
            "false_positive": bool(body.false_positive),
            "comment": body.comment,
            "target_agent": body.target_agent,
            "conflict_type": conflict_type,
        },
    )
    await db.commit()

    return {
        "feedback_id": str(event.id),
        "feedback_type": "conflict",
        "conflict_id": conflict_id,
        "false_positive": bool(body.false_positive),
        "conflict_type": conflict_type,
        "created_at": event.timestamp,
    }


@router.post("/anomaly/{anomaly_id}", status_code=status.HTTP_201_CREATED)
async def post_anomaly_feedback(
    anomaly_id: str,
    body: AnomalyFeedbackRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Record whether an anomaly alert was genuine."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    run_res = await db.execute(
        select(AgentRun).where(
            AgentRun.id == anomaly_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    run = run_res.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")

    event = await AuditService(db).log_event(
        event_type="feedback.anomaly",
        actor_id=tenant.user_id,
        organization_id=tenant.org_id,
        resource_type="anomaly",
        resource_id=anomaly_id,
        success=True,
        details={
            "genuine": bool(body.genuine),
            "comment": body.comment,
            "target_agent": body.target_agent or run.agent_name,
        },
    )
    await db.commit()

    return {
        "feedback_id": str(event.id),
        "feedback_type": "anomaly",
        "anomaly_id": anomaly_id,
        "genuine": bool(body.genuine),
        "target_agent": body.target_agent or run.agent_name,
        "created_at": event.timestamp,
    }


@router.get("/stats")
async def get_feedback_stats(
    since: datetime | None = Query(default=None),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Aggregate feedback metrics by type and agent/domain."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    since_ts = since
    if since_ts is None:
        since_ts = _utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    elif since_ts.tzinfo is None:
        since_ts = since_ts.replace(tzinfo=timezone.utc)

    event_res = await db.execute(
        select(AuditEvent)
        .where(
            AuditEvent.organization_id == tenant.org_id,
            AuditEvent.timestamp >= since_ts,
            AuditEvent.event_type.in_(
                [
                    "feedback.memory",
                    "feedback.decision",
                    "feedback.conflict",
                    "feedback.anomaly",
                ]
            ),
        )
    )
    events = list(event_res.scalars().all())

    totals = {
        "memory": 0,
        "decision": 0,
        "conflict": 0,
        "anomaly": 0,
    }
    positive = {
        "memory": 0,
        "decision": 0,
        "conflict": 0,
        "anomaly": 0,
    }
    per_agent: dict[str, dict[str, int]] = {}

    for event in events:
        event_name = str(event.event_type)
        kind = event_name.split(".", 1)[1] if "." in event_name else event_name
        if kind not in totals:
            continue

        totals[kind] += 1
        details = dict(event.details or {})

        if kind == "memory":
            if _as_bool(details.get("relevant"), False):
                positive[kind] += 1
        elif kind == "decision":
            if _as_bool(details.get("correct"), False):
                positive[kind] += 1
        elif kind == "conflict":
            if not _as_bool(details.get("false_positive"), False):
                positive[kind] += 1
        elif kind == "anomaly":
            if _as_bool(details.get("genuine"), False):
                positive[kind] += 1

        agent_name = str(details.get("target_agent") or "unspecified")
        bucket = per_agent.setdefault(agent_name, {"total": 0, "positive": 0})
        bucket["total"] += 1
        if kind in {"memory", "decision", "anomaly"} and positive[kind] > 0:
            pass

    # Re-compute per-agent positive counts directly for correctness.
    for event in events:
        details = dict(event.details or {})
        agent_name = str(details.get("target_agent") or "unspecified")
        bucket = per_agent.setdefault(agent_name, {"total": 0, "positive": 0})

        kind = str(event.event_type).split(".", 1)[1] if "." in str(event.event_type) else str(event.event_type)
        is_positive = False
        if kind == "memory":
            is_positive = _as_bool(details.get("relevant"), False)
        elif kind == "decision":
            is_positive = _as_bool(details.get("correct"), False)
        elif kind == "conflict":
            is_positive = not _as_bool(details.get("false_positive"), False)
        elif kind == "anomaly":
            is_positive = _as_bool(details.get("genuine"), False)
        if is_positive:
            bucket["positive"] += 1

    total_feedback = sum(totals.values())

    return {
        "since": since_ts,
        "total_feedback": total_feedback,
        "by_type": totals,
        "positive_by_type": positive,
        "positive_ratio_by_type": {
            key: (positive[key] / totals[key] if totals[key] else 0.0)
            for key in totals
        },
        "per_agent": per_agent,
    }