"""Explainability & Reasoning Trace API endpoints.

Feature 13 surfaces enterprise-readable reasoning traces for decisions,
memories, conflicts, and anomalies.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import asc, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun
from app.models.agent_run_event import AgentRunEvent
from app.models.audit import AgentDecisionTrail
from app.models.contradiction import Contradiction
from app.models.memory_fact import MemoryFact
from app.models.meta_agent import MetaConflictRegistry

router = APIRouter()


async def _set_ctx(db: AsyncSession, tenant: TenantContext) -> None:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@router.get("/memory/{memory_id}")
async def explain_memory(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Explain why a memory was scored and processed the way it was."""
    await _set_ctx(db, tenant)

    result = await db.execute(
        select(AgentDecisionTrail)
        .where(
            AgentDecisionTrail.organization_id == tenant.org_id,
            AgentDecisionTrail.memory_id == memory_id,
        )
        .order_by(asc(AgentDecisionTrail.timestamp))
    )
    rows = list(result.scalars().all())
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No decision trail found for memory")

    reasoning_steps = [
        {
            "step": idx + 1,
            "agent": row.agent_name,
            "finding": row.decision,
            "confidence": row.confidence,
            "reasoning_snapshot": row.reasoning_snapshot or {},
            "timestamp": row.timestamp,
            "trace_id": row.trace_id,
        }
        for idx, row in enumerate(rows)
    ]

    confidences = [_safe_float(r.confidence, 0.0) for r in rows]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "memory_id": memory_id,
        "average_confidence": round(avg_conf, 4),
        "agents": list(dict.fromkeys([r.agent_name for r in rows if r.agent_name])),
        "reasoning_steps": reasoning_steps,
    }


@router.get("/conflict/{conflict_id}")
async def explain_conflict(
    conflict_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Explain why a conflict was raised."""
    await _set_ctx(db, tenant)

    contradiction_result = await db.execute(
        select(Contradiction).where(
            Contradiction.id == conflict_id,
            Contradiction.organization_id == tenant.org_id,
        )
    )
    contradiction = contradiction_result.scalar_one_or_none()

    if contradiction:
        fact_a_result = await db.execute(select(MemoryFact).where(MemoryFact.id == contradiction.fact_a))
        fact_b_result = await db.execute(select(MemoryFact).where(MemoryFact.id == contradiction.fact_b))
        fact_a = fact_a_result.scalar_one_or_none()
        fact_b = fact_b_result.scalar_one_or_none()

        facts = []
        if fact_a:
            facts.append(
                {
                    "id": fact_a.id,
                    "subject": fact_a.subject,
                    "predicate": fact_a.predicate,
                    "object": fact_a.object,
                    "confidence": fact_a.confidence,
                    "source_memory_id": fact_a.source_memory_id,
                }
            )
        if fact_b:
            facts.append(
                {
                    "id": fact_b.id,
                    "subject": fact_b.subject,
                    "predicate": fact_b.predicate,
                    "object": fact_b.object,
                    "confidence": fact_b.confidence,
                    "source_memory_id": fact_b.source_memory_id,
                }
            )

        return {
            "conflict_id": contradiction.id,
            "conflict_type": "fact_contradiction",
            "severity": str(contradiction.severity),
            "reason": contradiction.reason,
            "created_at": contradiction.created_at,
            "resolved_at": contradiction.resolved_at,
            "facts": facts,
        }

    meta_result = await db.execute(
        select(MetaConflictRegistry).where(
            MetaConflictRegistry.id == conflict_id,
            MetaConflictRegistry.organization_id == tenant.org_id,
        )
    )
    meta_conflict = meta_result.scalar_one_or_none()

    if meta_conflict:
        return {
            "conflict_id": meta_conflict.id,
            "conflict_type": meta_conflict.conflict_type,
            "status": meta_conflict.status,
            "resource_type": meta_conflict.resource_type,
            "resource_id": meta_conflict.resource_id,
            "candidates": meta_conflict.candidates or {},
            "resolution": meta_conflict.resolution or {},
            "resolved_by": meta_conflict.resolved_by,
            "created_at": meta_conflict.created_at,
            "resolved_at": meta_conflict.resolved_at,
        }

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conflict not found")


@router.get("/anomaly/{anomaly_id}")
async def explain_anomaly(
    anomaly_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Explain why an anomaly was raised."""
    await _set_ctx(db, tenant)

    run_result = await db.execute(
        select(AgentRun).where(
            AgentRun.id == anomaly_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    run = run_result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly not found")

    outputs = run.outputs or {}
    has_anomaly_shape = "anomaly_detected" in outputs or "anomaly_score" in outputs
    if run.agent_name != "AnomalyDetectionAgent" and not has_anomaly_shape:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Anomaly trace not found")

    events_result = await db.execute(
        select(AgentRunEvent)
        .where(
            AgentRunEvent.organization_id == tenant.org_id,
            AgentRunEvent.agent_run_id == run.id,
        )
        .order_by(asc(AgentRunEvent.step_index), asc(AgentRunEvent.created_at))
    )
    events = list(events_result.scalars().all())

    reasoning_steps = [
        {
            "step": idx + 1,
            "event_type": event.event_type,
            "finding": event.summary_text or (event.payload or {}).get("finding") or "",
            "payload": event.payload or {},
            "created_at": event.created_at,
        }
        for idx, event in enumerate(events)
    ]

    return {
        "anomaly_id": anomaly_id,
        "memory_id": run.memory_id,
        "agent_name": run.agent_name,
        "anomaly_detected": bool(outputs.get("anomaly_detected", False)),
        "anomaly_type": outputs.get("anomaly_type"),
        "severity": outputs.get("severity"),
        "affected_fields": outputs.get("affected_fields") or [],
        "anomaly_score": _safe_float(outputs.get("anomaly_score"), 0.0),
        "confidence": _safe_float(outputs.get("confidence"), _safe_float(run.confidence, 0.0)),
        "reasoning_steps": reasoning_steps,
        "trace_id": run.trace_id,
    }


@router.get("/{decision_id}")
async def explain_decision(
    decision_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Return full reasoning trace for a decision id (AgentRun id)."""
    await _set_ctx(db, tenant)

    run_result = await db.execute(
        select(AgentRun).where(
            AgentRun.id == decision_id,
            AgentRun.organization_id == tenant.org_id,
        )
    )
    run = run_result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision not found")

    events_result = await db.execute(
        select(AgentRunEvent)
        .where(
            AgentRunEvent.organization_id == tenant.org_id,
            AgentRunEvent.agent_run_id == run.id,
        )
        .order_by(asc(AgentRunEvent.step_index), asc(AgentRunEvent.created_at))
    )
    events = list(events_result.scalars().all())

    reasoning_steps = []
    for idx, event in enumerate(events):
        payload = event.payload or {}
        finding = payload.get("finding") or payload.get("decision") or event.summary_text or event.event_type
        reasoning_steps.append(
            {
                "step": idx + 1,
                "agent": run.agent_name,
                "finding": finding,
                "event_type": event.event_type,
                "payload": payload,
                "created_at": event.created_at,
            }
        )

    if not reasoning_steps:
        reasoning_steps = [
            {
                "step": 1,
                "agent": run.agent_name,
                "finding": (run.outputs or {}).get("decision") or run.status,
                "event_type": "final",
                "payload": run.outputs or {},
                "created_at": run.finished_at,
            }
        ]

    memories_used: list[str] = []
    if run.memory_id:
        memories_used.append(str(run.memory_id))
    for prov in run.provenance or []:
        if not isinstance(prov, dict):
            continue
        for key in ("memory_id", "source_memory_id"):
            mid = prov.get(key)
            if isinstance(mid, str) and mid and mid not in memories_used:
                memories_used.append(mid)

    outputs = run.outputs or {}

    return {
        "decision_id": decision_id,
        "decision": outputs.get("decision") or outputs.get("recommendation") or run.status,
        "confidence": _safe_float(outputs.get("confidence"), _safe_float(run.confidence, 0.0)),
        "reasoning_steps": reasoning_steps,
        "memories_used": memories_used,
        "alternatives_considered": outputs.get("alternatives_considered") or [],
        "audit_trail_id": run.trace_id or run.id,
        "agent_name": run.agent_name,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
    }
