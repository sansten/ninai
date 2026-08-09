"""Human Review Queue API — Phase 32.

Surfaces review_recommended memories from UncertaintyReportingAgent into a
structured review queue. Resolved verdicts are written to MemoryFeedback so
FeedbackIntegrationAgent picks them up on the next enrichment pass.

Endpoints
---------
GET  /review/queue
    List memories queued for human review ordered by priority.

POST /review/claim
    Claim a queued memory; returns full reviewer context.

POST /review/resolve
    Submit a review verdict. Writes to MemoryFeedback for feedback integration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun
from app.models.memory_feedback import MemoryFeedback
from app.models.base import generate_uuid

router = APIRouter()

_REVIEW_QUEUE_AGENT = "HumanReviewQueueAgent"
_UNCERTAINTY_AGENT = "UncertaintyReportingAgent"
_AUTONOMOUS_ACTION_AGENT = "autonomous_action_agent"

# Lower number = higher urgency
_PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class QueueItem(BaseModel):
    memory_id: str
    priority: str
    review_reasons: list[str] = Field(default_factory=list)
    review_context: dict[str, Any] = Field(default_factory=dict)
    queue_entry_id: Optional[str] = None
    queued_at: datetime


class QueueResponse(BaseModel):
    items: list[QueueItem]
    total: int
    retrieved_at: datetime


class ClaimRequest(BaseModel):
    memory_id: str


class ClaimResponse(BaseModel):
    memory_id: str
    priority: str
    review_reasons: list[str]
    review_context: dict[str, Any]
    claimed_at: datetime


class ResolveRequest(BaseModel):
    memory_id: str
    verdict: Optional[str] = Field(
        None,
        description="approve | reject | partial",
    )
    notes: Optional[str] = None
    credibility_override: Optional[float] = Field(None, ge=0.0, le=1.0)
    resolution_confirmed: Optional[bool] = None
    playbook_correct: Optional[bool] = None


class ResolveResponse(BaseModel):
    memory_id: str
    verdict: Optional[str]
    feedback_id: str
    resolved_at: datetime


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


async def _load_review_runs(
    db: AsyncSession,
    org_id: str,
    limit: int = 500,
) -> list[AgentRun]:
    stmt = (
        select(AgentRun)
        .where(
            AgentRun.organization_id == org_id,
            AgentRun.agent_name == _REVIEW_QUEUE_AGENT,
            AgentRun.status == "success",
        )
        .order_by(AgentRun.finished_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _load_uncertainty_runs(
    db: AsyncSession,
    org_id: str,
    limit: int = 500,
) -> list[AgentRun]:
    stmt = (
        select(AgentRun)
        .where(
            AgentRun.organization_id == org_id,
            AgentRun.agent_name == _UNCERTAINTY_AGENT,
            AgentRun.status == "success",
        )
        .order_by(AgentRun.finished_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _load_autonomous_action_runs(
    db: AsyncSession,
    org_id: str,
    limit: int = 500,
) -> list[AgentRun]:
    stmt = (
        select(AgentRun)
        .where(
            AgentRun.organization_id == org_id,
            AgentRun.agent_name == _AUTONOMOUS_ACTION_AGENT,
            AgentRun.status == "success",
        )
        .order_by(AgentRun.finished_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


def _run_to_queue_item(run: AgentRun) -> QueueItem | None:
    """Convert an AgentRun to a QueueItem if the run marks the memory as eligible."""
    if not isinstance(run.outputs, dict):
        return None
    outputs = run.outputs

    if run.agent_name == _REVIEW_QUEUE_AGENT:
        if not outputs.get("queue_eligible"):
            return None
        return QueueItem(
            memory_id=run.memory_id,
            priority=str(outputs.get("priority") or "low"),
            review_reasons=list(outputs.get("review_reasons") or []),
            review_context=dict(outputs.get("review_context") or {}),
            queue_entry_id=outputs.get("queue_entry_id"),
            queued_at=run.finished_at,
        )

    if run.agent_name == _AUTONOMOUS_ACTION_AGENT:
        decision = str(outputs.get("policy_decision") or "").strip().lower()
        action_status = str(outputs.get("action_status") or "").strip().lower()
        if decision not in {"human_review_required", "denied"} and action_status != "pending_review":
            return None

        priority = "urgent" if decision == "denied" else "high"
        review_reason = (
            outputs.get("_runtime_control_reason")
            or outputs.get("_dispatch_error")
            or outputs.get("_decision_reason")
            or f"autonomous action policy decision: {decision or action_status}"
        )
        return QueueItem(
            memory_id=run.memory_id,
            priority=priority,
            review_reasons=[str(review_reason)],
            review_context={
                "source_agent": _AUTONOMOUS_ACTION_AGENT,
                "policy_decision": decision,
                "action_status": action_status,
                "action_type": outputs.get("action_type"),
                "dispatch_confidence": outputs.get("dispatch_confidence"),
            },
            queue_entry_id=run.memory_id,
            queued_at=run.finished_at,
        )

    # Fallback path: UncertaintyReportingAgent
    if not outputs.get("review_recommended"):
        return None
    level = str(outputs.get("uncertainty_level") or "medium").lower()
    priority_map = {"critical": "urgent", "high": "high", "medium": "normal", "low": "low"}
    priority = priority_map.get(level, "normal")
    return QueueItem(
        memory_id=run.memory_id,
        priority=priority,
        review_reasons=["flagged by UncertaintyReportingAgent"],
        review_context={
            "uncertainty_level": outputs.get("uncertainty_level"),
            "uncertainty_score": outputs.get("uncertainty_score"),
            "unresolved_conflicts": outputs.get("unresolved_conflicts") or [],
            "low_confidence_fields": outputs.get("low_confidence_fields") or [],
        },
        queue_entry_id=run.memory_id,
        queued_at=run.finished_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/queue",
    response_model=QueueResponse,
    summary="List memories queued for human review",
)
async def get_review_queue(
    priority: Optional[str] = Query(
        None, description="Filter by priority: low|normal|high|urgent"
    ),
    limit: int = Query(50, ge=1, le=200, description="Max items to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> QueueResponse:
    """Return memories currently in the review queue, ordered by priority."""
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    runs = await _load_review_runs(db, tenant.org_id)
    items = [item for run in runs if (item := _run_to_queue_item(run)) is not None]

    if not items:
        unc_runs = await _load_uncertainty_runs(db, tenant.org_id)
        items = [item for run in unc_runs if (item := _run_to_queue_item(run)) is not None]

    if not items:
        auto_runs = await _load_autonomous_action_runs(db, tenant.org_id)
        items = [item for run in auto_runs if (item := _run_to_queue_item(run)) is not None]

    # Deduplicate by memory_id — keep highest priority entry
    seen: dict[str, QueueItem] = {}
    for item in items:
        existing = seen.get(item.memory_id)
        if existing is None or (
            _PRIORITY_ORDER.get(item.priority, 99) < _PRIORITY_ORDER.get(existing.priority, 99)
        ):
            seen[item.memory_id] = item

    items = list(seen.values())

    if priority:
        items = [i for i in items if i.priority == priority]

    items.sort(
        key=lambda i: (_PRIORITY_ORDER.get(i.priority, 99), -i.queued_at.timestamp())
    )

    total = len(items)
    items = items[offset: offset + limit]

    return QueueResponse(
        items=items,
        total=total,
        retrieved_at=datetime.now(timezone.utc),
    )


@router.post(
    "/claim",
    response_model=ClaimResponse,
    summary="Claim a queued memory for review",
)
async def claim_review_item(
    body: ClaimRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ClaimResponse:
    """Claim a queued memory and return its full reviewer context."""
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    stmt = (
        select(AgentRun)
        .where(
            AgentRun.organization_id == tenant.org_id,
            AgentRun.memory_id == body.memory_id,
            AgentRun.agent_name == _REVIEW_QUEUE_AGENT,
            AgentRun.status == "success",
        )
        .order_by(AgentRun.finished_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    runs = result.scalars().all()
    run = runs[0] if runs else None

    if run is not None:
        outputs = run.outputs if isinstance(run.outputs, dict) else {}
        if not outputs.get("queue_eligible"):
            raise HTTPException(status_code=404, detail="memory is not in review queue")
        priority = str(outputs.get("priority") or "low")
        review_reasons = list(outputs.get("review_reasons") or [])
        review_context = dict(outputs.get("review_context") or {})
    else:
        # Fallback: check UncertaintyReportingAgent
        stmt2 = (
            select(AgentRun)
            .where(
                AgentRun.organization_id == tenant.org_id,
                AgentRun.memory_id == body.memory_id,
                AgentRun.agent_name == _UNCERTAINTY_AGENT,
                AgentRun.status == "success",
            )
            .order_by(AgentRun.finished_at.desc())
            .limit(1)
        )
        result2 = await db.execute(stmt2)
        unc_runs = result2.scalars().all()
        unc_run = unc_runs[0] if unc_runs else None

        if unc_run is not None and (unc_run.outputs or {}).get("review_recommended"):
            unc_outputs = unc_run.outputs if isinstance(unc_run.outputs, dict) else {}
            level = str(unc_outputs.get("uncertainty_level") or "medium").lower()
            priority_map = {"critical": "urgent", "high": "high", "medium": "normal", "low": "low"}
            priority = priority_map.get(level, "normal")
            review_reasons = ["flagged by UncertaintyReportingAgent"]
            review_context = {
                "uncertainty_level": unc_outputs.get("uncertainty_level"),
                "uncertainty_score": unc_outputs.get("uncertainty_score"),
                "unresolved_conflicts": unc_outputs.get("unresolved_conflicts") or [],
                "low_confidence_fields": unc_outputs.get("low_confidence_fields") or [],
            }
        else:
            stmt3 = (
                select(AgentRun)
                .where(
                    AgentRun.organization_id == tenant.org_id,
                    AgentRun.memory_id == body.memory_id,
                    AgentRun.agent_name == _AUTONOMOUS_ACTION_AGENT,
                    AgentRun.status == "success",
                )
                .order_by(AgentRun.finished_at.desc())
                .limit(1)
            )
            result3 = await db.execute(stmt3)
            auto_runs = result3.scalars().all()
            auto_run = auto_runs[0] if auto_runs else None

            auto_outputs = auto_run.outputs if auto_run is not None and isinstance(auto_run.outputs, dict) else {}
            auto_decision = str(auto_outputs.get("policy_decision") or "").lower()
            auto_status = str(auto_outputs.get("action_status") or "").lower()
            if auto_run is None or (auto_decision not in {"human_review_required", "denied"} and auto_status != "pending_review"):
                raise HTTPException(status_code=404, detail="memory is not in review queue")

            priority = "urgent" if auto_decision == "denied" else "high"
            review_reasons = [
                str(
                    auto_outputs.get("_runtime_control_reason")
                    or auto_outputs.get("_dispatch_error")
                    or auto_outputs.get("_decision_reason")
                    or f"autonomous action policy decision: {auto_decision or auto_status}"
                )
            ]
            review_context = {
                "source_agent": _AUTONOMOUS_ACTION_AGENT,
                "policy_decision": auto_decision,
                "action_status": auto_status,
                "action_type": auto_outputs.get("action_type"),
                "dispatch_confidence": auto_outputs.get("dispatch_confidence"),
            }

    return ClaimResponse(
        memory_id=body.memory_id,
        priority=priority,
        review_reasons=review_reasons,
        review_context=review_context,
        claimed_at=datetime.now(timezone.utc),
    )


@router.post(
    "/resolve",
    response_model=ResolveResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a review verdict for a queued memory",
)
async def resolve_review_item(
    body: ResolveRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ResolveResponse:
    """Record a human review verdict; feeds back into FeedbackIntegrationAgent."""
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    valid_verdicts = {"approve", "reject", "partial", None}
    if body.verdict not in valid_verdicts:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"verdict must be one of approve, reject, partial",
        )

    feedback_id = generate_uuid()
    payload: dict[str, Any] = {
        "verdict": body.verdict,
        "source": "human_review_queue",
    }
    if body.notes is not None:
        payload["notes"] = body.notes
    if body.credibility_override is not None:
        payload["credibility_override"] = body.credibility_override
    if body.resolution_confirmed is not None:
        payload["resolution_confirmed"] = body.resolution_confirmed
    if body.playbook_correct is not None:
        payload["playbook_correct"] = body.playbook_correct

    fb = MemoryFeedback(
        id=feedback_id,
        organization_id=tenant.org_id,
        memory_id=body.memory_id,
        actor_id=tenant.user_id,
        feedback_type="review_verdict",
        # See memory_enrichment.py's submit_feedback for why this targets
        # FeedbackLearningAgent (the actual local consumer) rather than the
        # enterprise-only FeedbackIntegrationAgent.
        target_agent="FeedbackLearningAgent",
        payload=payload,
        is_applied=False,
    )
    db.add(fb)
    await db.commit()

    return ResolveResponse(
        memory_id=body.memory_id,
        verdict=body.verdict,
        feedback_id=feedback_id,
        resolved_at=datetime.now(timezone.utc),
    )
