"""Autonomous Action Approval Gate endpoints (Feature 14)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import and_, desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, require_org_admin
from app.models.action_execution_record import ActionExecutionRecord
from app.services.webhook_service import WebhookService

router = APIRouter()


class ActionDecisionRequest(BaseModel):
    comment: str | None = Field(default=None, max_length=2000)
    reason: str | None = Field(default=None, max_length=2000)


class ActionRecordResponse(BaseModel):
    id: str
    session_id: str | None
    action_type: str
    connector_id: str | None
    target_url: str | None
    status: str
    policy_decision: str
    attempt_count: int
    http_status_code: int | None
    error_message: str | None
    confidence_at_dispatch: float | None
    payload_summary: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class ActionDecisionResponse(BaseModel):
    action: ActionRecordResponse
    decision: str


def _to_response(row: ActionExecutionRecord) -> ActionRecordResponse:
    return ActionRecordResponse(
        id=str(row.id),
        session_id=str(row.session_id) if row.session_id else None,
        action_type=row.action_type,
        connector_id=row.connector_id,
        target_url=row.target_url,
        status=row.status,
        policy_decision=row.policy_decision,
        attempt_count=row.attempt_count,
        http_status_code=row.http_status_code,
        error_message=row.error_message,
        confidence_at_dispatch=row.confidence_at_dispatch,
        payload_summary=dict(row.payload_summary or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
        completed_at=row.completed_at,
    )


async def _load_action(db: AsyncSession, *, org_id: str, action_id: str) -> ActionExecutionRecord:
    res = await db.execute(
        select(ActionExecutionRecord).where(
            ActionExecutionRecord.id == action_id,
            ActionExecutionRecord.organization_id == org_id,
        )
    )
    row = res.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Action not found")
    return row


@router.get("/pending", response_model=list[ActionRecordResponse])
async def list_pending_actions(
    limit: int = Query(default=100, ge=1, le=500),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """List actions awaiting approval."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = (
        select(ActionExecutionRecord)
        .where(
            ActionExecutionRecord.organization_id == tenant.org_id,
            or_(
                ActionExecutionRecord.policy_decision == "human_review_required",
                ActionExecutionRecord.status.in_(["pending", "pending_review"]),
            ),
        )
        .order_by(desc(ActionExecutionRecord.created_at))
        .limit(limit)
    )
    res = await db.execute(stmt)
    rows = list(res.scalars().all())
    return [_to_response(r) for r in rows]


@router.get("/history", response_model=list[ActionRecordResponse])
async def list_action_history(
    limit: int = Query(default=100, ge=1, le=500),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """List action approval/execution history."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)

    stmt = (
        select(ActionExecutionRecord)
        .where(
            ActionExecutionRecord.organization_id == tenant.org_id,
            and_(
                ActionExecutionRecord.policy_decision != "human_review_required",
                ActionExecutionRecord.status.not_in(["pending", "pending_review"]),
            ),
        )
        .order_by(desc(ActionExecutionRecord.created_at))
        .limit(limit)
    )
    res = await db.execute(stmt)
    rows = list(res.scalars().all())
    return [_to_response(r) for r in rows]


@router.get("/{action_id}", response_model=ActionRecordResponse)
async def get_action_details(
    action_id: str,
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Get details for one action."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    row = await _load_action(db, org_id=tenant.org_id, action_id=action_id)
    return _to_response(row)


@router.post("/{action_id}/approve", response_model=ActionDecisionResponse)
async def approve_action(
    action_id: str,
    body: ActionDecisionRequest = Body(default_factory=ActionDecisionRequest),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Approve a pending action."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    row = await _load_action(db, org_id=tenant.org_id, action_id=action_id)

    if row.policy_decision == "denied":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Action already rejected")

    row.policy_decision = "auto_approved"
    if row.status in {"pending_review", "pending"}:
        row.status = "pending"
    summary = dict(row.payload_summary or {})
    approvals = list(summary.get("approvals") or [])
    approvals.append(
        {
            "actor_id": tenant.user_id,
            "decision": "approved",
            "comment": body.comment,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    summary["approvals"] = approvals
    row.payload_summary = summary

    await db.commit()

    try:
        await WebhookService(db).emit_event(
            organization_id=tenant.org_id,
            event_type="action.approved",
            payload={
                "action_id": str(row.id),
                "session_id": str(row.session_id) if row.session_id else None,
                "status": row.status,
                "policy_decision": row.policy_decision,
                "approved_by": tenant.user_id,
                "comment": body.comment,
            },
        )
        await db.commit()
    except Exception:
        pass

    return ActionDecisionResponse(action=_to_response(row), decision="approved")


@router.post("/{action_id}/reject", response_model=ActionDecisionResponse)
async def reject_action(
    action_id: str,
    body: ActionDecisionRequest = Body(default_factory=ActionDecisionRequest),
    tenant: TenantContext = Depends(require_org_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Reject a pending action."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    row = await _load_action(db, org_id=tenant.org_id, action_id=action_id)

    row.policy_decision = "denied"
    row.status = "denied"
    row.error_message = body.reason or body.comment or "Rejected by human reviewer"
    row.completed_at = datetime.now(timezone.utc)

    summary = dict(row.payload_summary or {})
    approvals = list(summary.get("approvals") or [])
    approvals.append(
        {
            "actor_id": tenant.user_id,
            "decision": "rejected",
            "reason": body.reason,
            "comment": body.comment,
            "at": datetime.now(timezone.utc).isoformat(),
        }
    )
    summary["approvals"] = approvals
    row.payload_summary = summary

    await db.commit()

    try:
        await WebhookService(db).emit_event(
            organization_id=tenant.org_id,
            event_type="action.completed",
            payload={
                "action_id": str(row.id),
                "session_id": str(row.session_id) if row.session_id else None,
                "status": row.status,
                "policy_decision": row.policy_decision,
                "rejected_by": tenant.user_id,
                "reason": body.reason,
                "comment": body.comment,
            },
        )
        await db.commit()
    except Exception:
        pass

    return ActionDecisionResponse(action=_to_response(row), decision="rejected")
