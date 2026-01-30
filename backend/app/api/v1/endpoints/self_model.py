"""SelfModel API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID, uuid4

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.self_model import SelfModelBundleResponse, SelfModelPlannerSummary, SelfModelProfileResponse
from app.schemas.self_model_sampling import (
    ToolOutcomeSampleIn,
    ToolOutcomeSampleOut,
    ToolReliabilityResponse,
)
from app.services.permission_checker import PermissionChecker
from app.services.self_model_service import SelfModelService
from app.models.self_model import SelfModelEvent
from app.tasks.self_model import self_model_recompute_task


router = APIRouter()


def _normalize_uuid(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError):
        return None


async def _require_permission(*, db: AsyncSession, tenant: TenantContext, permission: str) -> None:
    checker = PermissionChecker(db)
    decision = await checker.check_permission(tenant.user_id, tenant.org_id, permission)
    if not decision.allowed:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=decision.reason)


@router.get("/bundle", response_model=SelfModelBundleResponse)
async def get_self_model_bundle(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        await _require_permission(db=db, tenant=tenant, permission="selfmodel:read:org")

        svc = SelfModelService(db)
        prof = await svc.get_profile(org_id=tenant.org_id)
        summary = await svc.get_planner_summary(org_id=tenant.org_id)

        return SelfModelBundleResponse(
            profile=SelfModelProfileResponse(
                organization_id=prof.organization_id,
                domain_confidence=prof.domain_confidence or {},
                tool_reliability=prof.tool_reliability or {},
                agent_accuracy=prof.agent_accuracy or {},
                last_updated=prof.last_updated,
            ),
            planner_summary=SelfModelPlannerSummary(
                unreliable_tools=summary.unreliable_tools,
                low_confidence_domains=summary.low_confidence_domains,
                recommended_evidence_multiplier=summary.recommended_evidence_multiplier,
            ),
        )


@router.post("/recompute", status_code=status.HTTP_202_ACCEPTED)
async def recompute_self_model(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        await _require_permission(db=db, tenant=tenant, permission="selfmodel:manage:org")

    # enqueue outside transaction; always returns quickly.
    self_model_recompute_task.delay(org_id=tenant.org_id)
    return {"queued": True}


@router.post("/samples/tool-outcome", response_model=ToolOutcomeSampleOut, status_code=status.HTTP_201_CREATED)
async def submit_tool_outcome_sample(
    body: ToolOutcomeSampleIn,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        await _require_permission(db=db, tenant=tenant, permission="selfmodel:manage:org")

        event_type = "tool_success" if bool(body.success) else "tool_failure"

        # The DB column is a UUID FK to cognitive_sessions.id/memory_metadata.id.
        # For notebook/SDK convenience, tolerate non-UUID values by storing them
        # in payload and leaving the FK NULL.
        normalized_session_id = _normalize_uuid(body.session_id)
        normalized_memory_id = _normalize_uuid(body.memory_id)

        extra: dict = dict(body.extra or {})
        if body.session_id and normalized_session_id is None:
            extra.setdefault("session_id_raw", body.session_id)
        if body.memory_id and normalized_memory_id is None:
            extra.setdefault("memory_id_raw", body.memory_id)

        row = SelfModelEvent(
            id=str(uuid4()),
            organization_id=tenant.org_id,
            event_type=event_type,
            tool_name=body.tool_name,
            agent_name=None,
            session_id=normalized_session_id,
            memory_id=normalized_memory_id,
            payload={
                "source": "api_sample",
                "success": bool(body.success),
                "duration_ms": body.duration_ms,
                "notes": body.notes,
                "extra": extra,
            },
        )
        db.add(row)
        await db.flush()

    # Keep behavior consistent with /recompute: enqueue outside transaction.
    self_model_recompute_task.delay(org_id=tenant.org_id)

    return ToolOutcomeSampleOut(
        id=str(getattr(row, "id")),
        organization_id=tenant.org_id,
        event_type=event_type,
        tool_name=body.tool_name,
        created_at=row.created_at,
    )


@router.get("/reliability/tools/{tool_name}", response_model=ToolReliabilityResponse)
async def get_tool_reliability(
    tool_name: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    async with db.begin():
        await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
        await _require_permission(db=db, tenant=tenant, permission="selfmodel:read:org")

        svc = SelfModelService(db)
        prof = await svc.get_profile(org_id=tenant.org_id)

        stats = (prof.tool_reliability or {}).get(tool_name)
        if not isinstance(stats, dict):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool not found in SelfModel profile")

        return ToolReliabilityResponse(tool_name=tool_name, stats=stats)
