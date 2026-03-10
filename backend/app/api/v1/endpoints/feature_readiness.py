"""Feature readiness API — Phase 28.

Lets the UI know which enrichment agents are data-ready for a tenant so
features that would produce noise on thin datasets can be greyed out.

Endpoints
---------
GET  /features/readiness
    Returns per-agent readiness state, effective params, and active flag.
    Triggers a fresh evaluation if the tenant has no persisted state yet.

POST /features/readiness/evaluate
    Force-triggers a synchronous evaluation and returns updated results.
    Useful after bulk data imports or during onboarding.

PATCH /features/readiness/{agent_name}
    Admin override: force-enable/disable an agent or adjust its params.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.feature_readiness_service import (
    AGENT_SPECS,
    FeatureReadinessService,
)

router = APIRouter()
_service = FeatureReadinessService()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AgentReadinessItem(BaseModel):
    agent_name: str
    description: str
    enabled: bool
    is_ready: bool
    active: bool = Field(
        description="True when both enabled=True and is_ready=True — the UI should show the feature as fully available",
    )
    readiness_state: dict = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)
    thresholds: dict = Field(default_factory=dict)
    notes: Optional[str] = None
    evaluated_at: Optional[Any] = None


class ReadinessListResponse(BaseModel):
    total: int
    ready: int
    items: list[AgentReadinessItem]


class UpdateConfigRequest(BaseModel):
    enabled: Optional[bool] = Field(
        default=None,
        description="Override readiness: True forces agent active, False greys it out",
    )
    params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Param overrides merged on top of agent defaults",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Reason for the override (shown in admin UI)",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/readiness",
    response_model=ReadinessListResponse,
    summary="Feature readiness for all enrichment agents",
)
async def get_readiness(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ReadinessListResponse:
    """Return the current readiness state for every enrichment agent.

    If no evaluation has been run for this tenant yet, triggers a fresh
    synchronous evaluation before returning.  Subsequent calls return the
    persisted state instantly.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    items = await _service.get_all(org_id=tenant.org_id, db=db)
    ready_count = sum(1 for it in items if it.get("is_ready"))
    return ReadinessListResponse(
        total=len(items),
        ready=ready_count,
        items=[AgentReadinessItem(**it) for it in items],
    )


@router.post(
    "/readiness/evaluate",
    response_model=ReadinessListResponse,
    summary="Force re-evaluate feature readiness",
)
async def evaluate_readiness(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ReadinessListResponse:
    """Trigger a synchronous readiness evaluation and return fresh results.

    Use this after a bulk data import or during tenant onboarding to get
    up-to-date readiness without waiting for the nightly Celery sweep.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    items = await _service.evaluate_all(org_id=tenant.org_id, db=db)
    ready_count = sum(1 for it in items if it.get("is_ready"))
    return ReadinessListResponse(
        total=len(items),
        ready=ready_count,
        items=[AgentReadinessItem(**it) for it in items],
    )


@router.patch(
    "/readiness/{agent_name}",
    response_model=AgentReadinessItem,
    summary="Override enabled flag or params for an agent",
)
async def update_agent_config(
    agent_name: str,
    body: UpdateConfigRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> AgentReadinessItem:
    """Admin endpoint to force-enable, force-disable, or tune an agent's params.

    Setting ``enabled=False`` greys out the feature in the UI regardless of
    data readiness.  Setting ``enabled=True`` forces it active even if the
    data thresholds are not yet met (use with caution — results may be noisy).
    """
    if agent_name not in AGENT_SPECS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_name!r} not found in readiness registry",
        )
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    item = await _service.update_config(
        org_id=tenant.org_id,
        agent_name=agent_name,
        enabled=body.enabled,
        custom_params=body.params,
        notes=body.notes,
        db=db,
    )
    return AgentReadinessItem(**item)
