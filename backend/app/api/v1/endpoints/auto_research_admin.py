"""Admin: Auto-Research Trigger Endpoints.

POST /api/v1/admin/auto-research/trigger — Trigger auto-research for an organization
GET  /api/v1/admin/auto-research/status — Check status of recent auto-research runs
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Optional
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.agents.auto_research_agent import AutoResearchAgent
from app.agents.types import AgentContext, AgentResult
from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context, require_roles
from app.models.agent_run import AgentRun
from app.models.cognitive_experiment_ledger import CognitiveExperimentLedger
from app.services.config_snapshot_service import ConfigSnapshotService
from app.services.cognitive_gateway_service import CognitiveGatewayService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/auto-research", tags=["admin - auto-research"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AutoResearchTriggerRequest(BaseModel):
    """Request to trigger auto-research for an organization."""
    parameter_keys: Optional[list[str]] = None
    """Specific parameters to optimize; None means all available parameters."""
    min_improvement: Optional[float] = None
    """Minimum score improvement (absolute delta) to accept a candidate; default 0.01."""


class AutoResearchTriggerResponse(BaseModel):
    """Response from triggering auto-research."""
    job_id: str
    """Unique identifier for this auto-research job."""
    org_id: str
    """Organization ID."""
    status: str
    """Current status: accepted|declined|processing."""
    message: str
    """Human-readable status message."""
    started_at: datetime


class ExperimentRecord(BaseModel):
    """Summary of one experiment from the ledger."""
    id: str
    parameter_key: str
    baseline_value: float
    candidate_value: float
    baseline_score: float
    candidate_score: float
    score_delta: float
    status: str
    """accepted|reverted."""
    created_at: datetime
    agent_name: str
    trace_id: Optional[str]


class AutoResearchStatusResponse(BaseModel):
    """Response with auto-research run status."""
    org_id: str
    total_experiments: int
    accepted_experiments: int
    reverted_experiments: int
    recent_experiments: list[ExperimentRecord]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/trigger",
    response_model=AutoResearchTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_auto_research(
    payload: AutoResearchTriggerRequest,
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AutoResearchTriggerResponse:
    """Trigger auto-research for the organization.

    Usage:
        POST /api/v1/admin/auto-research/trigger
        {
            "parameter_keys": ["meta_learning.learning_rate"],
            "min_improvement": 0.025
        }

    **Required role:** org_admin or system_admin

    Returns:
        Job with status 'accepted' or error 'declined' if rate-limited.
    """
    # Check authorization
    if "org_admin" not in tenant.roles and "system_admin" not in tenant.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only org_admin or system_admin can trigger auto-research",
        )

    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    # Create a unique job ID and context
    job_id = str(uuid4())
    org_id = tenant.org_id

    # Build enrichment for the agent
    enrichment = {
        "org_id": org_id,
        "parameter_keys": payload.parameter_keys or [],
        "min_improvement": payload.min_improvement or 0.01,
        "session": db,
    }

    # Create agent with real dependencies
    config_service = ConfigSnapshotService(db)
    gateway = CognitiveGatewayService()
    agent = AutoResearchAgent(
        session=db,
        config_service=config_service,
        gateway=gateway,
    )

    # Create agent context
    context = {
        "tenant": {
            "org_id": org_id,
            "user_id": tenant.user_id,
            "roles": tenant.roles,
        },
        "memory": {
            "enrichment": enrichment,
        },
        "runtime": {
            "job_id": job_id,
        },
    }

    try:
        # Execute the agent (non-blocking for now; in production could delegate to Celery)
        memory_id = f"auto_research_{job_id}"
        result: AgentResult = await agent.run(memory_id, context)

        # Persist the agent run result
        agent_run = AgentRun(
            id=str(uuid4()),
            organization_id=org_id,
            memory_id=memory_id,
            agent_name=agent.name,
            agent_version=agent.version,
            status=result.status,
            confidence=result.confidence,
            started_at=result.started_at,
            finished_at=result.finished_at,
            trace_id=result.trace_id or job_id,
            outputs=result.outputs or {},
            errors=result.errors or [],
        )
        db.add(agent_run)
        await db.commit()

        status_code = "success" if result.status == "success" else "declined"
        message = (
            f"Auto-research completed: {result.outputs.get('experiments', []).__len__()} experiments run"
            if result.status == "success"
            else f"Auto-research failed: {', '.join(result.errors)}"
        )

        return AutoResearchTriggerResponse(
            job_id=job_id,
            org_id=org_id,
            status=status_code,
            message=message,
            started_at=result.started_at,
        )
    except Exception as e:
        logger.exception(f"Error triggering auto-research for org {org_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger auto-research: {str(e)}",
        )


@router.get("/status", response_model=AutoResearchStatusResponse)
async def get_auto_research_status(
    tenant: Annotated[TenantContext, Depends(get_tenant_context)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=10, ge=1, le=100),
) -> AutoResearchStatusResponse:
    """Get auto-research status and recent experiments for the organization.

    **Required role:** org_admin or system_admin

    Returns:
        Summary of recent auto-research runs and aggregate statistics.
    """
    # Check authorization
    if "org_admin" not in tenant.roles and "system_admin" not in tenant.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only org_admin or system_admin can view auto-research status",
        )

    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    org_id = tenant.org_id

    # Query recent experiments
    stmt = (
        select(CognitiveExperimentLedger)
        .where(CognitiveExperimentLedger.org_id == org_id)
        .order_by(desc(CognitiveExperimentLedger.created_at))
        .limit(limit)
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    # Aggregate statistics
    total = len(records)
    accepted = sum(1 for r in records if r.status == "accepted")
    reverted = sum(1 for r in records if r.status == "reverted")

    experiments = [
        ExperimentRecord(
            id=r.id,
            parameter_key=r.parameter_key,
            baseline_value=r.baseline_value,
            candidate_value=r.candidate_value,
            baseline_score=r.baseline_score,
            candidate_score=r.candidate_score,
            score_delta=r.score_delta,
            status=r.status,
            created_at=r.created_at,
            agent_name=r.agent_name,
            trace_id=r.trace_id,
        )
        for r in records
    ]

    return AutoResearchStatusResponse(
        org_id=org_id,
        total_experiments=total,
        accepted_experiments=accepted,
        reverted_experiments=reverted,
        recent_experiments=experiments,
    )
