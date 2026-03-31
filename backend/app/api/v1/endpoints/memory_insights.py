"""Memory insights endpoint — aggregate analytics for the insights dashboard.

Provides tenant-scoped aggregate statistics across the memory corpus:
- Totals and active/inactive breakdown
- Creation trend (last 30 days)
- Scope distribution
- Memory type distribution
- Agent enrichment coverage (distinct memories per agent)
- Pending human review count
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import and_, case, cast, func, select
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun
from app.models.memory import MemoryMetadata

router = APIRouter()

_ENRICHMENT_AGENTS = {
    "SemanticNormalizationAgent",
    "EntityResolutionAgent",
    "ContextAmplifierAgent",
    "SiloPropagationAgent",
    "OrgAttentionAgent",
    "ProactiveMemoryPushAgent",
    "CausalReasoningAgent",
    "ConflictDetectionAgent",
    "AdaptiveConflictResolutionAgent",
    "MemoryDecayAgent",
    "MemoryConsolidationAgent",
    "TemporalReasoningAgent",
    "EpisodicGroupingAgent",
    "CredibilityAgent",
    "PlaybookAgent",
    "GoalDecompositionAgent",
    "UncertaintyReportingAgent",
    "NarrativeSynthesisAgent",
    "FeedbackIntegrationAgent",
    "AnomalyDetectionAgent",
    "QueryIntelligenceAgent",
}

_REVIEW_QUEUE_AGENT = "HumanReviewQueueAgent"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DistributionItem(BaseModel):
    label: str
    count: int


class TrendPoint(BaseModel):
    date: str
    count: int


class AgentCoverage(BaseModel):
    agent_name: str
    memory_count: int


class InsightsSummary(BaseModel):
    org_id: str
    total_memories: int
    active_memories: int
    inactive_memories: int
    pending_review: int
    enriched_memories: int
    creation_trend: list[TrendPoint] = Field(default_factory=list)
    scope_distribution: list[DistributionItem] = Field(default_factory=list)
    type_distribution: list[DistributionItem] = Field(default_factory=list)
    agent_coverage: list[AgentCoverage] = Field(default_factory=list)
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/insights",
    response_model=InsightsSummary,
    summary="Aggregate memory insights for the dashboard",
    tags=["Memory Insights"],
)
async def get_memory_insights(
    days: int = Query(30, ge=1, le=90, description="Lookback window for creation trend"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> InsightsSummary:
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )

    org_id = tenant.org_id
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # -- totals ----------------------------------------------------------
    total_stmt = select(
        func.count(MemoryMetadata.id).label("total"),
        func.sum(case((MemoryMetadata.is_active.is_(True), 1), else_=0)).label("active"),
        func.sum(case((MemoryMetadata.is_active.is_(False), 1), else_=0)).label("inactive"),
    ).where(MemoryMetadata.organization_id == org_id)

    total_row = (await db.execute(total_stmt)).one()
    total_memories = int(total_row.total or 0)
    active_memories = int(total_row.active or 0)
    inactive_memories = int(total_row.inactive or 0)

    # -- creation trend --------------------------------------------------
    _day = func.date_trunc("day", MemoryMetadata.created_at)
    trend_stmt = (
        select(
            _day.label("day"),
            func.count(MemoryMetadata.id).label("cnt"),
        )
        .where(
            and_(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.created_at >= cutoff,
            )
        )
        .group_by(_day)
        .order_by(_day.asc())
    )
    trend_rows = (await db.execute(trend_stmt)).all()
    creation_trend = [
        TrendPoint(date=row.day.strftime("%Y-%m-%d"), count=int(row.cnt))
        for row in trend_rows
    ]

    # -- scope distribution ----------------------------------------------
    scope_stmt = (
        select(MemoryMetadata.scope, func.count(MemoryMetadata.id).label("cnt"))
        .where(MemoryMetadata.organization_id == org_id)
        .group_by(MemoryMetadata.scope)
        .order_by(func.count(MemoryMetadata.id).desc())
    )
    scope_rows = (await db.execute(scope_stmt)).all()
    scope_distribution = [
        DistributionItem(label=row.scope, count=int(row.cnt)) for row in scope_rows
    ]

    # -- memory type distribution ----------------------------------------
    type_stmt = (
        select(MemoryMetadata.memory_type, func.count(MemoryMetadata.id).label("cnt"))
        .where(MemoryMetadata.organization_id == org_id)
        .group_by(MemoryMetadata.memory_type)
        .order_by(func.count(MemoryMetadata.id).desc())
    )
    type_rows = (await db.execute(type_stmt)).all()
    type_distribution = [
        DistributionItem(label=row.memory_type, count=int(row.cnt)) for row in type_rows
    ]

    # -- agent enrichment coverage ---------------------------------------
    coverage_stmt = (
        select(
            AgentRun.agent_name,
            func.count(func.distinct(AgentRun.memory_id)).label("cnt"),
        )
        .where(
            and_(
                AgentRun.organization_id == org_id,
                AgentRun.status == "success",
                AgentRun.agent_name.in_(_ENRICHMENT_AGENTS),
            )
        )
        .group_by(AgentRun.agent_name)
        .order_by(func.count(func.distinct(AgentRun.memory_id)).desc())
    )
    coverage_rows = (await db.execute(coverage_stmt)).all()
    agent_coverage = [
        AgentCoverage(agent_name=row.agent_name, memory_count=int(row.cnt))
        for row in coverage_rows
    ]

    # -- enriched memories (any enrichment agent run) --------------------
    enriched_stmt = select(
        func.count(func.distinct(AgentRun.memory_id))
    ).where(
        and_(
            AgentRun.organization_id == org_id,
            AgentRun.status == "success",
            AgentRun.agent_name.in_(_ENRICHMENT_AGENTS),
        )
    )
    enriched_count = (await db.execute(enriched_stmt)).scalar() or 0

    # -- pending review count -------------------------------------------
    review_stmt = select(
        func.count(func.distinct(AgentRun.memory_id))
    ).where(
        and_(
            AgentRun.organization_id == org_id,
            AgentRun.status == "success",
            AgentRun.agent_name == _REVIEW_QUEUE_AGENT,
        )
    )
    pending_review = int((await db.execute(review_stmt)).scalar() or 0)

    return InsightsSummary(
        org_id=org_id,
        total_memories=total_memories,
        active_memories=active_memories,
        inactive_memories=inactive_memories,
        pending_review=pending_review,
        enriched_memories=int(enriched_count),
        creation_trend=creation_trend,
        scope_distribution=scope_distribution,
        type_distribution=type_distribution,
        agent_coverage=agent_coverage,
        retrieved_at=now,
    )
