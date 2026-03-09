"""Memory enrichment read API — Phase 26 / 27.

Exposes the enrichment pipeline outputs as a clean read API so dashboards,
copilots, and external tools can query a memory's fully-enriched state without
knowing internal agent structure.

Endpoints:
  GET  /memories/{memory_id}/enrichment         — full merged enrichment dict
  GET  /memories/{memory_id}/narrative          — Phase 23 narrative output
  GET  /memories/{memory_id}/uncertainty        — Phase 22 uncertainty report
  GET  /memories/{memory_id}/anomalies          — Phase 25 anomaly report
  GET  /memories/{memory_id}/query-intelligence — Phase 27 query intelligence
  POST /memories/{memory_id}/feedback           — submit human review verdict
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun
from app.models.memory_feedback import MemoryFeedback
from app.models.base import generate_uuid


router = APIRouter()

# ---------------------------------------------------------------------------
# Known enrichment agents (Phase 6–25)
# ---------------------------------------------------------------------------

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

_NARRATIVE_AGENTS = {"NarrativeSynthesisAgent"}
_UNCERTAINTY_AGENTS = {"UncertaintyReportingAgent"}
_ANOMALY_AGENTS = {"AnomalyDetectionAgent"}
_QUERY_INTELLIGENCE_AGENTS = {"QueryIntelligenceAgent"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class EnrichmentResponse(BaseModel):
    memory_id: str
    enrichment: dict[str, Any]
    agent_count: int
    retrieved_at: datetime


class NarrativeResponse(BaseModel):
    memory_id: str
    narrative_text: str | None = None
    key_entities: list[str] = Field(default_factory=list)
    timeline_summary: str | None = None
    action_items: list[str] = Field(default_factory=list)
    tone: str | None = None
    confidence: float | None = None
    retrieved_at: datetime


class UncertaintyResponse(BaseModel):
    memory_id: str
    uncertainty_level: str | None = None
    unresolved_conflicts: list[dict] = Field(default_factory=list)
    low_confidence_fields: list[str] = Field(default_factory=list)
    review_recommended: bool | None = None
    uncertainty_score: float | None = None
    confidence: float | None = None
    retrieved_at: datetime


class AnomalyResponse(BaseModel):
    memory_id: str
    anomaly_detected: bool | None = None
    anomaly_type: str | None = None
    severity: str | None = None
    affected_fields: list[str] = Field(default_factory=list)
    anomaly_score: float | None = None
    confidence: float | None = None
    retrieved_at: datetime


class QueryIntelligenceResponse(BaseModel):
    memory_id: str
    query_intent: str | None = None
    extracted_entities: list[str] = Field(default_factory=list)
    dynamic_filters: dict = Field(default_factory=dict)
    suggested_agents: list[str] = Field(default_factory=list)
    confidence: float | None = None
    retrieved_at: datetime


class FeedbackRequest(BaseModel):
    verdict: str | None = Field(
        default=None,
        description="approve | reject | partial",
    )
    credibility_override: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Explicit credibility score from reviewer",
    )
    resolution_confirmed: bool | None = Field(
        default=None,
        description="True = all conflicts resolved, False = still open",
    )
    playbook_correct: bool | None = Field(
        default=None,
        description="False = playbook match was wrong",
    )
    note: str | None = Field(
        default=None, max_length=2000,
        description="Optional free-text note from reviewer",
    )


class FeedbackResponse(BaseModel):
    feedback_id: str
    memory_id: str
    verdict: str | None
    accepted_at: datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_agent_outputs(
    db: AsyncSession,
    org_id: str,
    memory_id: str,
    agent_names: set[str],
) -> list[AgentRun]:
    stmt = (
        select(AgentRun)
        .where(
            AgentRun.organization_id == org_id,
            AgentRun.memory_id == memory_id,
            AgentRun.status == "success",
            AgentRun.agent_name.in_(agent_names),
        )
        .order_by(AgentRun.finished_at.asc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


def _merge_outputs(runs: list[AgentRun]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for run in runs:
        if isinstance(run.outputs, dict):
            merged.update(run.outputs)
    return merged


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/{memory_id}/enrichment",
    response_model=EnrichmentResponse,
    summary="Full enrichment for a memory",
)
async def get_enrichment(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_agent_outputs(db, tenant.org_id, memory_id, _ENRICHMENT_AGENTS)
    if not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No enrichment found")
    enrichment = _merge_outputs(runs)
    return EnrichmentResponse(
        memory_id=memory_id,
        enrichment=enrichment,
        agent_count=len(runs),
        retrieved_at=datetime.now(timezone.utc),
    )


@router.get(
    "/{memory_id}/narrative",
    response_model=NarrativeResponse,
    summary="Narrative synthesis for a memory",
)
async def get_narrative(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_agent_outputs(db, tenant.org_id, memory_id, _NARRATIVE_AGENTS)
    if not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No narrative found")
    outputs = _merge_outputs(runs)
    return NarrativeResponse(
        memory_id=memory_id,
        retrieved_at=datetime.now(timezone.utc),
        narrative_text=outputs.get("narrative_text"),
        key_entities=outputs.get("key_entities") or [],
        timeline_summary=outputs.get("timeline_summary"),
        action_items=outputs.get("action_items") or [],
        tone=outputs.get("tone"),
        confidence=outputs.get("confidence"),
    )


@router.get(
    "/{memory_id}/uncertainty",
    response_model=UncertaintyResponse,
    summary="Uncertainty report for a memory",
)
async def get_uncertainty(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_agent_outputs(db, tenant.org_id, memory_id, _UNCERTAINTY_AGENTS)
    if not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No uncertainty report found")
    outputs = _merge_outputs(runs)
    return UncertaintyResponse(
        memory_id=memory_id,
        retrieved_at=datetime.now(timezone.utc),
        uncertainty_level=outputs.get("uncertainty_level"),
        unresolved_conflicts=outputs.get("unresolved_conflicts") or [],
        low_confidence_fields=outputs.get("low_confidence_fields") or [],
        review_recommended=outputs.get("review_recommended"),
        uncertainty_score=outputs.get("uncertainty_score"),
        confidence=outputs.get("confidence"),
    )


@router.get(
    "/{memory_id}/anomalies",
    response_model=AnomalyResponse,
    summary="Anomaly detection report for a memory",
)
async def get_anomalies(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_agent_outputs(db, tenant.org_id, memory_id, _ANOMALY_AGENTS)
    if not runs:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No anomaly report found")
    outputs = _merge_outputs(runs)
    return AnomalyResponse(
        memory_id=memory_id,
        retrieved_at=datetime.now(timezone.utc),
        anomaly_detected=outputs.get("anomaly_detected"),
        anomaly_type=outputs.get("anomaly_type"),
        severity=outputs.get("severity"),
        affected_fields=outputs.get("affected_fields") or [],
        anomaly_score=outputs.get("anomaly_score"),
        confidence=outputs.get("confidence"),
    )


@router.get(
    "/{memory_id}/query-intelligence",
    response_model=QueryIntelligenceResponse,
    summary="Query intelligence report for a memory",
)
async def get_query_intelligence(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_agent_outputs(db, tenant.org_id, memory_id, _QUERY_INTELLIGENCE_AGENTS)
    if not runs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No query intelligence report found",
        )
    outputs = _merge_outputs(runs)
    return QueryIntelligenceResponse(
        memory_id=memory_id,
        retrieved_at=datetime.now(timezone.utc),
        query_intent=outputs.get("query_intent"),
        extracted_entities=outputs.get("extracted_entities") or [],
        dynamic_filters=outputs.get("dynamic_filters") or {},
        suggested_agents=outputs.get("suggested_agents") or [],
        confidence=outputs.get("confidence"),
    )


@router.post(
    "/{memory_id}/enrichment-feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit human review verdict for a memory's enrichment",
)
async def submit_feedback(
    memory_id: str,
    body: FeedbackRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    feedback = MemoryFeedback(
        id=generate_uuid(),
        organization_id=tenant.org_id,
        memory_id=memory_id,
        actor_id=tenant.user_id,
        feedback_type="enrichment_verdict",
        target_agent="FeedbackIntegrationAgent",
        payload={
            "verdict": body.verdict,
            "credibility_override": body.credibility_override,
            "resolution_confirmed": body.resolution_confirmed,
            "playbook_correct": body.playbook_correct,
            "note": body.note,
        },
        is_applied=False,
    )
    db.add(feedback)
    await db.commit()
    return FeedbackResponse(
        feedback_id=feedback.id,
        memory_id=memory_id,
        verdict=body.verdict,
        accepted_at=datetime.now(timezone.utc),
    )
