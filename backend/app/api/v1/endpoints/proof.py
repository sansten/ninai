"""Proof layer endpoints for scorecard and ROI reporting (Phase 54 Slice 2)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.proof_scorecard_service import ProofScorecardService

router = APIRouter()
_service = ProofScorecardService()


class ProofScorecardRequest(BaseModel):
    records: list[dict[str, Any]] = Field(default_factory=list)
    baseline: dict[str, float] = Field(default_factory=dict)


class ProofScorecardResponse(BaseModel):
    lead_time_gain_pct: float
    sla_avoidance_rate: float
    mttr_delta_pct: float
    false_escalation_reduction_pct: float
    incidents_count: int
    score: float
    reproducibility_hash: str


class MonthlyImpactRequest(BaseModel):
    month: str = Field(description="Month label, e.g. 2026-03")
    records: list[dict[str, Any]] = Field(default_factory=list)
    baseline: dict[str, float] = Field(default_factory=dict)
    labor_cost_per_hour: float = 120.0
    false_escalation_cost: float = 250.0
    monthly_operating_cost: float = 3000.0


class MonthlyImpactResponse(BaseModel):
    month: str
    tenant_id: str
    incidents_count: int
    lead_time_saved_hours: float
    mttr_saved_hours: float
    avoided_sla_penalty: float
    estimated_savings: float
    operating_cost: float
    net_impact: float
    roi_pct: float
    reproducibility_hash: str


@router.post("/scorecard", response_model=ProofScorecardResponse)
async def compute_proof_scorecard(
    body: ProofScorecardRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ProofScorecardResponse:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    scorecard = _service.compute_scorecard(records=body.records, baseline=body.baseline)
    reproducibility_hash = _service.reproducibility_hash(records=body.records, baseline=body.baseline)

    return ProofScorecardResponse(
        lead_time_gain_pct=scorecard.lead_time_gain_pct,
        sla_avoidance_rate=scorecard.sla_avoidance_rate,
        mttr_delta_pct=scorecard.mttr_delta_pct,
        false_escalation_reduction_pct=scorecard.false_escalation_reduction_pct,
        incidents_count=scorecard.incidents_count,
        score=scorecard.score,
        reproducibility_hash=reproducibility_hash,
    )


@router.post("/monthly-impact", response_model=MonthlyImpactResponse)
async def compute_monthly_impact(
    body: MonthlyImpactRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MonthlyImpactResponse:
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    report = _service.compute_monthly_roi_report(
        tenant_id=tenant.org_id,
        month=body.month,
        records=body.records,
        baseline=body.baseline,
        labor_cost_per_hour=body.labor_cost_per_hour,
        false_escalation_cost=body.false_escalation_cost,
        monthly_operating_cost=body.monthly_operating_cost,
    )
    reproducibility_hash = _service.reproducibility_hash(records=body.records, baseline=body.baseline)

    return MonthlyImpactResponse(
        month=report.month,
        tenant_id=report.tenant_id,
        incidents_count=report.incidents_count,
        lead_time_saved_hours=report.lead_time_saved_hours,
        mttr_saved_hours=report.mttr_saved_hours,
        avoided_sla_penalty=report.avoided_sla_penalty,
        estimated_savings=report.estimated_savings,
        operating_cost=report.operating_cost,
        net_impact=report.net_impact,
        roi_pct=report.roi_pct,
        reproducibility_hash=reproducibility_hash,
    )