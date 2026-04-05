"""Prospective Simulation endpoint (Feature 24.9).

Exposes the ProspectiveSimulationService as:

  POST /cognitive/simulate

Implements the roadmap-specified request contract:
  {
    "scenario": "What if we delay the deployment by 2 weeks?",
    "horizon_days": 14,
    "variables_to_watch": ["incident_rate", "customer_satisfaction"]
  }
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.middleware.tenant_context import TenantContext, require_org_admin
from app.services.prospective_simulation_service import ProspectiveSimulationService

router = APIRouter()


@router.post("/simulate")
async def cognitive_simulate(
    payload: dict[str, Any] = Body(...),
    tenant: TenantContext = Depends(require_org_admin()),
) -> dict[str, Any]:
    """Run a prospective mental-time-travel simulation from a scenario.

    Request body:
      scenario           (str, required)   — "What if …" question
      horizon_days       (int, optional, default 14)
      variables_to_watch (list[str], optional)
      historical_episodes (list[dict], optional) — past episodes for context
      current_metrics    (dict[str, float], optional) — current baseline values

    Returns a simulation timeline with per-step variable projections,
    success probability, risk events, and precautions.
    """
    scenario = str(payload.get("scenario") or "").strip()
    if not scenario:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="scenario is required",
        )

    horizon_days = int(payload.get("horizon_days") or 14)
    if horizon_days < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="horizon_days must be >= 1",
        )

    svc = ProspectiveSimulationService()
    result = svc.simulate(
        scenario=scenario,
        horizon_days=horizon_days,
        variables_to_watch=list(payload.get("variables_to_watch") or []),
        historical_episodes=list(payload.get("historical_episodes") or []),
        current_metrics=dict(payload.get("current_metrics") or {}),
    )

    return {
        "scenario": result.scenario,
        "horizon_days": result.horizon_days,
        "variables_to_watch": result.variables_to_watch,
        "simulation_timeline": [
            {
                "step": e.step,
                "horizon_date": e.horizon_date,
                "event_description": e.event_description,
                "probability": e.probability,
                "severity_change": e.severity_change,
                "entities_affected": e.entities_affected,
                "variable_projections": e.variable_projections,
            }
            for e in result.simulation_timeline
        ],
        "variable_summaries": [
            {
                "variable": vs.variable,
                "baseline": vs.baseline,
                "projected_values": vs.projected_values,
                "trend": vs.trend,
                "peak_risk_step": vs.peak_risk_step,
            }
            for vs in result.variable_summaries
        ],
        "success_probability": result.success_probability,
        "risk_events": result.risk_events,
        "recommended_precautions": result.recommended_precautions,
        "confidence": result.confidence,
        "simulated_at": result.simulated_at,
        "org_id": tenant.org_id,
    }
