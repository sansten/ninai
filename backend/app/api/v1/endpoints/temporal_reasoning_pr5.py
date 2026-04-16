"""
PR-5: Temporal Reasoning REST API Endpoints

Time-aware query endpoints for facts, sequences, trajectories, and forecasting.
"""

from datetime import datetime
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.temporal_reasoning_service import TemporalReasoningService
from app.schemas.temporal_reasoning_pr5 import (
    TemporalFactRequest,
    TemporalSequenceRequest,
    TrajectoryRequest,
    ForecastRequest,
    ActionTimingRequest,
)

router = APIRouter()


@router.post("/v1/temporal/facts/tag-validity", response_model=Dict[str, Any])
async def tag_fact_validity(
    request: TemporalFactRequest,
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Tag a fact with temporal validity interval."""
    org_id = tenant.org_id
    svc = TemporalReasoningService(session=session)

    valid_from = request.onset_timestamp if hasattr(request, "onset_timestamp") else datetime.utcnow()
    valid_to = getattr(request, "offset_timestamp", None)
    fact_id = str(getattr(request, "fact_id", "unknown"))
    change_type = getattr(request, "change_type", "stable")
    confidence = float(getattr(request, "confidence", 0.8))

    result = await svc.tag_facts_with_temporal_validity(
        org_id=org_id,
        fact_id=fact_id,
        valid_from=valid_from,
        valid_to=valid_to,
        change_type=change_type,
        confidence=confidence,
    )
    return result


@router.post("/v1/temporal/sequences/detect", response_model=Dict[str, Any])
async def detect_sequences(
    requests: List[TemporalSequenceRequest],
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Detect recurring event sequences."""
    org_id = tenant.org_id
    svc = TemporalReasoningService(session=session)

    all_sequences = []
    for req in requests:
        entities = getattr(req, "entities", [])
        timeline = [(str(e), datetime.utcnow()) for e in entities]
        seqs = await svc.detect_sequences(
            org_id=org_id,
            entity_timeline=timeline,
            min_occurrences=3,
        )
        all_sequences.extend(seqs)

    return {"sequences": all_sequences, "count": len(all_sequences)}


@router.post("/v1/temporal/trajectories/compute", response_model=Dict[str, Any])
async def compute_trajectory(
    requests: List[TrajectoryRequest],
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Analyze how quantities change over time."""
    org_id = tenant.org_id
    svc = TemporalReasoningService(session=session)

    trajectories = []
    for req in requests:
        measurements = []
        for m in req.measurements:
            ts = m.get("timestamp") or m.get("time")
            val = m.get("value", m.get("v", 0.0))
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            measurements.append((ts, float(val)))

        traj = await svc.compute_trajectories(
            org_id=org_id,
            entity_id=req.entity_id,
            quantity=req.quantity,
            measurements=measurements,
        )
        if traj:
            trajectories.append(traj)

    return {"trajectories": trajectories, "count": len(trajectories)}


@router.post("/v1/temporal/trajectories/forecast", response_model=Dict[str, Any])
async def forecast_trajectory(
    request: ForecastRequest,
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Forecast future values for a trajectory."""
    svc = TemporalReasoningService(session=session)

    trajectory = {"entity_id": request.entity_id, "measurements": [], "predicted_future": []}
    forecasts = await svc.forecast_trajectory(
        trajectory=trajectory,
        horizon_periods=request.horizon_steps,
    )

    return {"entity_id": request.entity_id, "forecasts": forecasts}


@router.get("/v1/temporal/trajectories/{entity_id}/inflection-points", response_model=Dict[str, Any])
async def get_inflection_points(
    entity_id: str,
    sensitivity: float = Query(1.5, ge=0.5, le=3.0),
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Identify significant changes in trajectory."""
    svc = TemporalReasoningService(session=session)

    trajectory = {"entity_id": entity_id, "measurements": []}
    inflections = await svc.detect_inflection_points(
        trajectory=trajectory,
        threshold_std=sensitivity,
    )

    return {"entity_id": entity_id, "inflection_points": inflections}


@router.get("/v1/temporal/query", response_model=Dict[str, Any])
async def temporal_query(
    query_type: str = Query(..., description="facts_valid_at_time | facts_updated_after | trajectory_crosses_threshold"),
    timestamp: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    threshold: Optional[float] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Execute SQL-like temporal queries."""
    org_id = tenant.org_id
    svc = TemporalReasoningService(session=session)

    kwargs: Dict[str, Any] = {}
    if timestamp:
        kwargs["timestamp"] = datetime.fromisoformat(timestamp)
    if entity_id:
        kwargs["entity_id"] = entity_id
    if threshold is not None:
        kwargs["threshold"] = threshold

    results = await svc.temporal_query(org_id=org_id, query_type=query_type, **kwargs)
    return {"query_type": query_type, "success": True, "results": results}


@router.post("/v1/temporal/when-should-act", response_model=Dict[str, Any])
async def when_should_act(
    request: ActionTimingRequest,
    session: AsyncSession = Depends(get_db),
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, Any]:
    """Estimate optimal time to act given goal and trajectory."""
    org_id = tenant.org_id
    svc = TemporalReasoningService(session=session)

    measurements = [
        {"timestamp": m.get("timestamp") or m.get("time"), "value": float(m.get("value", m.get("v", 0.0)))}
        for m in request.measurements
    ]
    trajectory = {"entity_id": request.entity_id, "measurements": measurements, "predicted_future": []}
    goal_context = {"critical_threshold": request.threshold}

    result = await svc.when_should_act(
        org_id=org_id,
        goal_context=goal_context,
        trajectory=trajectory,
        action_lead_time_hours=request.lookahead_hours,
    )

    if not result:
        return {"status": "no_action_needed", "reason": "Trajectory within acceptable range"}

    return result
