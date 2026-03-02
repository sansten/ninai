"""
PR-5: Temporal Reasoning REST API Endpoints

Time-aware query endpoints for facts, sequences, trajectories, and forecasting.
"""

from datetime import datetime
from typing import List, Optional, Any, Dict
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.temporal_reasoning_service import TemporalReasoningService
from app.schemas.temporal_reasoning_pr5 import (
    TemporalFactRequest,
    TemporalSequenceRequest,
    TrajectoryRequest,
    ForecastRequest,
    TemporalQueryRequest,
    ActionTimingRequest,
    TemporalFactResponse,
    TemporalSequenceResponse,
    TrajectoryResponse,
    ForecastResponse,
    InflectionPointResponse,
    ActionTimingResponse,
)

router = APIRouter()


@router.post("/v1/temporal/facts/tag-validity", response_model=Dict[str, Any])
async def tag_fact_validity(
    request: TemporalFactRequest,
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Tag a fact with temporal validity interval.
    
    Args:
        request: Temporal fact request with:
            - fact_id: Fact being tagged
            - onset_timestamp: When fact becomes true
            - offset_timestamp: When fact stops being true (optional)
            - confidence: How sure? (0-1)
            - change_type: onset | offset | stable | transient
    
    Returns:
        Temporal fact metadata with ID
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    result = await svc.tag_facts_with_temporal_validity([request])
    
    return result


@router.post("/v1/temporal/sequences/detect", response_model=Dict[str, Any])
async def detect_sequences(
    requests: List[TemporalSequenceRequest],
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Detect recurring event sequences.
    
    Args:
        requests: List of sequence detection requests with:
            - entities: Ordered list of entity IDs
            - min_gap_seconds: Minimum gap between events
            - max_gap_seconds: Maximum gap between events
            - min_strength: Minimum pattern strength (0-1)
    
    Returns:
        List of detected sequences with pattern metadata
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    result = await svc.detect_sequences(requests)
    
    return result


@router.post("/v1/temporal/trajectories/compute", response_model=Dict[str, Any])
async def compute_trajectory(
    requests: List[TrajectoryRequest],
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Analyze how quantities change over time.
    
    Args:
        requests: List of trajectory requests with:
            - entity_id: What entity (user, problem, metric)
            - quantity: What metric (sentiment, strength, rate)
            - measurements: List of {timestamp, value} measurements
    
    Returns:
        Trajectory analysis with trend, inflection points, forecast
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    result = await svc.compute_trajectories(requests)
    
    return result


@router.post("/v1/temporal/trajectories/forecast", response_model=Dict[str, Any])
async def forecast_trajectory(
    request: ForecastRequest,
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Forecast future values for a trajectory.
    
    Args:
        request: Forecast request with:
            - entity_id: ID of trajectory to forecast
            - horizon_steps: How many periods ahead?
            - confidence_level: Desired confidence (0.8-0.99)
    
    Returns:
        List of forecasts with confidence intervals
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    # For demo, use empty measurements (would retrieve from DB in real app)
    result = await svc.forecast_trajectory(
        request,
        measurements=[],
    )
    
    return result


@router.get("/v1/temporal/trajectories/{entity_id}/inflection-points", response_model=Dict[str, Any])
async def get_inflection_points(
    entity_id: str,
    sensitivity: float = Query(1.5, ge=0.5, le=3.0),
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Identify significant changes in trajectory.
    
    Args:
        entity_id: ID of trajectory to analyze
        sensitivity: How many std devs = inflection point?
    
    Returns:
        List of inflection points with timestamps and severity
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    # For demo, use empty measurements (would retrieve from DB in real app)
    result = await svc.detect_inflection_points(
        entity_id,
        measurements=[],
        sensitivity=sensitivity,
    )
    
    return result


@router.get("/v1/temporal/query", response_model=Dict[str, Any])
async def temporal_query(
    query_type: str = Query(..., description="facts_valid_at_time | facts_updated_after | trajectory_crosses_threshold"),
    timestamp: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    threshold: Optional[float] = Query(None),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Execute SQL-like temporal queries.
    
    Examples:
    - query_type=facts_valid_at_time&timestamp=2026-03-01T12:00:00
    - query_type=facts_updated_after&after=2026-02-01T00:00:00
    - query_type=trajectory_crosses_threshold&entity_id=xyz&threshold=0.8
    
    Args:
        query_type: Type of temporal query
        timestamp: Query timestamp (ISO format)
        entity_id: Entity to query
        threshold: Threshold value for crossing queries
        start_time: Query start time
        end_time: Query end time
    
    Returns:
        Query results
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    # Parse timestamps if provided
    ts = None
    if timestamp:
        ts = datetime.fromisoformat(timestamp)
    
    result = await svc.temporal_query(
        query_type=query_type,
        timestamp=ts,
        entity_id=entity_id,
        threshold=threshold,
        start_time=start_time,
        end_time=end_time,
    )
    
    return {
        "query_type": query_type,
        "success": result.get("success", True),
        "results": result,
    }


@router.post("/v1/temporal/when-should-act", response_model=Dict[str, Any])
async def when_should_act(
    request: ActionTimingRequest,
    session: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Estimate optimal time to act given goal and trajectory.
    
    Args:
        request: Action timing request with:
            - entity_id: Entity to analyze
            - threshold: Critical threshold value
            - measurements: Current measurements
            - lookahead_hours: How far to look ahead?
    
    Returns:
        Recommended action time and reasoning
    """
    from app.core.auth import get_current_org_id
    org_id = await get_current_org_id()
    
    svc = TemporalReasoningService(db_session=session, org_id=org_id)
    
    result = await svc.when_should_act(
        entity_id=request.entity_id,
        threshold=request.threshold,
        measurements=request.measurements,
        lookahead_hours=request.lookahead_hours,
    )
    
    if not result:
        return {
            "status": "no_action_needed",
            "reason": "Trajectory within acceptable range",
        }
    
    return result
