"""
API Endpoints for PR-4: Tool Capability Learning & Adaptive Strategy Selection

Provides REST endpoints for managing tool capabilities, recording usage,
and getting adaptive strategy recommendations.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.core.database import get_db
from app.services.adaptive_strategy_service import AdaptiveStrategyService
from app.schemas.adaptive_strategy_pr4 import (
    ToolCapabilityCreate,
    ToolCapabilityResponse,
    ToolUsageRecord,
    ToolUsageResponse,
    ToolRecommendationRequest,
    ToolRecommendationResponse,
    StrategyAdaptationRecord,
    StrategyAdaptationResponse,
    CapabilityDiscoveryRecord,
    CapabilityDiscoveryResponse,
    StrategyHistoryEntry,
    ToolRecommendationsListResponse,
)

router = APIRouter(prefix="/v1/tools", tags=["adaptive_strategy"])


async def get_strategy_service(session: AsyncSession = Depends(get_db)) -> AdaptiveStrategyService:
    """Get the adaptive strategy service."""
    return AdaptiveStrategyService(session=session)


@router.post("/capabilities", response_model=ToolCapabilityResponse)
async def register_tool_capability(
    org_id: UUID,
    request: ToolCapabilityCreate,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Register or update a tool capability.
    
    Tool capabilities track learned performance metrics for tools,
    enabling the agent to adapt its tool selection over time.
    """
    result = await service.register_tool_capability(
        org_id=str(org_id),
        tool_name=request.tool_name,
        tool_type=request.tool_type,
        description=request.description or "",
        supported_goal_types=request.supported_goal_types,
        supported_domains=request.supported_domains,
        known_limitations=request.known_limitations,
    )
    return result


@router.post("/usage", response_model=ToolUsageResponse)
async def record_tool_usage(
    org_id: UUID,
    request: ToolUsageRecord,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Record tool usage to update performance metrics.
    
    This feedback helps the agent learn which tools are most effective
    for different types of tasks.
    """
    result = await service.record_tool_usage(
        org_id=str(org_id),
        tool_name=request.tool_name,
        goal_id=str(request.goal_id),
        success=request.success,
        execution_time=request.execution_time,
        cost=request.cost,
    )
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["error"]
        )
    
    return result


@router.post("/recommendation", response_model=ToolRecommendationResponse)
async def get_tool_recommendation(
    org_id: UUID,
    request: ToolRecommendationRequest,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Get the best tool recommendation for a goal type.
    
    Uses learned performance metrics to rank tools and recommend
    the most effective one based on success rates and efficiency.
    """
    result = await service.select_best_tool(
        org_id=str(org_id),
        goal_type=request.goal_type,
        domain=request.domain or "",
        exclude_tools=request.exclude_tools or [],
    )
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No suitable tools found for this goal type"
        )
    
    return result


@router.get("/recommendations", response_model=list[ToolRecommendationsListResponse])
async def list_tool_recommendations(
    org_id: UUID,
    goal_type: str,
    domain: str = "",
    limit: int = 5,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Get ranked list of tool recommendations for a goal.
    
    Returns multiple tools scored by reliability and success rate,
    allowing the agent to make informed choices.
    """
    results = await service.get_tool_recommendations(
        org_id=str(org_id),
        goal_id="",  # Not needed for this endpoint
        goal_type=goal_type,
        domain=domain,
        limit=limit,
    )
    
    return results


@router.post("/strategies/adaptation", response_model=StrategyAdaptationResponse)
async def record_strategy_adaptation(
    org_id: UUID,
    request: StrategyAdaptationRecord,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Record strategy adaptation for a goal.
    
    Tracks when the agent switches tools/strategies, why it made the change,
    and enables measurement of adaptation effectiveness.
    """
    result = await service.record_strategy_adaptation(
        org_id=str(org_id),
        goal_id=str(request.goal_id),
        goal_type=request.goal_type,
        previous_strategy=request.previous_strategy,
        new_strategy=request.new_strategy,
        reason=request.reason,
        confidence=request.confidence,
    )
    
    return result


@router.get("/strategies/{goal_id}", response_model=list[StrategyHistoryEntry])
async def get_strategy_history(
    org_id: UUID,
    goal_id: UUID,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Get strategy adaptation history for a goal.
    
    Returns all strategy changes made for a specific goal in chronological order,
    allowing analysis of the agent's learning process.
    """
    results = await service.get_strategy_history(
        org_id=str(org_id),
        goal_id=str(goal_id),
    )
    
    return results


@router.post("/discoveries", response_model=CapabilityDiscoveryResponse)
async def record_capability_discovery(
    org_id: UUID,
    request: CapabilityDiscoveryRecord,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Record a discovered tool or capability.
    
    Tracks new tool discoveries, capability gaps, and synergies
    that should be investigated further.
    """
    result = await service.discover_capability(
        org_id=str(org_id),
        discovery_type=request.discovery_type,
        tool_or_capability=request.tool_or_capability,
        description=request.description,
        potential_value=request.potential_value,
    )
    
    return result


@router.get("/discoveries/backlog")
async def get_discovery_backlog(
    org_id: UUID,
    status: str = "unvalidated",
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    Get pending capability discoveries for investigation.
    
    Returns discoveries that haven't been validated yet,
    prioritized by potential value.
    """
    results = await service.get_discovery_backlog(
        org_id=str(org_id),
        status=status,
    )
    
    return results


@router.get("/capabilities", response_model=dict)
async def list_tool_capabilities(
    org_id: UUID,
    tool_type: str = None,
    min_reliability: float = 0.0,
    service: AdaptiveStrategyService = Depends(get_strategy_service),
):
    """
    List all registered tool capabilities for an organization.
    
    Optionally filter by tool type or minimum reliability score.
    """
    # This is a simple mock since we don't have a list method in service yet
    return {
        "org_id": str(org_id),
        "capabilities": [],
        "message": "Use /v1/tools/recommendations instead for ranked tool lists"
    }
