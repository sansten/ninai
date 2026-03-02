"""
PR-8: Emotional & Affective Memory API Endpoints

Provides endpoints for:
- Analyzing emotional content
- Tracking emotional trajectories
- De-escalating interactions
- Adjusting tone empathetically
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.affective_memory_pr8 import (
    AffectiveAnalysisResponse,
    AffectiveMemoryRequest,
    EmotionalTrajectoryResponse,
    EscalationRiskResponse,
    EmpatheticResponseRequest,
    EmpatheticResponseResponse,
    InteractionOutcomeRequest,
    InteractionOutcomeResponse,
)
from app.services.affective_analysis_service import (
    AffectiveAnalysisService,
    EmotionalRegulationService,
)


router = APIRouter()


async def get_affective_service(
    session: AsyncSession = Depends(get_db),
) -> AffectiveAnalysisService:
    return AffectiveAnalysisService(session=session)


async def get_regulation_service(
    session: AsyncSession = Depends(get_db),
) -> EmotionalRegulationService:
    return EmotionalRegulationService(session=session)


@router.post(
    "/v1/affective/analyze",
    response_model=AffectiveAnalysisResponse,
)
async def analyze_emotional_content(
    request: AffectiveMemoryRequest,
    service: AffectiveAnalysisService = Depends(get_affective_service),
) -> AffectiveAnalysisResponse:
    """
    Analyze emotional valence, arousal, and tags from memory content.
    
    Returns:
    - valence: -1.0 (very negative) to +1.0 (very positive)
    - arousal: 0 (calm) to 1 (intense/excited)
    - emotional_tags: categorical emotional labels
    - significance: how emotionally important (0-1)
    """
    analysis = await service.analyze_memory_affect(
        memory_content=request.memory_content,
        memory_id="",  # Not needed for this operation
        organization_id="",  # Will be from auth context in real implementation
        user_ids=request.user_ids,
    )
    return AffectiveAnalysisResponse(**analysis)


@router.get(
    "/v1/user/{user_id}/emotional-trajectory",
    response_model=EmotionalTrajectoryResponse,
)
async def get_emotional_trajectory(
    user_id: str,
    lookback_days: int = Query(30, description="Days to look back"),
    service: AffectiveAnalysisService = Depends(get_affective_service),
    session: AsyncSession = Depends(get_db),
) -> EmotionalTrajectoryResponse:
    """
    Get user's emotional state over time.
    
    Returns trajectory with trend analysis, current state, escalation risk, and
    recommended de-escalation strategies.
    """
    # Note: organization_id should come from auth context
    organization_id = ""  # Placeholder - use from auth in production
    
    trajectory_data = await service.compute_user_emotional_trajectory(
        user_id=user_id,
        organization_id=organization_id,
        lookback_days=lookback_days,
        session=session,
    )
    
    # Convert measurements format
    measurements = [
        {
            "timestamp": m["timestamp"],
            "valence": m["valence"],
            "arousal": m["arousal"],
        }
        for m in trajectory_data.get("measurements", [])
    ]
    trajectory_data["measurements"] = measurements
    
    return EmotionalTrajectoryResponse(**trajectory_data)


@router.get(
    "/v1/user/{user_id}/escalation-risk",
    response_model=EscalationRiskResponse,
)
async def get_escalation_risk(
    user_id: str,
    service: AffectiveAnalysisService = Depends(get_affective_service),
    regulation_service: EmotionalRegulationService = Depends(get_regulation_service),
    session: AsyncSession = Depends(get_db),
) -> EscalationRiskResponse:
    """
    Get escalation risk for a user.
    
    Returns:
    - escalation_risk: 0-1 probability of escalation
    - should_escalate_to_human: whether human should take over
    - recommended_strategy: suggested de-escalation approach
    """
    organization_id = ""  # Placeholder - use from auth in production
    
    risk = await service.detect_escalation_risk(
        user_id=user_id,
        organization_id=organization_id,
        session=session,
    )
    
    should_escalate = await regulation_service.should_escalate_to_human(
        user_id=user_id,
        organization_id=organization_id,
        session=session,
    )
    
    strategy = await regulation_service.suggest_de_escalation_strategy(
        user_id=user_id,
        organization_id=organization_id,
        session=session,
    )
    
    return EscalationRiskResponse(
        user_id=user_id,
        escalation_risk=risk,
        should_escalate_to_human=should_escalate,
        recommended_strategy=strategy,
    )


@router.post(
    "/v1/respond-empathetically",
    response_model=EmpatheticResponseResponse,
)
async def adjust_response_empathetically(
    request: EmpatheticResponseRequest,
    service: AffectiveAnalysisService = Depends(get_affective_service),
) -> EmpatheticResponseResponse:
    """
    Adjust response tone to match user's emotional state.
    
    Makes response more empathetic by:
    - Adding validation of feelings
    - Matching arousal level
    - Personalizing to emotional context
    """
    user_state = {
        "valence": request.user_valence,
        "arousal": request.user_arousal,
        "tags": request.user_emotional_tags or [],
    }
    
    adjusted = await service.select_empathetic_tone(
        user_emotional_state=user_state,
        response_content=request.response_content,
    )
    
    # Determine what adjustments were made
    adjustments = []
    if request.user_arousal > 0.7:
        adjustments.append("Added calming language for high arousal")
    if request.user_valence < -0.5:
        adjustments.append("Added empathy prefix for negative sentiment")
    if "anxiety" in (request.user_emotional_tags or []):
        adjustments.append("Addressed specific concerns")
    
    return EmpatheticResponseResponse(
        original_response=request.response_content,
        adjusted_response=adjusted,
        tone_adjustments=adjustments or ["Response structure optimized for emotional context"],
    )


@router.post(
    "/v1/interaction-outcome",
    response_model=InteractionOutcomeResponse,
)
async def record_interaction_outcome(
    user_id: str,
    request: InteractionOutcomeRequest,
    regulation_service: EmotionalRegulationService = Depends(get_regulation_service),
    session: AsyncSession = Depends(get_db),
) -> InteractionOutcomeResponse:
    """
    Record the outcome of an emotional interaction.
    
    Used for:
    - Training emotion detection
    - Understanding escalation patterns
    - Improving de-escalation strategies
    """
    organization_id = ""  # Placeholder - use from auth in production
    
    event_id = await regulation_service.record_interaction_outcome(
        user_id=user_id,
        organization_id=organization_id,
        interaction_content=request.interaction_content,
        initial_valence=request.initial_valence,
        initial_arousal=request.initial_arousal,
        final_valence=request.final_valence,
        final_arousal=request.final_arousal,
        agent_response_tone=request.agent_response_tone,
        de_escalation_applied=request.de_escalation_applied,
        outcome=request.outcome,
        session=session,
    )
    
    return InteractionOutcomeResponse(
        event_id=event_id,
        user_id=user_id,
        was_escalation=request.initial_valence < -0.3 or request.initial_arousal > 0.6,
        was_de_escalated=(
            request.final_valence is not None
            and request.final_arousal is not None
            and request.final_valence > request.initial_valence
        ),
        outcome=request.outcome,
        created_at=None,  # Will be set by DB
    )
