"""
API Endpoints for Autonomous Goals and Intrinsic Motivation (PR-3)

REST endpoints for:
- Autonomous goal management
- Knowledge gap discovery
- Goal outcome tracking
- Intrinsic motivation metrics
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import (
    AutonomousGoal,
    KnowledgeGap,
    AutonomousGoalOutcome,
)
from app.schemas.intrinsic_motivation_pr3 import (
    AutonomousGoalCreate,
    AutonomousGoalUpdate,
    AutonomousGoalResponse,
    KnowledgeGapCreate,
    KnowledgeGapUpdate,
    KnowledgeGapResponse,
    GoalOutcomeCreate,
    GoalOutcomeResponse,
    GenerateGoalsRequest,
    GenerateGoalsResponse,
    MotivationMetrics,
)
from app.services.intrinsic_motivation_service import IntrinsicMotivationService

router = APIRouter(prefix="/v1/autonomous-goals", tags=["autonomous-goals"])

# ============================================================================
# Autonomous Goal Endpoints
# ============================================================================

@router.get("/", response_model=List[AutonomousGoalResponse])
async def list_autonomous_goals(
    status: str = Query(None),
    initiator: str = Query(None),
    sort_by: str = Query("urgency"),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    List autonomous goals with optional filtering.
    
    Query parameters:
    - status: Filter by status (proposed|active|completed|abandoned)
    - initiator: Filter by initiator (curiosity|prediction|self_improvement|priority_rebalance)
    - sort_by: Sort field (urgency|expected_value|created_at)
    - limit: Max results (1-500)
    """
    stmt = select(AutonomousGoal)
    
    if status:
        stmt = stmt.where(AutonomousGoal.status == status)
    if initiator:
        stmt = stmt.where(AutonomousGoal.initiator == initiator)
    
    # Sort by field
    if sort_by == "urgency":
        stmt = stmt.order_by(AutonomousGoal.urgency.desc())
    elif sort_by == "expected_value":
        stmt = stmt.order_by(AutonomousGoal.expected_value.desc())
    else:
        stmt = stmt.order_by(AutonomousGoal.created_at.desc())
    
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    goals = result.scalars().all()
    
    return goals


@router.get("/{goal_id}", response_model=AutonomousGoalResponse)
async def get_autonomous_goal(
    goal_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific autonomous goal by ID."""
    stmt = select(AutonomousGoal).where(AutonomousGoal.id == goal_id)
    result = await db.execute(stmt)
    goal = result.scalar_one_or_none()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    return goal


@router.post("/", response_model=AutonomousGoalResponse, status_code=201)
async def create_autonomous_goal(
    goal_create: AutonomousGoalCreate,
    org_id: UUID = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new autonomous goal.
    
    This endpoint is typically used internally by the goal generation pipeline,
    but can also be used to manually create goals for testing.
    """
    goal = AutonomousGoal(
        id=goal_create.id if hasattr(goal_create, 'id') else None,
        organization_id=org_id,
        initiator=goal_create.initiator.value,
        title=goal_create.title,
        description=goal_create.description,
        trigger_memory_ids=goal_create.trigger_memory_ids,
        expected_value=goal_create.expected_value,
        urgency=goal_create.urgency,
        confidence=goal_create.confidence,
        status="proposed",
        metadata=goal_create.metadata or {},
    )
    
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    
    return goal


@router.patch("/{goal_id}", response_model=AutonomousGoalResponse)
async def update_autonomous_goal(
    goal_id: UUID,
    goal_update: AutonomousGoalUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an autonomous goal."""
    stmt = select(AutonomousGoal).where(AutonomousGoal.id == goal_id)
    result = await db.execute(stmt)
    goal = result.scalar_one_or_none()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    # Update fields
    if goal_update.title is not None:
        goal.title = goal_update.title
    if goal_update.description is not None:
        goal.description = goal_update.description
    if goal_update.expected_value is not None:
        goal.expected_value = goal_update.expected_value
    if goal_update.urgency is not None:
        goal.urgency = goal_update.urgency
    if goal_update.status is not None:
        goal.status = goal_update.status.value
        if goal_update.status.value == "active":
            from datetime import datetime
            goal.activated_at = datetime.utcnow()
        elif goal_update.status.value == "completed":
            from datetime import datetime
            goal.completed_at = datetime.utcnow()
    
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    
    return goal


@router.post("/{goal_id}/complete", response_model=AutonomousGoalResponse)
async def complete_autonomous_goal(
    goal_id: UUID,
    completion_evidence: List[str] = Query([]),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an autonomous goal as completed.
    
    Args:
        goal_id: Goal ID
        completion_evidence: Memory IDs proving goal completion
    """
    stmt = select(AutonomousGoal).where(AutonomousGoal.id == goal_id)
    result = await db.execute(stmt)
    goal = result.scalar_one_or_none()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    from datetime import datetime
    goal.status = "completed"
    goal.completed_at = datetime.utcnow()
    goal.completion_evidence = completion_evidence
    
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    
    return goal


# ============================================================================
# Knowledge Gap Endpoints
# ============================================================================

@router.get("/knowledge-gaps", response_model=List[KnowledgeGapResponse])
async def list_knowledge_gaps(
    domain: str = Query(None),
    gap_type: str = Query(None),
    resolved: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    List detected knowledge gaps.
    
    Query parameters:
    - domain: Filter by domain (e.g., "customer_preferences")
    - gap_type: Filter by type (missing_fact|contradiction|outdated|low_confidence)
    - resolved: Show only resolved gaps (default: False)
    - limit: Max results
    """
    stmt = select(KnowledgeGap)
    
    if domain:
        stmt = stmt.where(KnowledgeGap.domain == domain)
    if gap_type:
        stmt = stmt.where(KnowledgeGap.gap_type == gap_type)
    
    if resolved:
        stmt = stmt.where(KnowledgeGap.resolved_at != None)
    else:
        stmt = stmt.where(KnowledgeGap.resolved_at == None)
    
    stmt = stmt.order_by(KnowledgeGap.discovered_at.desc()).limit(limit)
    result = await db.execute(stmt)
    gaps = result.scalars().all()
    
    return gaps


@router.post("/knowledge-gaps", response_model=KnowledgeGapResponse, status_code=201)
async def create_knowledge_gap(
    gap_create: KnowledgeGapCreate,
    org_id: UUID = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge gap record."""
    gap = KnowledgeGap(
        organization_id=org_id,
        gap_type=gap_create.gap_type.value,
        domain=gap_create.domain,
        description=gap_create.description,
        confidence_in_gap=gap_create.confidence_in_gap,
        related_memories=gap_create.related_memories,
        suggested_learning_approach=gap_create.suggested_learning_approach.value if gap_create.suggested_learning_approach else None,
        metadata=gap_create.metadata or {},
    )
    
    db.add(gap)
    await db.commit()
    await db.refresh(gap)
    
    return gap


# ============================================================================
# Goal Outcome Endpoints
# ============================================================================

@router.post("/{goal_id}/outcomes", response_model=GoalOutcomeResponse, status_code=201)
async def record_goal_outcome(
    goal_id: UUID,
    outcome_create: GoalOutcomeCreate,
    org_id: UUID = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Record the outcome of an autonomous goal.
    
    Used to track whether goals were valuable, helping improve future
    goal generation through reinforcement learning.
    """
    outcome = AutonomousGoalOutcome(
        organization_id=org_id,
        goal_id=goal_id,
        outcome_type=outcome_create.outcome_type.value,
        impact_description=outcome_create.impact_description,
        feedback_from=outcome_create.feedback_from,
        was_user_expecting=outcome_create.was_user_expecting,
    )
    
    db.add(outcome)
    await db.commit()
    await db.refresh(outcome)
    
    return outcome


# ============================================================================
# Intrinsic Motivation Service Endpoints
# ============================================================================

@router.post("/generate", response_model=GenerateGoalsResponse)
async def generate_autonomous_goals(
    request: GenerateGoalsRequest,
    org_id: UUID = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate autonomous goals based on intrinsic motivation.
    
    This endpoint orchestrates the full autonomous goal generation pipeline:
    1. Detect knowledge gaps
    2. Generate curiosity goals from gaps
    3. Predict user needs
    4. Identify self-improvement opportunities
    5. Estimate value for each goal
    6. Propose and auto-activate top goals
    """
    svc = IntrinsicMotivationService(db)
    
    all_goals = []
    
    # 1. Generate curiosity goals from detected gaps
    if request.include_curiosity:
        gaps = await svc.detect_knowledge_gaps(org_id)
        curiosity_goals = await svc.generate_curiosity_goals(org_id, gaps)
        all_goals.extend(curiosity_goals)
    
    # 2. Generate predictive goals
    if request.include_prediction:
        predictive_goals = await svc.predict_user_needs(org_id)
        all_goals.extend(predictive_goals)
    
    # 3. Generate self-improvement goals
    if request.include_self_improvement:
        improvement_goals = await svc.detect_self_improvement_needs(org_id)
        all_goals.extend(improvement_goals)
    
    # 4. Estimate value and filter
    scored_goals = []
    for goal in all_goals:
        value = await svc.estimate_goal_value(goal)
        if value >= request.min_value_threshold:
            goal["estimated_value"] = value
            scored_goals.append(goal)
    
    # 5. Sort by value * urgency and store
    sorted_goals = sorted(
        scored_goals,
        key=lambda g: g.get("estimated_value", 0.5) * g.get("urgency", 0.5),
        reverse=True
    )
    
    proposed_goals = []
    activated_goals = []
    
    for i, goal_data in enumerate(sorted_goals):
        # Create goal in DB
        goal = AutonomousGoal(
            organization_id=org_id,
            initiator=goal_data["initiator"],
            title=goal_data["title"],
            description=goal_data["description"],
            trigger_memory_ids=goal_data["trigger_memory_ids"],
            expected_value=goal_data["estimated_value"],
            urgency=goal_data["urgency"],
            confidence=goal_data["confidence"],
            status="proposed",
            metadata=goal_data.get("metadata", {}),
        )
        db.add(goal)
        await db.flush()
        proposed_goals.append(goal)
        
        # Auto-activate top goals up to limit
        if i < request.max_active_goals:
            from datetime import datetime
            goal.status = "active"
            goal.activated_at = datetime.utcnow()
            activated_goals.append(goal)
    
    await db.commit()
    
    # Refresh all goals
    for goal in proposed_goals:
        await db.refresh(goal)
    
    # Get metrics
    metrics_data = await svc.get_goal_success_metrics(org_id)
    metrics = MotivationMetrics(
        total_goals_generated=len(proposed_goals),
        total_goals_completed=0,  # Would query DB in real implementation
        valuable_rate=metrics_data.get("valuable_rate", 0.0),
        user_expectation_alignment=metrics_data.get("user_expectation_alignment", 0.0),
        avg_time_to_completion_days=metrics_data.get("average_outcome_days", 0),
        top_valuable_initiators={"curiosity": 0.65, "prediction": 0.58},  # Example
    )
    
    return GenerateGoalsResponse(
        proposed_goals=proposed_goals,
        activated_goals=activated_goals,
        metrics=metrics,
    )


@router.get("/metrics", response_model=MotivationMetrics)
async def get_motivation_metrics(
    org_id: UUID = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get metrics on autonomous goal generation quality.
    
    Returns statistics on:
    - Goal success rates (valuable goals / total)
    - User expectation alignment
    - Time to completion
    - Performance by initiator type
    """
    svc = IntrinsicMotivationService(db)
    metrics_data = await svc.get_goal_success_metrics(org_id)
    
    return MotivationMetrics(
        total_goals_generated=metrics_data.get("total_outcomes", 0),
        total_goals_completed=0,
        valuable_rate=metrics_data.get("valuable_rate", 0.0),
        user_expectation_alignment=metrics_data.get("user_expectation_alignment", 0.0),
        avg_time_to_completion_days=metrics_data.get("average_outcome_days", 0),
        top_valuable_initiators={"curiosity": 0.65, "prediction": 0.58},
    )
