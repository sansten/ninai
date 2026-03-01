"""
Schemas for Autonomous Goals and Intrinsic Motivation (PR-3)

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class GoalInitiator(str, Enum):
    """Source of goal generation"""
    CURIOSITY = "curiosity"
    PREDICTION = "prediction"
    SELF_IMPROVEMENT = "self_improvement"
    PRIORITY_REBALANCE = "priority_rebalance"


class GoalStatus(str, Enum):
    """Goal lifecycle status"""
    PROPOSED = "proposed"
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class KnowledgeGapType(str, Enum):
    """Type of knowledge gap"""
    MISSING_FACT = "missing_fact"
    CONTRADICTION = "contradiction"
    OUTDATED = "outdated"
    LOW_CONFIDENCE = "low_confidence"


class LearningApproach(str, Enum):
    """Suggested approach to fill gap"""
    SEARCH = "search"
    ASK_USER = "ask_user"
    EXPERIMENT = "experiment"
    OBSERVATION = "observation"


# ============================================================================
# Autonomous Goal Schemas
# ============================================================================

class AutonomousGoalBase(BaseModel):
    """Base autonomous goal schema"""
    initiator: GoalInitiator
    title: str
    description: str
    trigger_memory_ids: List[str] = []
    expected_value: float = Field(0.5, ge=0.0, le=1.0)
    urgency: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.7, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = {}


class AutonomousGoalCreate(AutonomousGoalBase):
    """Create autonomous goal"""
    pass


class AutonomousGoalUpdate(BaseModel):
    """Update autonomous goal"""
    title: Optional[str] = None
    description: Optional[str] = None
    expected_value: Optional[float] = None
    urgency: Optional[float] = None
    status: Optional[GoalStatus] = None


class AutonomousGoalResponse(AutonomousGoalBase):
    """Autonomous goal response from API"""
    id: UUID
    status: GoalStatus
    created_at: datetime
    activated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completion_evidence: Optional[List[str]] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Knowledge Gap Schemas
# ============================================================================

class KnowledgeGapBase(BaseModel):
    """Base knowledge gap schema"""
    gap_type: KnowledgeGapType
    domain: str
    description: str
    confidence_in_gap: float = Field(0.7, ge=0.0, le=1.0)
    related_memories: List[str] = []
    suggested_learning_approach: Optional[LearningApproach] = None
    metadata: Optional[Dict[str, Any]] = {}


class KnowledgeGapCreate(KnowledgeGapBase):
    """Create knowledge gap"""
    pass


class KnowledgeGapUpdate(BaseModel):
    """Update knowledge gap"""
    description: Optional[str] = None
    confidence_in_gap: Optional[float] = None
    suggested_learning_approach: Optional[LearningApproach] = None


class KnowledgeGapResponse(KnowledgeGapBase):
    """Knowledge gap response from API"""
    id: UUID
    discovered_at: datetime
    resolved_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Goal Outcome Schemas
# ============================================================================

class OutcomeType(str, Enum):
    """Outcome assessment"""
    VALUABLE = "valuable"
    NOT_VALUABLE = "not_valuable"
    PREMATURE = "premature"


class GoalOutcomeBase(BaseModel):
    """Base goal outcome schema"""
    outcome_type: OutcomeType
    impact_description: Optional[str] = None
    feedback_from: Optional[str] = "auto_detection"
    was_user_expecting: bool = False


class GoalOutcomeCreate(GoalOutcomeBase):
    """Create goal outcome"""
    goal_id: UUID


class GoalOutcomeResponse(GoalOutcomeBase):
    """Goal outcome response from API"""
    id: UUID
    goal_id: UUID
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================================================
# Intrinsic Motivation Schemas
# ============================================================================

class GoalValueEstimate(BaseModel):
    """Estimated value of a goal"""
    goal_id: UUID
    estimated_value: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    effort: float = Field(ge=0.0, le=1.0)
    rationale: str


class MotivationMetrics(BaseModel):
    """Metrics on autonomous goal generation quality"""
    total_goals_generated: int
    total_goals_completed: int
    valuable_rate: float
    user_expectation_alignment: float
    avg_time_to_completion_days: float
    top_valuable_initiators: Dict[str, float]  # e.g., {"curiosity": 0.75, "prediction": 0.62}


class GenerateGoalsRequest(BaseModel):
    """Request to generate autonomous goals"""
    include_curiosity: bool = True
    include_prediction: bool = True
    include_self_improvement: bool = True
    min_value_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_active_goals: int = Field(10, ge=1, le=50)


class GenerateGoalsResponse(BaseModel):
    """Response from goal generation"""
    proposed_goals: List[AutonomousGoalResponse]
    activated_goals: List[AutonomousGoalResponse]
    metrics: MotivationMetrics
