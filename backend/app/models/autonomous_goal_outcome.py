"""
Autonomous Goal Outcome Model

Tracks whether autonomous goals were actually valuable to the agent.
Used for reinforcement learning and continuous improvement of goal generation quality.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.database import Base


class AutonomousGoalOutcome(Base):
    """
    Post-hoc evaluation of an autonomous goal.
    
    After an autonomous goal is completed or abandoned, this record tracks
    whether it was actually valuable. Outcomes are determined by:
    - User feedback (e.g., "this was useful")
    - Metrics analysis (did it improve performance?)
    - Manual review
    
    Outcomes feed back into the IntrinsicMotivationService to improve
    future goal generation quality.
    """
    __tablename__ = "autonomous_goal_outcomes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    goal_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Outcome assessment
    outcome_type = Column(String, nullable=False)  # "valuable" | "not_valuable" | "premature"
    impact_description = Column(String, nullable=True)  # e.g., "Led to 3 successful customer interactions"
    
    # Source of feedback
    feedback_from = Column(String, nullable=True)  # "user" | "metrics" | "manual_review" | "auto_detection"
    was_user_expecting = Column(Boolean, nullable=False, default=False)  # Did user also want this?
    
    # Metadata
    meta = Column(JSON, nullable=True, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    def __repr__(self) -> str:
        return f"<AutonomousGoalOutcome(id={self.id}, goal_id={self.goal_id}, outcome_type={self.outcome_type})>"
