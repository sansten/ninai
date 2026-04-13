"""
Autonomous Goal Model

Models self-generated goals created by the agent based on intrinsic motivation.
Goals can be triggered by curiosity, prediction, self-improvement, or priority rebalancing.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, String, Float, DateTime, ARRAY, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.models.base import Base


class AutonomousGoal(Base):
    """
    Self-generated goal by agent based on intrinsic motivation.
    
    An autonomous goal is a goal the agent creates for itself, rather than
    receiving from a user or external system. These goals drive proactive,
    self-directed behavior toward general intelligence.
    """
    __tablename__ = "autonomous_goals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Goal origin and type
    initiator = Column(String, nullable=False)  # "curiosity" | "prediction" | "self_improvement" | "priority_rebalance"
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)
    
    # Tracking and reasoning
    trigger_memory_ids = Column(ARRAY(String), nullable=False, default=[])  # Memory IDs that triggered this goal
    expected_value = Column(Float, nullable=False, default=0.5)  # 0-1 estimate of benefit
    urgency = Column(Float, nullable=False, default=0.5)  # 0-1 priority
    confidence = Column(Float, nullable=False, default=0.7)  # 0-1 confidence we should care
    
    # Lifecycle
    status = Column(String, nullable=False)  # "proposed" | "active" | "completed" | "abandoned"
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True, index=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    completion_evidence = Column(JSON, nullable=True)  # List of memory IDs showing goal completion
    
    # Metadata
    meta = Column(JSON, nullable=True, default={})  # Additional context
    
    def __repr__(self) -> str:
        return f"<AutonomousGoal(id={self.id}, initiator={self.initiator}, title={self.title}, status={self.status})>"
