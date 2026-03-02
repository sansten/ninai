"""
Tool Capability Model - PR-4: Tool Capability Learning & Adaptive Strategy Selection

Tracks learned performance metrics for tools across different goal types,
enabling the agent to adapt its tool selection and strategy over time.
"""

from typing import Optional, List, Dict
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, Index, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from .base import Base
import uuid
import enum


class ToolType(str, enum.Enum):
    """Types of tools the agent can use."""
    DATA_ANALYSIS = "data_analysis"
    API_CALL = "api_call"
    CODE_EXECUTION = "code_execution"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    WEB_SEARCH = "web_search"
    MEMORY_QUERY = "memory_query"
    SYSTEM_COMMAND = "system_command"


class ToolCapability(Base):
    """
    Model for tracking learned tool performance metrics.
    
    Enables the agent to learn which tools are most effective
    for different goal types and task scenarios.
    """
    __tablename__ = "tool_capabilities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Tool identification
    tool_name = Column(String(255), nullable=False, index=True)  # e.g., "python_executor", "gpt4_model"
    tool_type = Column(Enum(ToolType), nullable=False, index=True)
    description = Column(String(1000), nullable=True)
    
    # Performance metrics
    success_rate = Column(Float, nullable=False, default=0.0)  # 0-1 success ratio
    avg_execution_time = Column(Float, nullable=False, default=0.0)  # seconds
    avg_cost = Column(Float, nullable=False, default=0.0)  # normalized cost 0-1
    reliability_score = Column(Float, nullable=False, default=0.5)  # 0-1 based on consistency
    
    # Usage tracking
    total_uses = Column(Integer, nullable=False, default=0)
    successful_uses = Column(Integer, nullable=False, default=0)
    failed_uses = Column(Integer, nullable=False, default=0)
    last_used = Column(DateTime, nullable=True)
    
    # Capability metadata
    supported_goal_types = Column(JSON, nullable=True, default=[])  # ["curiosity", "prediction", "self_improvement"]
    supported_domains = Column(JSON, nullable=True, default=[])  # ["customer_data", "analytics", "reports"]
    known_limitations = Column(JSON, nullable=True, default={})  # {"max_tokens": 8000, "requires_api_key": true}
    meta = Column(JSON, nullable=True, default={})  # Additional context
    
    # Timestamps
    discovered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_evaluated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index("idx_tool_capability_org_type", organization_id, tool_type),
        Index("idx_tool_capability_reliability", organization_id, reliability_score),
        Index("idx_tool_capability_org_name", organization_id, tool_name),
    )
    
    def __repr__(self) -> str:
        return f"<ToolCapability(id={self.id}, tool_name={self.tool_name}, success_rate={self.success_rate:.2%})>"


class StrategyAdaptation(Base):
    """
    Model for tracking strategy adaptations made by the agent.
    
    Records when the agent switched from one tool/strategy to another,
    why it made the change, and the outcome of the new strategy.
    """
    __tablename__ = "strategy_adaptations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Goal information
    goal_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # FK to AutonomousGoal
    goal_type = Column(String(100), nullable=False)  # "curiosity", "prediction", etc.
    
    # Strategy details
    previous_strategy = Column(String(255), nullable=True)  # Previous tool/approach
    new_strategy = Column(String(255), nullable=False)  # Newly selected tool/approach
    adaptation_reason = Column(String(500), nullable=False)  # Why the change was made
    
    # Performance comparison
    previous_success_rate = Column(Float, nullable=True)
    new_success_rate = Column(Float, nullable=True)
    predicted_improvement = Column(Float, nullable=True)  # Expected % improvement
    actual_improvement = Column(Float, nullable=True)  # Measured % improvement
    
    # Metadata
    triggered_by = Column(String(100), nullable=False)  # "performance", "capability_discovery", "learning_loop"
    confidence_in_adaptation = Column(Float, nullable=False, default=0.5)  # 0-1
    meta = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    evaluated_at = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_adaptation_org_goal", organization_id, goal_id),
        Index("idx_adaptation_trigger", organization_id, triggered_by),
    )
    
    def __repr__(self) -> str:
        return f"<StrategyAdaptation(id={self.id}, goal={self.goal_id}, from={self.previous_strategy} to={self.new_strategy})>"


class CapabilityDiscovery(Base):
    """
    Model for tracking new tool/capability discoveries.
    
    Records when the agent discovers new tools, learns about their capabilities,
    or identifies unexpected capability gaps.
    """
    __tablename__ = "capability_discoveries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Discovery details
    discovery_type = Column(String(100), nullable=False)  # "new_tool", "capability_gap", "synergy", "limitation"
    tool_or_capability = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=False)
    
    # Impact assessment
    potential_value = Column(Float, nullable=False, default=0.5)  # 0-1 estimated value
    required_investigation = Column(String(1000), nullable=True)  # What needs to be learned
    can_exploit_now = Column(JSON, nullable=True, default=[])  # Goals that can use this now
    
    # Metadata
    discovered_when = Column(String(100), nullable=False)  # "goal_execution", "gap_analysis", "planning"
    validation_status = Column(String(100), nullable=False, default="unvalidated")  # "unvalidated", "testing", "confirmed"
    meta = Column(JSON, nullable=True, default={})
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    last_investigated = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_discovery_org_type", organization_id, discovery_type),
        Index("idx_discovery_status", organization_id, validation_status),
    )
    
    def __repr__(self) -> str:
        return f"<CapabilityDiscovery(id={self.id}, type={self.discovery_type}, tool={self.tool_or_capability})>"
