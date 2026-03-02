"""
Pydantic Schemas for PR-4: Tool Capability Learning & Adaptive Strategy Selection
"""

from typing import Optional, List, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class ToolCapabilityCreate(BaseModel):
    """Request to register a new tool capability."""
    tool_name: str = Field(..., min_length=1, max_length=255)
    tool_type: str  # "data_analysis", "api_call", etc.
    description: Optional[str] = None
    supported_goal_types: Optional[List[str]] = None
    supported_domains: Optional[List[str]] = None
    known_limitations: Optional[Dict] = None


class ToolCapabilityResponse(BaseModel):
    """Response with tool capability details."""
    id: str
    tool_name: str
    tool_type: str
    description: Optional[str]
    success_rate: float
    reliability_score: float
    total_uses: int
    avg_execution_time: float
    supported_goal_types: Optional[List[str]]
    supported_domains: Optional[List[str]]
    known_limitations: Optional[Dict]
    last_used: Optional[datetime]


class ToolUsageRecord(BaseModel):
    """Request to record tool usage."""
    tool_name: str
    goal_id: str
    success: bool
    execution_time: float = 0.0
    cost: float = 0.0


class ToolUsageResponse(BaseModel):
    """Response after recording tool usage."""
    tool_name: str
    success_rate: float
    reliability_score: float
    total_uses: int


class ToolRecommendationRequest(BaseModel):
    """Request for tool recommendation."""
    goal_type: str
    domain: Optional[str] = None
    exclude_tools: Optional[List[str]] = None


class ToolRecommendationResponse(BaseModel):
    """Tool recommendation response."""
    recommended_tool: str
    reasoning: str
    details: Dict


class StrategyAdaptationRecord(BaseModel):
    """Request to record strategy adaptation."""
    goal_id: str
    goal_type: str
    previous_strategy: Optional[str] = None
    new_strategy: str
    reason: str
    confidence: float = 0.5


class StrategyAdaptationResponse(BaseModel):
    """Response after recording adaptation."""
    id: str
    goal_id: str
    from_strategy: Optional[str]
    to_strategy: str


class CapabilityDiscoveryRecord(BaseModel):
    """Request to record capability discovery."""
    discovery_type: str  # "new_tool", "capability_gap", "synergy", "limitation"
    tool_or_capability: str
    description: str
    potential_value: float = 0.5


class CapabilityDiscoveryResponse(BaseModel):
    """Response after recording discovery."""
    id: str
    discovery_type: str
    tool_or_capability: str
    potential_value: float


class StrategyHistoryEntry(BaseModel):
    """Single entry in strategy history."""
    id: str
    from_strategy: Optional[str]
    to_strategy: str
    reason: str
    created_at: str


class ToolRecommendationsListResponse(BaseModel):
    """List of tool recommendations."""
    tool_name: str
    tool_type: str
    score: float
    success_rate: float
    total_uses: int
    supported_goal_types: Optional[List[str]]
    known_limitations: Optional[Dict]


class CapabilityDiscoveryResponse2(BaseModel):
    """Discovery item in backlog."""
    id: str
    type: str
    tool_or_capability: str
    description: str
    potential_value: float
    status: str
