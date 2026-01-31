"""
Ninai SDK Data Models
=====================

Pydantic models for API responses.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, model_validator


class GoalPlannerGoal(BaseModel):
    title: str
    description: str | None = None
    goal_type: str | None = None
    visibility_scope: str | None = None
    scope_id: str | None = None
    priority: int = 0
    due_at: datetime | None = None


class GoalPlannerNode(BaseModel):
    temp_id: str
    parent_temp_id: str | None = None
    node_type: str
    title: str
    description: str | None = None
    success_criteria: List[str] = Field(default_factory=list)
    expected_outputs: Dict[str, Any] = Field(default_factory=dict)


class GoalPlannerEdge(BaseModel):
    from_temp_id: str
    to_temp_id: str
    edge_type: str


class GoalPlannerAgentOutput(BaseModel):
    create_goal: bool = False
    goal: GoalPlannerGoal | None = None
    nodes: List[GoalPlannerNode] = Field(default_factory=list)
    edges: List[GoalPlannerEdge] = Field(default_factory=list)
    confidence: float = 0.0


class GoalLinkSuggestion(BaseModel):
    goal_id: str
    node_id: str | None = None
    memory_id: str
    link_type: str
    confidence: float = 0.0
    reason: str | None = None


class GoalLinkingAgentOutput(BaseModel):
    links: List[GoalLinkSuggestion] = Field(default_factory=list)
    confidence: float = 0.0


class User(BaseModel):
    """Authenticated user information."""
    id: str
    email: str
    full_name: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    clearance_level: int = 0
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    roles: List[str] = []


class AuthTokens(BaseModel):
    """Authentication tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: User


class Memory(BaseModel):
    """Memory object."""
    id: str
    organization_id: str
    owner_id: str
    scope: str
    scope_id: Optional[str] = None
    memory_type: str
    classification: str
    required_clearance: int = 0
    title: Optional[str] = None
    content_preview: str
    tags: List[str] = []
    entities: Dict[str, Any] = {}
    extra_metadata: Dict[str, Any] = {}
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    is_promoted: bool = False
    created_at: datetime
    updated_at: datetime
    score: Optional[float] = None  # Present in search results


class MemoryList(BaseModel):
    """Paginated list of memories."""
    items: List[Memory]
    total: int
    page: int = 1
    page_size: int = 20
    has_more: bool = False


class SearchResult(BaseModel):
    """Search results with memories and metadata."""
    items: List[Memory]
    total: int
    query: str
    took_ms: Optional[float] = None


class Organization(BaseModel):
    """Organization information."""
    id: str
    name: str
    slug: str
    tier: str = "free"
    settings: Dict[str, Any] = {}
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class Team(BaseModel):
    """Team information."""
    id: str
    organization_id: str
    name: str
    description: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


class SelfModelProfile(BaseModel):
    organization_id: str
    domain_confidence: Dict[str, Any] = {}
    tool_reliability: Dict[str, Any] = {}
    agent_accuracy: Dict[str, Any] = {}
    last_updated: datetime


class SelfModelPlannerSummary(BaseModel):
    unreliable_tools: List[str] = []
    low_confidence_domains: List[str] = []
    recommended_evidence_multiplier: int = 1


class SelfModelBundle(BaseModel):
    profile: SelfModelProfile
    planner_summary: SelfModelPlannerSummary


class ToolSpec(BaseModel):
    name: str
    version: str
    required_permissions: List[str] = []
    allowed_scopes: List[str] | None = None
    require_justification: bool = False
    min_clearance_level: int = 0
    sensitivity: Dict[str, Any] = {}


class ToolInvocationResult(BaseModel):
    status: str
    success: bool = False
    tool_name: str
    tool_call_id: str | None = None
    output: Dict[str, Any] | None = None
    denial_reason: str | None = None
    error: str | None = None
    warnings: List[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_success(cls, data: Any):
        if isinstance(data, dict) and "success" not in data:
            data = dict(data)
            data["success"] = data.get("status") == "success"
        return data


class LLMCompleteJsonResponse(BaseModel):
    data: Dict[str, Any] = {}
