"""GoalGraph agent-only schemas (Pydantic v2).

These models validate deterministic JSON-only agent outputs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.base import BaseSchema


GoalType = Literal["task", "project", "objective", "policy", "research"]
VisibilityScope = Literal["personal", "team", "department", "division", "organization"]
NodeType = Literal["subgoal", "task", "milestone"]
EdgeType = Literal["depends_on", "blocks", "related_to"]


class GoalPlannerGoal(BaseSchema):
    title: str
    description: str | None = None
    goal_type: GoalType
    visibility_scope: VisibilityScope
    scope_id: str | None = None
    priority: int = Field(0, ge=0, le=5)
    due_at: datetime | None = None


class GoalPlannerNode(BaseSchema):
    temp_id: str
    parent_temp_id: str | None = None
    node_type: NodeType
    title: str
    description: str | None = None
    success_criteria: list[str] = Field(default_factory=list)
    expected_outputs: dict[str, Any] = Field(default_factory=dict)


class GoalPlannerEdge(BaseSchema):
    from_temp_id: str
    to_temp_id: str
    edge_type: EdgeType


class GoalPlannerAgentOutput(BaseSchema):
    create_goal: bool = False
    goal: GoalPlannerGoal | None = None
    nodes: list[GoalPlannerNode] = Field(default_factory=list)
    edges: list[GoalPlannerEdge] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence_memory_ids: list[str] = Field(default_factory=list)


LinkType = Literal["evidence", "progress", "blocker", "reference"]


class GoalLinkSuggestion(BaseSchema):
    goal_id: str
    node_id: str | None = None
    memory_id: str
    link_type: LinkType
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reason: str


class GoalLinkingAgentOutput(BaseSchema):
    links: list[GoalLinkSuggestion] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
