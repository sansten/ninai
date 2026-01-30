"""GoalGraph API schemas (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.base import BaseSchema, TimestampSchema


OwnerType = Literal["user", "team", "department", "organization"]
GoalType = Literal["task", "project", "objective", "policy", "research"]
GoalStatus = Literal["proposed", "active", "blocked", "completed", "abandoned"]
VisibilityScope = Literal["personal", "team", "department", "division", "organization"]

NodeType = Literal["subgoal", "task", "milestone"]
NodeStatus = Literal["todo", "in_progress", "blocked", "done", "cancelled"]

EdgeType = Literal["depends_on", "blocks", "related_to"]
LinkType = Literal["evidence", "progress", "blocker", "reference"]
LinkedBy = Literal["auto", "user", "agent"]


class GoalCreateRequest(BaseSchema):
    title: str
    description: str | None = None

    owner_type: OwnerType = "user"
    owner_id: str | None = None

    goal_type: GoalType = "task"
    status: GoalStatus = "proposed"

    priority: int = Field(0, ge=0, le=5)
    due_at: datetime | None = None

    confidence: float = Field(0.5, ge=0.0, le=1.0)

    visibility_scope: VisibilityScope = "personal"
    scope_id: str | None = None

    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoalUpdateRequest(BaseSchema):
    title: str | None = None
    description: str | None = None

    status: GoalStatus | None = None
    priority: int | None = Field(default=None, ge=0, le=5)
    due_at: datetime | None = None

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    visibility_scope: VisibilityScope | None = None
    scope_id: str | None = None

    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class GoalResponse(TimestampSchema):
    id: str
    organization_id: str
    created_by_user_id: str | None

    owner_type: OwnerType
    owner_id: str | None

    title: str
    description: str | None

    goal_type: GoalType
    status: GoalStatus
    priority: int
    due_at: datetime | None
    completed_at: datetime | None

    confidence: float

    visibility_scope: VisibilityScope
    scope_id: str | None

    tags: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict, validation_alias="extra_metadata")


class GoalNodeCreateRequest(BaseSchema):
    parent_node_id: str | None = None

    node_type: NodeType
    title: str
    description: str | None = None

    status: NodeStatus = "todo"
    priority: int = Field(0, ge=0, le=5)

    assigned_to_user_id: str | None = None
    assigned_to_team_id: str | None = None

    ordering: int = 0

    expected_outputs: dict[str, Any] | None = None
    success_criteria: list[str] | None = None
    blockers: dict[str, Any] | None = None

    confidence: float = Field(0.5, ge=0.0, le=1.0)


class GoalNodeUpdateRequest(BaseSchema):
    title: str | None = None
    description: str | None = None

    status: NodeStatus | None = None
    priority: int | None = Field(default=None, ge=0, le=5)
    ordering: int | None = None

    assigned_to_user_id: str | None = None
    assigned_to_team_id: str | None = None

    expected_outputs: dict[str, Any] | None = None
    success_criteria: list[str] | None = None
    blockers: dict[str, Any] | None = None

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class GoalNodeResponse(TimestampSchema):
    id: str
    organization_id: str
    goal_id: str
    parent_node_id: str | None

    node_type: NodeType
    title: str
    description: str | None

    status: NodeStatus
    priority: int

    assigned_to_user_id: str | None
    assigned_to_team_id: str | None

    completed_at: datetime | None

    confidence: float
    ordering: int

    expected_outputs: dict[str, Any] | None
    success_criteria: list[str] | None
    blockers: dict[str, Any] | None


class GoalEdgeCreateRequest(BaseSchema):
    from_node_id: str
    to_node_id: str
    edge_type: EdgeType


class GoalEdgeResponse(BaseSchema):
    id: str
    organization_id: str
    from_node_id: str
    to_node_id: str
    edge_type: EdgeType
    created_at: datetime


class GoalMemoryLinkCreateRequest(BaseSchema):
    memory_id: str
    link_type: LinkType
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    node_id: str | None = None


class GoalMemoryLinkResponse(BaseSchema):
    id: str
    organization_id: str
    goal_id: str
    node_id: str | None
    memory_id: str
    link_type: LinkType
    linked_by: LinkedBy
    confidence: float
    created_at: datetime


class GoalActivityLogResponse(BaseSchema):
    id: str
    organization_id: str
    goal_id: str
    node_id: str | None
    actor_type: Literal["user", "agent", "system"]
    actor_id: str | None
    action: str
    details: dict[str, Any]
    created_at: datetime


class GoalProgressSummary(BaseModel):
    percent_complete: float = Field(0.0, ge=0.0, le=100.0)
    completed_nodes: int = 0
    total_nodes: int = 0
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class GoalDetailResponse(GoalResponse):
    nodes: list[GoalNodeResponse] = Field(default_factory=list)
    edges: list[GoalEdgeResponse] = Field(default_factory=list)
    memory_links: list[GoalMemoryLinkResponse] = Field(default_factory=list)
    progress: GoalProgressSummary = Field(default_factory=GoalProgressSummary)


class GoalTreeResponse(BaseSchema):
    goal_id: str
    nodes: list[GoalNodeResponse] = Field(default_factory=list)
    edges: list[GoalEdgeResponse] = Field(default_factory=list)


class GoalBlockersResponse(BaseSchema):
    goal_id: str
    blockers: list[GoalNodeResponse] = Field(default_factory=list)
