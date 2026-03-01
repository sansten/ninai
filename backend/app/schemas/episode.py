"""
Episode Schemas (PR1: Advanced Memory Features)
================================================

Request and response schemas for episode case continuity.
Episodes enable tracking of support cases, research threads, legal cases, etc.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import Field, field_validator

from app.schemas.base import BaseSchema


# ═══════════════════════════════════════════════════════════════════
# Episode CRUD Schemas
# ═══════════════════════════════════════════════════════════════════


class EpisodeCreate(BaseSchema):
    """Request schema for creating an episode."""

    scope_type: str = Field(
        "personal",
        description="Visibility scope: personal, team, department, division, organization",
    )
    scope_id: Optional[str] = Field(
        None,
        description="ID of team/department/division if scoped",
    )
    owner_user_id: Optional[str] = Field(
        None,
        description="User who owns/initiated this episode",
    )
    episode_type: Optional[str] = Field(
        None,
        max_length=100,
        description="Type: support_case, research_thread, legal_case, customer_incident, etc.",
    )
    title: Optional[str] = Field(
        None,
        max_length=500,
        description="Episode title/subject",
    )
    tags: Optional[list[str]] = Field(
        None,
        description="Tags for categorization/filtering",
    )
    entities: Optional[dict[str, Any]] = Field(
        None,
        description="Extracted entities (customer_id, device_id, account_number, etc.)",
    )


class EpisodeUpdate(BaseSchema):
    """Request schema for updating an episode."""

    scope_type: Optional[str] = None
    scope_id: Optional[str] = None
    owner_user_id: Optional[str] = None
    episode_type: Optional[str] = None
    status: Optional[str] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[list[str]] = None
    entities: Optional[dict[str, Any]] = None
    resolved_at: Optional[datetime] = None


class EpisodeResponse(BaseSchema):
    """Response schema for a single episode."""

    id: str
    organization_id: str
    scope_type: str
    scope_id: Optional[str]
    owner_user_id: Optional[str]
    episode_type: Optional[str]
    status: str
    title: Optional[str]
    summary: Optional[str]
    started_at: datetime
    last_event_at: datetime
    resolved_at: Optional[datetime]
    tags: Optional[list[str]]
    entities: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EpisodeListResponse(BaseSchema):
    """Response schema for listing episodes."""

    episodes: list[EpisodeResponse]
    total: int
    page: int
    page_size: int


# ═══════════════════════════════════════════════════════════════════
# Episode Event CRUD Schemas
# ═══════════════════════════════════════════════════════════════════


class EpisodeEventCreate(BaseSchema):
    """Request schema for creating an episode event."""

    episode_id: str = Field(
        ...,
        description="Parent episode ID",
    )
    memory_id: Optional[str] = Field(
        None,
        description="Associated memory ID (if event originated from memory write)",
    )
    event_type: str = Field(
        ...,
        description="Type: user_report, agent_action, tool_result, resolution, followup, note",
    )
    event_ts: Optional[datetime] = Field(
        None,
        description="When the event occurred (defaults to now)",
    )
    actor_type: str = Field(
        "user",
        description="Who/what created this event: user, agent, system",
    )
    actor_id: Optional[str] = Field(
        None,
        description="User/agent ID if applicable",
    )
    content: Optional[str] = Field(
        None,
        description="Human-readable event description/message",
    )
    payload: Optional[dict[str, Any]] = Field(
        None,
        description="Structured event data (tool results, metadata, etc.)",
    )


class EpisodeEventResponse(BaseSchema):
    """Response schema for a single episode event."""

    id: str
    organization_id: str
    episode_id: str
    memory_id: Optional[str]
    event_type: str
    event_ts: datetime
    actor_type: str
    actor_id: Optional[str]
    content: Optional[str]
    payload: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EpisodeEventListResponse(BaseSchema):
    """Response schema for listing episode events."""

    events: list[EpisodeEventResponse]
    total: int


# ═══════════════════════════════════════════════════════════════════
# Episode Link CRUD Schemas
# ═══════════════════════════════════════════════════════════════════


class EpisodeLinkCreate(BaseSchema):
    """Request schema for creating an episode link."""

    from_episode_id: str = Field(
        ...,
        description="Source episode ID",
    )
    to_episode_id: str = Field(
        ...,
        description="Target episode ID",
    )
    relation: str = Field(
        ...,
        description="Type: duplicate, causal_hypothesis, follow_on, same_account, same_device",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0, 1.0]",
    )
    evidence: Optional[dict[str, Any]] = Field(
        None,
        description="Evidence supporting the relationship",
    )


class EpisodeLinkResponse(BaseSchema):
    """Response schema for a single episode link."""

    id: str
    organization_id: str
    from_episode_id: str
    to_episode_id: str
    relation: str
    confidence: Optional[float]
    evidence: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class EpisodeLinkListResponse(BaseSchema):
    """Response schema for listing episode links."""

    links: list[EpisodeLinkResponse]
    total: int


# ═══════════════════════════════════════════════════════════════════
# Enrichment Response Schemas (for Memory API)
# ═══════════════════════════════════════════════════════════════════


class EpisodeContextResponse(BaseSchema):
    """Episode context for memory enrichment (when include_episode=true)."""

    episode_id: str
    title: Optional[str]
    status: str
    summary: Optional[str]
    timeline: list[EpisodeEventResponse]
    open_actions: list[dict[str, Any]]


class EpisodeFilterQuery(BaseSchema):
    """Query parameters for filtering episodes."""

    status: Optional[str] = None
    episode_type: Optional[str] = None
    owner_user_id: Optional[str] = None
    tags: Optional[list[str]] = None
    started_after: Optional[datetime] = None
    started_before: Optional[datetime] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
