"""GoalGraph agent endpoints schemas.

These schemas define request bodies for GoalPlannerAgent/GoalLinkingAgent endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema


class GoalProposeRequest(BaseSchema):
    user_request: str = Field(..., min_length=1, max_length=10_000)
    session_context: dict[str, Any] = Field(default_factory=dict)
    existing_goals: list[dict[str, Any]] = Field(default_factory=list)


class GoalLinkSuggestionsRequest(BaseSchema):
    memory: dict[str, Any] = Field(default_factory=dict)
    active_goals: list[dict[str, Any]] = Field(default_factory=list)
