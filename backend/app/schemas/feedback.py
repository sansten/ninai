from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from app.schemas.base import BaseSchema


FeedbackType = Literal[
    "tag_add",
    "tag_remove",
    "classification_override",
    "entity_add",
    "entity_remove",
    "note",
    "relevance",
]


class MemoryFeedbackCreate(BaseSchema):
    feedback_type: FeedbackType
    payload: dict[str, Any] = {}
    target_agent: Optional[str] = None


class MemoryFeedbackResponse(BaseSchema):
    id: str
    organization_id: str
    memory_id: str
    actor_id: str

    feedback_type: str
    target_agent: Optional[str] = None
    payload: dict[str, Any]

    is_applied: bool
    applied_at: Optional[datetime] = None
    applied_by: Optional[str] = None

    created_at: datetime
    updated_at: datetime


class MemoryFeedbackListResponse(BaseSchema):
    items: list[MemoryFeedbackResponse]
    total: int
