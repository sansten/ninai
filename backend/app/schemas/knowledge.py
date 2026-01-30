"""Knowledge base / HITL review schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from app.schemas.base import BaseSchema
from app.schemas.provenance import ProvenanceSource


class KnowledgeSubmissionCreate(BaseSchema):
    """Create a knowledge submission for review."""

    item_id: Optional[str] = Field(
        None,
        description="If set, submits a new version for an existing item. If omitted, a new item is created.",
    )
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=200000)
    key: Optional[str] = Field(
        None,
        max_length=255,
        description="Optional stable key for the item (unique per org). Only used when creating a new item.",
    )
    extra_metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: list[ProvenanceSource] = Field(default_factory=list)


class KnowledgeReviewRequestResponse(BaseSchema):
    id: str
    organization_id: str
    item_id: str
    item_version_id: str
    status: str
    requested_by_user_id: Optional[str]
    reviewed_by_user_id: Optional[str]
    reviewed_at: Optional[datetime]
    decision_comment: Optional[str]
    created_at: datetime
    updated_at: datetime


class KnowledgeReviewListResponse(BaseSchema):
    items: list[KnowledgeReviewRequestResponse]


class KnowledgeReviewDecision(BaseSchema):
    comment: Optional[str] = Field(None, max_length=20000)

    # Optional: promote approved knowledge into long-term memory.
    promote_to_memory: bool = Field(
        default=False,
        description="If true, approval will also create a long-term memory record from the version content.",
    )

    # Mapping helpers (used when promote_to_memory=true)
    tags: list[str] = Field(default_factory=list, description="Tags to attach to the created memory")
    topics: list[str] = Field(default_factory=list, description="Topics (labels) to attach via topic memberships")
    primary_topic: Optional[str] = Field(
        None,
        description="Primary topic label (defaults to first topic if omitted)",
        max_length=200,
    )
    topic_confidence: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Confidence/weight applied to topic memberships",
    )

    memory_scope: str = Field(
        default="organization",
        description="Scope for the promoted memory (personal/team/department/division/organization/global)",
    )
    memory_type: str = Field(
        default="procedural",
        description="Memory type for the promoted memory (long_term/semantic/procedural)",
    )
    classification: str = Field(
        default="internal",
        description="Classification for the promoted memory (public/internal/confidential/restricted)",
    )


class KnowledgeItemResponse(BaseSchema):
    id: str
    organization_id: str
    title: str
    key: Optional[str]
    is_published: bool
    published_version_id: Optional[str]
    published_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class KnowledgeItemVersionResponse(BaseSchema):
    id: str
    organization_id: str
    item_id: str
    version_number: int
    title: Optional[str]
    content: str
    extra_metadata: dict
    created_by_user_id: Optional[str]
    trace_id: Optional[str]
    provenance: list[dict]
    created_at: datetime
    updated_at: datetime


class KnowledgeItemVersionsResponse(BaseSchema):
    items: list[KnowledgeItemVersionResponse]


class KnowledgeRollbackRequest(BaseSchema):
    target_version_id: str
    comment: Optional[str] = Field(None, max_length=20000)
