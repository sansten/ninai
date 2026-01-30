"""
Memory Schemas
==============

Request and response schemas for memory operations.
"""

from typing import Optional, List, Any
from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator

from app.schemas.base import BaseSchema
from app.schemas.provenance import ProvenanceSource


class MemoryScope(str, Enum):
    """Memory visibility scope."""
    PERSONAL = "personal"
    TEAM = "team"
    DEPARTMENT = "department"
    DIVISION = "division"
    ORGANIZATION = "organization"
    GLOBAL = "global"


class MemoryType(str, Enum):
    """Memory type classification."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class SearchHnmsMode(str, Enum):
    """HNMS-inspired ranking mode selector for search."""

    BALANCED = "balanced"
    PERFORMANCE = "performance"
    RESEARCH = "research"


class Classification(str, Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class SharePermission(str, Enum):
    """Permission levels for shared access."""
    READ = "read"
    COMMENT = "comment"
    EDIT = "edit"


class ShareType(str, Enum):
    """Types of share targets."""
    USER = "user"
    TEAM = "team"
    DEPARTMENT = "department"
    DIVISION = "division"
    ORGANIZATION = "organization"
    EXTERNAL = "external"


# =============================================================================
# Memory CRUD Schemas
# =============================================================================

class MemoryCreate(BaseSchema):
    """Request schema for creating a memory."""
    
    content: str = Field(..., min_length=1, max_length=100000)
    title: Optional[str] = Field(None, max_length=500)
    scope: MemoryScope = MemoryScope.PERSONAL
    scope_id: Optional[str] = None  # Required for team/dept/etc scopes
    memory_type: MemoryType = MemoryType.LONG_TERM
    classification: Classification = Classification.INTERNAL
    required_clearance: Optional[int] = Field(None, ge=0, le=4)
    tags: Optional[List[str]] = None
    entities: Optional[dict] = None
    extra_metadata: Optional[dict] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    retention_days: Optional[int] = Field(None, ge=1)
    ttl: Optional[int] = Field(None, ge=1, description="Short-term memory TTL in seconds (overrides default if set)")
    
    @field_validator("scope_id")
    @classmethod
    def validate_scope_id(cls, v, info):
        """Require scope_id for non-personal scopes."""
        scope = info.data.get("scope")
        if scope and scope != MemoryScope.PERSONAL and not v:
            raise ValueError(f"scope_id is required for scope '{scope}'")
        return v


class MemoryUpdate(BaseSchema):
    """Request schema for updating a memory."""
    
    title: Optional[str] = Field(None, max_length=500)
    tags: Optional[List[str]] = None
    classification: Optional[Classification] = None
    extra_metadata: Optional[dict] = None
    retention_days: Optional[int] = Field(None, ge=1)


class MemoryResponse(BaseSchema):
    """Response schema for memory data."""
    
    id: str
    organization_id: str
    owner_id: str
    scope: str
    scope_id: Optional[str]
    memory_type: str
    classification: str
    required_clearance: int
    title: Optional[str]
    content_preview: str
    tags: List[str]
    entities: dict
    extra_metadata: dict
    source_type: Optional[str]
    source_id: Optional[str]
    access_count: int
    last_accessed_at: Optional[datetime]
    is_promoted: bool
    created_at: datetime
    updated_at: datetime
    
    # Search result score (only present in search results)
    score: Optional[float] = None

    # Provenance/citations (optional; populated for search/RAG-style responses)
    provenance: Optional[List[ProvenanceSource]] = None


# =============================================================================
# Search Schemas
# =============================================================================

class MemorySearchRequest(BaseSchema):
    """Request schema for memory search."""
    
    query: str = Field(..., min_length=1, max_length=1000)
    scope: Optional[MemoryScope] = None
    team_id: Optional[str] = None
    classification_max: Optional[Classification] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(10, ge=1, le=100)
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    hybrid: bool = Field(False, description="Enable hybrid search (lexical + vector)")
    hnms_mode: Optional[SearchHnmsMode] = Field(
        None,
        description="Ranking mode override: balanced|performance|research",
    )


class MemorySearchResponse(BaseSchema):
    """Response schema for memory search."""
    
    trace_id: Optional[str] = None
    query: str
    results: List[MemoryResponse]
    total: int
    took_ms: float
    ranking_meta: Optional[dict[str, Any]] = None


# =============================================================================
# List Schemas
# =============================================================================

class MemoryListResponse(BaseSchema):
    """Response schema for memory list."""

    items: List[MemoryResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


# =============================================================================
# Sharing Schemas
# =============================================================================

class MemoryShareRequest(BaseSchema):
    """Request schema for sharing a memory."""
    
    share_type: ShareType
    target_id: str
    permission: SharePermission = SharePermission.READ
    expires_at: Optional[datetime] = None
    reason: Optional[str] = Field(None, max_length=500)
    notify: bool = True


class MemorySharingResponse(BaseSchema):
    """Response schema for memory sharing data."""
    
    id: str
    memory_id: str
    share_type: str
    target_id: str
    permission: str
    expires_at: Optional[datetime]
    shared_by: str
    share_reason: Optional[str]
    is_active: bool
    created_at: datetime


class RevokeShareRequest(BaseSchema):
    """Request to revoke a share."""
    
    share_id: str
    reason: Optional[str] = None


# =============================================================================
# Access Explanation
# =============================================================================

class AccessDecisionDetail(BaseSchema):
    """Detail of a single access decision."""
    
    allowed: bool
    reason: str
    method: str
    details: dict = {}


class AccessExplanation(BaseSchema):
    """Full explanation of access to a memory."""
    
    memory_id: str
    user_id: str
    organization_id: str
    clearance_level: int
    access: dict  # read, write, share, delete -> AccessDecisionDetail
    user_roles: List[dict]
    checked_at: datetime
