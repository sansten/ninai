"""
Memory Models
=============

Models for memory metadata and sharing.
Actual embeddings are stored in Qdrant; this stores structured metadata.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import (
    String, Text, Boolean, Integer, Float,
    ForeignKey, Index, CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, TSVECTOR

from app.models.base import Base, UUIDMixin, TimestampMixin


class MemoryMetadata(Base, UUIDMixin, TimestampMixin):
    """
    Memory metadata model.
    
    Stores structured metadata for memories. The actual embedding
    vector and full content are stored in Qdrant for efficient
    similarity search.
    
    SECURITY: RLS policies filter by organization_id and scope.
    Qdrant searches must be re-verified against this table.
    
    Attributes:
        organization_id: Organization (tenant isolation)
        owner_id: User who created the memory
        scope: Visibility level (personal/team/dept/div/org)
        classification: Security classification
        content_preview: Short preview of content
        tags: Searchable tags
        entities: Extracted entities (people, places, etc.)
    """
    
    __tablename__ = "memory_metadata"
    
    # Organization (tenant isolation)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this memory belongs to",
    )
    
    # Ownership
    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
        doc="User who created this memory",
    )
    
    # Scope (visibility)
    scope: Mapped[str] = mapped_column(
        String(50),
        default="personal",
        nullable=False,
        doc="Scope: personal, team, department, division, organization, global",
    )
    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Scope entity ID (e.g., team_id if scope is 'team')",
    )
    
    # Memory type
    memory_type: Mapped[str] = mapped_column(
        String(50),
        default="long_term",
        nullable=False,
        doc="Type: short_term, long_term, semantic, procedural",
    )
    
    # Classification (security level)
    classification: Mapped[str] = mapped_column(
        String(50),
        default="internal",
        nullable=False,
        doc="Classification: public, internal, confidential, restricted",
    )
    required_clearance: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Minimum clearance level required (0-4)",
    )
    
    # Content
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Optional title for the memory",
    )
    content_preview: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        doc="Short preview of memory content",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        doc="SHA-256 hash of content for deduplication",
    )
    
    # Metadata
    tags: Mapped[List[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
        doc="Searchable tags",
    )
    entities: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Extracted entities (people, places, orgs, etc.)",
    )
    extra_metadata: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Additional metadata (JSON)",
    )
    
    # Source tracking
    source_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Source type: manual, agent, integration, etc.",
    )
    source_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Source identifier (agent_id, integration reference, etc.)",
    )
    
    # Vector reference
    vector_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        doc="ID in Qdrant vector store",
    )
    embedding_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="Model used to generate embedding",
    )
    
    # Access tracking
    access_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of times this memory was accessed",
    )
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="Last access timestamp",
    )
    
    # Retention
    retention_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        doc="Retention period in days (null = indefinite)",
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        index=True,
        doc="When memory expires (based on retention)",
    )
    legal_hold: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether memory is under legal hold (prevents deletion)",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether memory is active (soft delete)",
    )
    is_promoted: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether memory was promoted from lower scope",
    )
    promoted_from_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Original memory ID if this was promoted",
    )
    
    # Full-text search support (for hybrid search)
    # This column is automatically maintained by database trigger
    # See migration: 2026_01_27_add_fts_hybrid_search.py
    search_vector: Mapped[Optional[str]] = mapped_column(
        TSVECTOR,
        nullable=True,
        doc="Pre-computed tsvector for full-text search (auto-maintained by trigger)",
    )
    
    # Relationships
    shares: Mapped[List["MemorySharing"]] = relationship(
        "MemorySharing",
        back_populates="memory",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        # Index for scope-based queries
        Index("ix_memory_org_scope", "organization_id", "scope", "scope_id"),
        # Index for owner queries
        Index("ix_memory_owner", "organization_id", "owner_id"),
        # Index for tag searches (GIN)
        Index("ix_memory_tags", "tags", postgresql_using="gin"),
        # Index for entity searches (GIN)
        Index("ix_memory_entities", "entities", postgresql_using="gin"),
        # Ensure valid scope values
        CheckConstraint(
            "scope IN ('personal', 'team', 'department', 'division', 'organization', 'global')",
            name="ck_memory_scope",
        ),
        # Ensure valid classification values
        CheckConstraint(
            "classification IN ('public', 'internal', 'confidential', 'restricted')",
            name="ck_memory_classification",
        ),
    )
    
    def __repr__(self) -> str:
        return f"<MemoryMetadata {self.id[:8]}...>"


class MemorySharing(Base, UUIDMixin, TimestampMixin):
    """
    Memory sharing model.
    
    Tracks explicit sharing of memories with users, teams, or other scopes.
    
    Attributes:
        memory_id: Memory being shared
        share_type: Type of target (user, team, department, etc.)
        target_id: ID of the share target
        permission: Access level granted (read, comment, edit)
        expires_at: When the share expires
    """
    
    __tablename__ = "memory_sharing"
    
    # Memory reference
    memory_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Memory being shared",
    )
    
    # Organization (for RLS)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization context",
    )
    
    # Share target
    share_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Type: user, team, department, division, organization, external",
    )
    target_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Target entity ID (user_id, team_id, etc.)",
    )
    
    # Permission level
    permission: Mapped[str] = mapped_column(
        String(50),
        default="read",
        nullable=False,
        doc="Access level: read, comment, edit",
    )
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        index=True,
        doc="When this share expires",
    )
    
    # Audit
    shared_by: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        doc="User who created this share",
    )
    share_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Reason for sharing (for audit)",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether share is active",
    )
    
    # Notification
    notified_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When target was notified of share",
    )
    
    # Relationships
    memory: Mapped["MemoryMetadata"] = relationship(
        "MemoryMetadata",
        back_populates="shares",
    )
    
    __table_args__ = (
        # Index for looking up shares for a target
        Index("ix_sharing_target", "share_type", "target_id"),
        # Index for expiration checks
        Index("ix_sharing_expires", "expires_at"),
        # Unique constraint to prevent duplicate shares
        Index(
            "ix_sharing_unique",
            "memory_id", "share_type", "target_id",
            unique=True,
        ),
        # Ensure valid share types
        CheckConstraint(
            "share_type IN ('user', 'team', 'department', 'division', 'organization', 'external')",
            name="ck_share_type",
        ),
        # Ensure valid permission values
        CheckConstraint(
            "permission IN ('read', 'comment', 'edit')",
            name="ck_share_permission",
        ),
    )
    
    def __repr__(self) -> str:
        return f"<MemorySharing memory={self.memory_id[:8]} target={self.target_id[:8]}>"
    
    @property
    def is_expired(self) -> bool:
        """Check if share has expired."""
        if self.expires_at is None:
            return False
        return self.expires_at < datetime.now()


# Backward compatibility alias for legacy imports
Memory = MemoryMetadata
