"""Memory Semantic Node model (GAP-1: Four-Level Hierarchy – Level 3).

A *semantic node* is a distilled, **reusable** knowledge unit extracted from
one or more episodes.  It represents a single high-value fact, rule, or
preference that passed the four xMemory distillation filters:

    1. **Persistence** – will this information remain useful across sessions?
    2. **Specificity** – is this specific enough to inform future actions?
    3. **Utility**     – can an agent use this to improve responses?
    4. **Independence** – does it stand alone without conversational context?

Hierarchy:   Messages ─▶ Episodes ─▶ **SemanticNodes** ─▶ Topics

Each semantic node stores its Qdrant centroid vector_id used by the kNN
navigation graph (GAP-6) for efficient cross-cluster traversal.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemorySemanticNode(Base, UUIDMixin, TimestampMixin):
    """Level-3 node in the four-level memory hierarchy.

    Messages ─▶ Episodes ─▶ **SemanticNodes** ─▶ Topics
    """

    __tablename__ = "memory_semantic_nodes"

    # ── Tenant isolation ────────────────────────────────────────────
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this semantic node belongs to (tenant isolation)",
    )

    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
        doc="User who owns this knowledge",
    )

    # ── Scope ───────────────────────────────────────────────────────
    scope: Mapped[str] = mapped_column(
        String(50),
        default="personal",
        nullable=False,
        doc="Scope: personal/team/department/division/organization/global",
    )

    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Optional scope entity id",
    )

    # ── Content ─────────────────────────────────────────────────────
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        doc="Distilled knowledge statement (1-3 sentences)",
    )

    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        doc="SHA-256 hash of content for deduplication",
    )

    # ── Distillation scores (xMemory § 3.2) ────────────────────────
    persistence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Will this survive across sessions? (0-1)",
    )

    specificity_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Is this specific enough to act on? (0-1)",
    )

    utility_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Can an agent use this to improve replies? (0-1)",
    )

    independence_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Does it stand alone without conversational context? (0-1)",
    )

    composite_quality: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        doc="Geometric mean of the four distillation scores",
    )

    # ── Hierarchy ───────────────────────────────────────────────────
    topic_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Parent topic (Level-4) this semantic node belongs to",
    )

    # ── Embedding (Qdrant) ──────────────────────────────────────────
    vector_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Qdrant point id for the semantic node embedding",
    )

    # ── Source tracking ─────────────────────────────────────────────
    source_episode_ids: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        doc="Episode IDs this semantic node was distilled from",
    )

    source_memory_ids: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        doc="Original memory IDs that contributed to this node",
    )

    # ── Metadata ────────────────────────────────────────────────────
    reference_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="How many times this node has been retrieved/referenced",
    )

    entities: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        doc="Extracted entities (people, places, products, etc.)",
    )

    tags: Mapped[Optional[list]] = mapped_column(
        JSONB,
        nullable=True,
        default=list,
        doc="Semantic tags for quick filtering",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="active",
        nullable=False,
        doc="Lifecycle: active | superseded | merged | deleted",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        default="agent",
        nullable=False,
        doc="Creator identifier (agent/system/user)",
    )

    __table_args__ = (
        Index("ix_memory_semantic_nodes_org_owner", "organization_id", "owner_id"),
        Index("ix_memory_semantic_nodes_org_topic", "organization_id", "topic_id"),
        Index("ix_memory_semantic_nodes_org_quality", "organization_id", "composite_quality"),
        Index("ix_memory_semantic_nodes_content_hash", "organization_id", "content_hash"),
    )
