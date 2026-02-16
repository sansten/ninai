"""Memory semantic node topic reassignment history (GAP-5).

Tracks topic reassignments for retroactive restructuring. Enables computation of
reassignment ratio metric per xMemory framework: the percentage of semantic nodes
that have changed topics over time, which should target 40%+ for dynamic adaptation.

Reference: Hu et al. (2026), "Beyond RAG for Agent Memory", Section 4.4
"""

from sqlalchemy import String, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base, UUIDMixin, TimestampMixin


class MemorySemanticNodeTopicHistory(Base, UUIDMixin, TimestampMixin):
    """History of topic reassignments for a semantic node.
    
    Each record represents a topic assignment event (initial or reassignment).
    Used to compute the reassignment ratio: % of nodes that have changed topics.
    
    Attributes:
        organization_id: Organization (tenant isolation)
        semantic_node_id: The semantic node being assigned
        topic_id: Topic assigned at this event
        previous_topic_id: Previous topic (None for initial assignment)
        reason: Why reassignment occurred (split, merge, guided_attach, periodic_restructure)
        guidance_score_before: f(P) before this reassignment
        guidance_score_after: f(P) after this reassignment
    """
    
    __tablename__ = "memory_semantic_node_topic_history"
    
    # Organization (tenant isolation)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this history entry belongs to",
    )
    
    # Semantic node being reassigned
    semantic_node_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_semantic_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Semantic node that was reassigned",
    )
    
    # Topic assignment
    topic_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_topics.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Topic assigned at this event",
    )
    
    previous_topic_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Previous topic ID (None for initial assignment)",
    )
    
    # Reassignment metadata
    reason: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Reason: initial_attach, split, merge, guided_attach, periodic_restructure",
    )
    
    guidance_score_before: Mapped[float | None] = mapped_column(
        nullable=True,
        doc="Guidance score f(P) before reassignment",
    )
    
    guidance_score_after: Mapped[float | None] = mapped_column(
        nullable=True,
        doc="Guidance score f(P) after reassignment",
    )
    
    __table_args__ = (
        # Index for finding node's reassignment history
        Index(
            "ix_semantic_node_topic_history_node_created",
            "semantic_node_id",
            "created_at",
        ),
        # Index for tracking reassignments in org
        Index(
            "ix_semantic_node_topic_history_org_created",
            "organization_id",
            "created_at",
        ),
        # Index for counting reassignments (exclude initial attach)
        Index(
            "ix_semantic_node_topic_history_org_reason",
            "organization_id",
            "reason",
        ),
    )
