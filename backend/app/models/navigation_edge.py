"""Navigation Edge model (GAP-6: kNN Navigation Graph).

Materialized kNN graph that connects memory hierarchy nodes (episodes,
semantic nodes, topics) based on embedding similarity.  This enables
**cross-cluster traversal** – the key missing capability identified in
the xMemory gap analysis.

Each node maintains its top-k nearest neighbours (default k=5).  Edges
are recomputed incrementally when:
    • A new episode or semantic node is created
    • A topic centroid shifts after absorbing new children
    • Periodic background refresh (Celery beat)

The graph supports the **top-down adaptive retrieval** strategy:
    Query → match Topics → traverse NavigationEdges to SemanticNodes → expand to Episodes → Messages

Edge weights are cosine similarity scores from Qdrant's recommend API.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class NavigationEdge(Base, UUIDMixin, TimestampMixin):
    """kNN navigation edge between hierarchy nodes.

    source_type / target_type ∈ { 'episode', 'semantic_node', 'topic' }
    """

    __tablename__ = "navigation_edges"

    # ── Tenant isolation ────────────────────────────────────────────
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization (tenant isolation)",
    )

    # ── Source node ─────────────────────────────────────────────────
    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Node type: episode | semantic_node | topic",
    )

    source_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        doc="Primary key of the source node in its respective table",
    )

    # ── Target node ─────────────────────────────────────────────────
    target_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        doc="Node type: episode | semantic_node | topic",
    )

    target_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        doc="Primary key of the target node in its respective table",
    )

    # ── Edge properties ─────────────────────────────────────────────
    similarity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        doc="Cosine similarity between source and target centroids (0-1)",
    )

    k_rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        doc="Neighbour rank for the source node (1 = nearest, k = farthest)",
    )

    # ── Metadata ────────────────────────────────────────────────────
    generation: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        doc="Rebuild generation counter (monotonic; stale rows have lower gen)",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        default="system",
        nullable=False,
        doc="Creator identifier (system/agent/user)",
    )

    __table_args__ = (
        # Fast lookup: "give me all k-NN edges for a given source node"
        Index(
            "ix_navigation_edges_source",
            "organization_id",
            "source_type",
            "source_id",
        ),
        # Fast lookup: "which nodes point to this target?"
        Index(
            "ix_navigation_edges_target",
            "organization_id",
            "target_type",
            "target_id",
        ),
        # Dedup: one edge per (org, source, target) pair
        Index(
            "ux_navigation_edges_pair",
            "organization_id",
            "source_type",
            "source_id",
            "target_type",
            "target_id",
            unique=True,
        ),
        # Prune by generation
        Index(
            "ix_navigation_edges_generation",
            "organization_id",
            "generation",
        ),
    )
