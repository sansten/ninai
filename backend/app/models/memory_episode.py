"""Memory Episode model (GAP-1: Four-Level Hierarchy – Level 2).

An *episode* groups a contiguous sequence of messages (MemoryMetadata rows)
that share conversational context.  Boundary detection is driven by:

    • topic shift  (cosine distance between consecutive embeddings > θ_topic)
    • temporal gap  (Δt > θ_time)
    • intent transition  (LLM-detected intent change)

The episode carries a **narrative summary** distilled by an LLM and a Qdrant
vector_id for its centroid embedding used by the kNN navigation graph (GAP-6).

Split threshold derived from Fano inequality:
    n_k = 2^B / (1 − H(P_e))  where B ≈ 2 bits, P_e ≈ 0.15 → n_k ≈ 12
Episodes with > n_k messages should be considered for sub-splitting.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class MemoryEpisode(Base, UUIDMixin, TimestampMixin):
    """Level-2 node in the four-level memory hierarchy.

    Messages ─▶ **Episodes** ─▶ SemanticNodes ─▶ Topics
    """

    __tablename__ = "memory_episodes"

    # ── Tenant isolation ────────────────────────────────────────────
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this episode belongs to (tenant isolation)",
    )

    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
        doc="User whose conversation produced this episode",
    )

    # ── Scope (matches MemoryMetadata convention) ───────────────────
    scope: Mapped[str] = mapped_column(
        String(50),
        default="personal",
        nullable=False,
        doc="Scope: personal/team/department/division/organization/global",
    )

    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Optional scope entity id (e.g. team_id)",
    )

    # ── Content ─────────────────────────────────────────────────────
    title: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="Auto-generated episode title",
    )

    narrative_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="LLM-generated narrative summary of the episode",
    )

    # ── Temporal boundaries ─────────────────────────────────────────
    boundary_start: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of the first message in the episode",
    )

    boundary_end: Mapped[Optional[str]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of the last message in the episode",
    )

    # ── Hierarchy ───────────────────────────────────────────────────
    topic_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("memory_topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        doc="Parent topic (Level-4) this episode rolls up into",
    )

    # ── Embedding (Qdrant) ──────────────────────────────────────────
    vector_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        doc="Qdrant point id for the episode centroid embedding",
    )

    # ── Metadata ────────────────────────────────────────────────────
    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of messages grouped in this episode",
    )

    boundary_reason: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Reason boundary was placed: topic_shift / temporal_gap / intent_change / manual",
    )

    boundary_confidence: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        nullable=False,
        doc="Confidence of the boundary detection (0-1)",
    )

    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        doc="Arbitrary metadata (e.g. entity list, detected intents)",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        default="open",
        nullable=False,
        doc="Episode lifecycle: open (still receiving messages) | closed | merged",
    )

    created_by: Mapped[str] = mapped_column(
        String(50),
        default="system",
        nullable=False,
        doc="Creator identifier (system/agent/user)",
    )

    __table_args__ = (
        Index("ix_memory_episodes_org_owner", "organization_id", "owner_id"),
        Index("ix_memory_episodes_org_topic", "organization_id", "topic_id"),
        Index("ix_memory_episodes_org_status", "organization_id", "status"),
        Index(
            "ix_memory_episodes_org_boundary",
            "organization_id",
            "boundary_start",
            "boundary_end",
        ),
    )
