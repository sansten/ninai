"""Episode Link model for relationship tracking (PR1: Advanced Memory Features).

Links represent relationships between episodes (duplicates, causal, follow-ons, etc.).
Used for detecting patterns, clustering similar cases, and navigating case history.

Schema aligned with Ninai_Advanced_Memory_Features_Copilot_Requirements.md
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from sqlalchemy import (
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class EpisodeLinkRelation(str, Enum):
    """Type of relationship between episodes."""

    DUPLICATE = "duplicate"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"
    FOLLOW_ON = "follow_on"
    SAME_ACCOUNT = "same_account"
    SAME_DEVICE = "same_device"


class EpisodeLink(Base, UUIDMixin, TimestampMixin):
    """Relationship between two episodes.

    Examples:
    - duplicate: Two support cases reporting same issue
    - causal_hypothesis: Episode B may have been caused by Episode A
    - follow_on: Episode B is a continuation/escalation of Episode A
    - same_account/same_device: Episodes share entity context
    """

    __tablename__ = "episode_links"

    # ── Tenant isolation ────────────────────────────────────────────
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        doc="Organization these episodes belong to (RLS enforcement)",
    )

    # ── Episode references ──────────────────────────────────────────
    from_episode_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
        doc="Source episode",
    )

    to_episode_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
        doc="Target episode",
    )

    # ── Link metadata ───────────────────────────────────────────────
    relation: Mapped[EpisodeLinkRelation] = mapped_column(
        SQLEnum(EpisodeLinkRelation, native_enum=False),
        nullable=False,
        doc="Type of relationship",
    )

    confidence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        doc="Confidence score [0.0, 1.0] for relationship strength",
    )

    evidence: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        doc="Evidence supporting the relationship (similarity scores, shared entities, etc.)",
    )

    # ── Indexes and constraints ─────────────────────────────────────
    __table_args__ = (
        Index(
            "ix_episode_links_organization_id",
            "organization_id",
            postgresql_using="btree",
        ),
        Index(
            "ix_episode_links_from_episode",
            "from_episode_id",
            postgresql_using="btree",
        ),
        Index(
            "ix_episode_links_to_episode",
            "to_episode_id",
            postgresql_using="btree",
        ),
        Index(
            "ix_episode_links_relation",
            "relation",
            postgresql_using="btree",
        ),
        UniqueConstraint(
            "from_episode_id",
            "to_episode_id",
            "relation",
            name="uq_episode_links_from_to_relation",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EpisodeLink(from={self.from_episode_id}, to={self.to_episode_id}, "
            f"relation={self.relation}, confidence={self.confidence})>"
        )
