"""Goal hierarchy node model for hierarchical planning (Phase 65)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GoalHierarchyNode(Base, UUIDMixin):
    __tablename__ = "goal_hierarchy_nodes"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    root_goal_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("goals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    parent_node_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("goal_hierarchy_nodes.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    goal_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("goals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    estimated_effort: Mapped[str] = mapped_column(String(32), nullable=False, default="small")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_goal_hierarchy_nodes_org_root_depth", "org_id", "root_goal_id", "depth"),
    )
