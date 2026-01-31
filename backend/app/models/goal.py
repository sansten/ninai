"""GoalGraph models.

Implements long-horizon goal tracking:
- Goal: top-level goal
- GoalNode: subgoals/tasks/milestones
- GoalEdge: dependencies between nodes
- GoalMemoryLink: links to memory items
- GoalActivityLog: audit-like activity trail

All tables are tenant-scoped via organization_id and protected by Postgres RLS.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDMixin


class Goal(Base, UUIDMixin):
    __tablename__ = "goals"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    created_by_user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    owner_type: Mapped[str] = mapped_column(String(length=30), nullable=False)
    owner_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)

    title: Mapped[str] = mapped_column(Text(), nullable=False)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)

    goal_type: Mapped[str] = mapped_column(String(length=30), nullable=False)
    status: Mapped[str] = mapped_column(String(length=20), nullable=False)
    priority: Mapped[int] = mapped_column(Integer(), nullable=False, server_default="0")

    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    confidence: Mapped[float] = mapped_column(Float(), nullable=False, server_default="0.5")

    visibility_scope: Mapped[str] = mapped_column(String(length=30), nullable=False)
    scope_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)

    tags: Mapped[list[str]] = mapped_column(ARRAY(Text()), nullable=False, server_default="{}")
    extra_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB(),
        nullable=False,
        server_default="{}",
    )

    nodes: Mapped[list["GoalNode"]] = relationship(
        "GoalNode",
        back_populates="goal",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    memory_links: Mapped[list["GoalMemoryLink"]] = relationship(
        "GoalMemoryLink",
        back_populates="goal",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class GoalNode(Base, UUIDMixin):
    __tablename__ = "goal_nodes"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    goal_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    parent_node_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("goal_nodes.id", ondelete="SET NULL"),
        nullable=True,
    )

    node_type: Mapped[str] = mapped_column(String(length=20), nullable=False)
    title: Mapped[str] = mapped_column(Text(), nullable=False)
    description: Mapped[str | None] = mapped_column(Text(), nullable=True)

    status: Mapped[str] = mapped_column(String(length=20), nullable=False, server_default="todo")
    priority: Mapped[int] = mapped_column(Integer(), nullable=False, server_default="0")

    assigned_to_user_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)
    assigned_to_team_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    confidence: Mapped[float] = mapped_column(Float(), nullable=False, server_default="0.5")
    ordering: Mapped[int] = mapped_column(Integer(), nullable=False, server_default="0")

    expected_outputs: Mapped[dict | None] = mapped_column(JSONB(), nullable=True)
    success_criteria: Mapped[list[str] | None] = mapped_column(ARRAY(Text()), nullable=True)
    blockers: Mapped[dict | None] = mapped_column(JSONB(), nullable=True)

    goal: Mapped[Goal] = relationship("Goal", back_populates="nodes")
    parent: Mapped["GoalNode"] = relationship("GoalNode", remote_side="GoalNode.id")


class GoalEdge(Base, UUIDMixin):
    __tablename__ = "goal_edges"

    __table_args__ = (
        UniqueConstraint("from_node_id", "to_node_id", "edge_type", name="uq_goal_edges_triplet"),
    )

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    from_node_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("goal_nodes.id", ondelete="CASCADE"), nullable=False)
    to_node_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("goal_nodes.id", ondelete="CASCADE"), nullable=False)

    edge_type: Mapped[str] = mapped_column(String(length=20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


class GoalMemoryLink(Base, UUIDMixin):
    __tablename__ = "goal_memory_links"

    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "goal_id",
            "memory_id",
            name="uq_goal_memory_links_goal_memory",
        ),
    )

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    goal_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("goals.id", ondelete="CASCADE"), nullable=False, index=True)
    node_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("goal_nodes.id", ondelete="SET NULL"), nullable=True)
    memory_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("memory_metadata.id", ondelete="CASCADE"), nullable=False, index=True)

    link_type: Mapped[str] = mapped_column(String(length=20), nullable=False)
    linked_by: Mapped[str] = mapped_column(String(length=10), nullable=False, server_default="user")
    confidence: Mapped[float] = mapped_column(Float(), nullable=False, server_default="0.5")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    goal: Mapped[Goal] = relationship("Goal", back_populates="memory_links")


class GoalActivityLog(Base, UUIDMixin):
    __tablename__ = "goal_activity_log"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    goal_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("goals.id", ondelete="CASCADE"), nullable=False, index=True)
    node_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("goal_nodes.id", ondelete="SET NULL"), nullable=True)

    actor_type: Mapped[str] = mapped_column(String(length=10), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)

    action: Mapped[str] = mapped_column(String(length=80), nullable=False)
    details: Mapped[dict] = mapped_column(JSONB(), nullable=False, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
