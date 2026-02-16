"""Add four-level hierarchy (GAP-1) and kNN navigation graph (GAP-6)

Creates tables:
  - memory_episodes           (Level 2 – episode boundaries)
  - memory_episode_memberships (Level 1↔2 association)
  - memory_semantic_nodes     (Level 3 – distilled knowledge)
  - navigation_edges          (GAP-6 – kNN cross-cluster graph)

Revision ID: 20260202_hierarchy_knn
Revises: 20260128_mem_activation
Create Date: 2026-02-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260202_hierarchy_knn"
down_revision: Union[str, None] = "20260128_mem_activation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── memory_episodes ─────────────────────────────────────────────
    op.create_table(
        "memory_episodes",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("scope", sa.String(50), nullable=False, server_default=sa.text("'personal'")),
        sa.Column("scope_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("narrative_summary", sa.Text(), nullable=True),
        sa.Column("boundary_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("boundary_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("topic_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("vector_id", sa.String(128), nullable=True),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("boundary_reason", sa.String(50), nullable=True),
        sa.Column("boundary_confidence", sa.Float(), nullable=False, server_default=sa.text("1.0")),
        sa.Column("extra_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'open'")),
        sa.Column("created_by", sa.String(50), nullable=False, server_default=sa.text("'system'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["topic_id"], ["memory_topics.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_episodes_organization_id", "memory_episodes", ["organization_id"])
    op.create_index("ix_memory_episodes_owner_id", "memory_episodes", ["owner_id"])
    op.create_index("ix_memory_episodes_topic_id", "memory_episodes", ["topic_id"])
    op.create_index("ix_memory_episodes_org_owner", "memory_episodes", ["organization_id", "owner_id"])
    op.create_index("ix_memory_episodes_org_topic", "memory_episodes", ["organization_id", "topic_id"])
    op.create_index("ix_memory_episodes_org_status", "memory_episodes", ["organization_id", "status"])
    op.create_index(
        "ix_memory_episodes_org_boundary",
        "memory_episodes",
        ["organization_id", "boundary_start", "boundary_end"],
    )

    # ── memory_episode_memberships ──────────────────────────────────
    op.create_table(
        "memory_episode_memberships",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("episode_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_by", sa.String(50), nullable=False, server_default=sa.text("'system'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["episode_id"], ["memory_episodes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_episode_memberships_organization_id", "memory_episode_memberships", ["organization_id"])
    op.create_index("ix_memory_episode_memberships_memory_id", "memory_episode_memberships", ["memory_id"])
    op.create_index("ix_memory_episode_memberships_episode_id", "memory_episode_memberships", ["episode_id"])
    op.create_index(
        "ux_memory_episode_membership",
        "memory_episode_memberships",
        ["organization_id", "memory_id", "episode_id"],
        unique=True,
    )
    op.create_index(
        "ix_memory_episode_membership_episode",
        "memory_episode_memberships",
        ["episode_id", "position"],
    )

    # ── memory_semantic_nodes ───────────────────────────────────────
    op.create_table(
        "memory_semantic_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("scope", sa.String(50), nullable=False, server_default=sa.text("'personal'")),
        sa.Column("scope_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("persistence_score", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("specificity_score", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("utility_score", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("independence_score", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("composite_quality", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("topic_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("vector_id", sa.String(128), nullable=True),
        sa.Column("source_episode_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_memory_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default=sa.text("'[]'::jsonb")),
        sa.Column("reference_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("entities", postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default=sa.text("'[]'::jsonb")),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default=sa.text("'[]'::jsonb")),
        sa.Column("status", sa.String(32), nullable=False, server_default=sa.text("'active'")),
        sa.Column("created_by", sa.String(50), nullable=False, server_default=sa.text("'agent'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["topic_id"], ["memory_topics.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_memory_semantic_nodes_organization_id", "memory_semantic_nodes", ["organization_id"])
    op.create_index("ix_memory_semantic_nodes_owner_id", "memory_semantic_nodes", ["owner_id"])
    op.create_index("ix_memory_semantic_nodes_topic_id", "memory_semantic_nodes", ["topic_id"])
    op.create_index("ix_memory_semantic_nodes_content_hash_idx", "memory_semantic_nodes", ["content_hash"])
    op.create_index("ix_memory_semantic_nodes_org_owner", "memory_semantic_nodes", ["organization_id", "owner_id"])
    op.create_index("ix_memory_semantic_nodes_org_topic", "memory_semantic_nodes", ["organization_id", "topic_id"])
    op.create_index("ix_memory_semantic_nodes_org_quality", "memory_semantic_nodes", ["organization_id", "composite_quality"])
    op.create_index("ix_memory_semantic_nodes_org_content_hash", "memory_semantic_nodes", ["organization_id", "content_hash"])

    # ── navigation_edges ────────────────────────────────────────────
    op.create_table(
        "navigation_edges",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("source_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("target_type", sa.String(32), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("k_rank", sa.Integer(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_by", sa.String(50), nullable=False, server_default=sa.text("'system'")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_navigation_edges_organization_id", "navigation_edges", ["organization_id"])
    op.create_index(
        "ix_navigation_edges_source",
        "navigation_edges",
        ["organization_id", "source_type", "source_id"],
    )
    op.create_index(
        "ix_navigation_edges_target",
        "navigation_edges",
        ["organization_id", "target_type", "target_id"],
    )
    op.create_index(
        "ux_navigation_edges_pair",
        "navigation_edges",
        ["organization_id", "source_type", "source_id", "target_type", "target_id"],
        unique=True,
    )
    op.create_index(
        "ix_navigation_edges_generation",
        "navigation_edges",
        ["organization_id", "generation"],
    )

    # ── RLS policies ────────────────────────────────────────────────
    for table in ["memory_episodes", "memory_episode_memberships", "memory_semantic_nodes", "navigation_edges"]:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant_isolation ON {table} "
            f"USING (organization_id = current_setting('app.current_org_id')::uuid)"
        )


def downgrade() -> None:
    # Drop RLS policies
    for table in ["navigation_edges", "memory_semantic_nodes", "memory_episode_memberships", "memory_episodes"]:
        op.execute(f"DROP POLICY IF EXISTS {table}_tenant_isolation ON {table}")

    op.drop_table("navigation_edges")
    op.drop_table("memory_semantic_nodes")
    op.drop_table("memory_episode_memberships")
    op.drop_table("memory_episodes")
