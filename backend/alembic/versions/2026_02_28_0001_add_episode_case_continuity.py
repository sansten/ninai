"""Add episode case continuity tables (PR1: Advanced Memory Features)

Creates tables for case/ticket/thread tracking:
  - episodes       (case histories: support tickets, research threads, etc.)
  - episode_events (timeline events within episodes)
  - episode_links  (relationships between episodes)

Includes RLS policies for multi-tenancy.

Revision ID: 20260228_episodes
Revises: 20260202_hierarchy_knn
Create Date: 2026-02-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260228_episodes"
down_revision: Union[str, None] = "20260202_hierarchy_knn"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── episodes ────────────────────────────────────────────────────
    op.create_table(
        "episodes",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "scope_type",
            sa.String(20),
            nullable=False,
            server_default="personal",
        ),
        sa.Column(
            "scope_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column(
            "owner_user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("episode_type", sa.String(100), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "last_event_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "resolved_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.String()),
            nullable=True,
        ),
        sa.Column(
            "entities",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Indexes for episodes
    op.create_index(
        "ix_episodes_organization_id",
        "episodes",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_scope_id",
        "episodes",
        ["scope_id"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_owner_user_id",
        "episodes",
        ["owner_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_episode_type",
        "episodes",
        ["episode_type"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_status",
        "episodes",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_started_at",
        "episodes",
        ["started_at"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_last_event_at",
        "episodes",
        ["last_event_at"],
        unique=False,
    )
    op.create_index(
        "ix_episodes_org_last_event",
        "episodes",
        ["organization_id", "last_event_at"],
        unique=False,
        postgresql_using="btree",
    )
    op.create_index(
        "ix_episodes_status_org",
        "episodes",
        ["status", "organization_id"],
        unique=False,
        postgresql_using="btree",
    )
    op.create_index(
        "ix_episodes_tags",
        "episodes",
        ["tags"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_episodes_entities",
        "episodes",
        ["entities"],
        unique=False,
        postgresql_using="gin",
    )

    # ── episode_events ──────────────────────────────────────────────
    op.create_table(
        "episode_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column(
            "event_ts",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("actor_type", sa.String(20), nullable=False),
        sa.Column(
            "actor_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Indexes for episode_events
    op.create_index(
        "ix_episode_events_organization_id",
        "episode_events",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_episode_events_episode_id",
        "episode_events",
        ["episode_id"],
        unique=False,
    )
    op.create_index(
        "ix_episode_events_memory_id",
        "episode_events",
        ["memory_id"],
        unique=False,
    )
    op.create_index(
        "ix_episode_events_event_type",
        "episode_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        "ix_episode_events_event_ts",
        "episode_events",
        ["event_ts"],
        unique=False,
    )
    op.create_index(
        "ix_episode_events_episode_ts",
        "episode_events",
        ["episode_id", "event_ts"],
        unique=False,
        postgresql_using="btree",
    )
    op.create_index(
        "ix_episode_events_org_ts",
        "episode_events",
        ["organization_id", "event_ts"],
        unique=False,
        postgresql_using="btree",
    )
    op.create_index(
        "ix_episode_events_type",
        "episode_events",
        ["event_type"],
        unique=False,
        postgresql_using="btree",
    )

    # ── episode_links ───────────────────────────────────────────────
    op.create_table(
        "episode_links",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "from_episode_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_episode_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Indexes for episode_links
    op.create_index(
        "ix_episode_links_organization_id",
        "episode_links",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_episode_links_from_episode",
        "episode_links",
        ["from_episode_id"],
        unique=False,
        postgresql_using="btree",
    )
    op.create_index(
        "ix_episode_links_to_episode",
        "episode_links",
        ["to_episode_id"],
        unique=False,
        postgresql_using="btree",
    )
    op.create_index(
        "ix_episode_links_relation",
        "episode_links",
        ["relation"],
        unique=False,
        postgresql_using="btree",
    )

    # Unique constraint
    op.create_unique_constraint(
        "uq_episode_links_from_to_relation",
        "episode_links",
        ["from_episode_id", "to_episode_id", "relation"],
    )

    # ── RLS Policies ────────────────────────────────────────────────
    # Enable RLS on all three tables
    op.execute("ALTER TABLE episodes ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE episode_events ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE episode_links ENABLE ROW LEVEL SECURITY")

    # RLS for episodes
    op.execute("""
        CREATE POLICY episodes_org_isolation ON episodes
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
    """)

    # RLS for episode_events
    op.execute("""
        CREATE POLICY episode_events_org_isolation ON episode_events
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
    """)

    # RLS for episode_links
    op.execute("""
        CREATE POLICY episode_links_org_isolation ON episode_links
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
    """)


def downgrade() -> None:
    # Drop RLS policies
    op.execute("DROP POLICY IF EXISTS episodes_org_isolation ON episodes")
    op.execute("DROP POLICY IF EXISTS episode_events_org_isolation ON episode_events")
    op.execute("DROP POLICY IF EXISTS episode_links_org_isolation ON episode_links")

    # Drop episode_links
    op.drop_constraint("uq_episode_links_from_to_relation", "episode_links", type_="unique")
    op.drop_index("ix_episode_links_relation", table_name="episode_links")
    op.drop_index("ix_episode_links_to_episode", table_name="episode_links")
    op.drop_index("ix_episode_links_from_episode", table_name="episode_links")
    op.drop_index("ix_episode_links_organization_id", table_name="episode_links")
    op.drop_table("episode_links")

    # Drop episode_events
    op.drop_index("ix_episode_events_type", table_name="episode_events")
    op.drop_index("ix_episode_events_org_ts", table_name="episode_events")
    op.drop_index("ix_episode_events_episode_ts", table_name="episode_events")
    op.drop_index("ix_episode_events_event_ts", table_name="episode_events")
    op.drop_index("ix_episode_events_event_type", table_name="episode_events")
    op.drop_index("ix_episode_events_memory_id", table_name="episode_events")
    op.drop_index("ix_episode_events_episode_id", table_name="episode_events")
    op.drop_index("ix_episode_events_organization_id", table_name="episode_events")
    op.drop_table("episode_events")

    # Drop episodes
    op.drop_index("ix_episodes_entities", table_name="episodes")
    op.drop_index("ix_episodes_tags", table_name="episodes")
    op.drop_index("ix_episodes_status_org", table_name="episodes")
    op.drop_index("ix_episodes_org_last_event", table_name="episodes")
    op.drop_index("ix_episodes_last_event_at", table_name="episodes")
    op.drop_index("ix_episodes_started_at", table_name="episodes")
    op.drop_index("ix_episodes_status", table_name="episodes")
    op.drop_index("ix_episodes_episode_type", table_name="episodes")
    op.drop_index("ix_episodes_owner_user_id", table_name="episodes")
    op.drop_index("ix_episodes_scope_id", table_name="episodes")
    op.drop_index("ix_episodes_organization_id", table_name="episodes")
    op.drop_table("episodes")
