"""Add GoalGraph tables.

Revision ID: 2026_01_24_0003
Revises: 2026_01_24_0002
Create Date: 2026-01-24

Adds:
- goals
- goal_nodes
- goal_edges
- goal_memory_links
- goal_activity_log

All tables are protected by Postgres RLS and scoped by organization_id.
Visibility rules:
- organization: any user in org
- personal: created_by_user_id == app.current_user_id
- team: user must be team member of scope_id
- department/division: user must be in a team under hierarchy node scope_id

Note: RLS policies rely on app.current_org_id/app.current_user_id being set
(via set_tenant_context / get_db_with_tenant patterns).
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_24_0003"
down_revision = "2026_01_24_0002"
branch_labels = None
depends_on = None


_GOAL_VISIBILITY_CHECK_SQL = """
(
  {table}.organization_id::text = current_setting('app.current_org_id', true)
  AND (
    {table}.visibility_scope = 'organization'
    OR ({table}.visibility_scope = 'personal' AND {table}.created_by_user_id::text = current_setting('app.current_user_id', true))
    OR (
      {table}.visibility_scope = 'team'
      AND EXISTS (
        SELECT 1 FROM team_members tm
        WHERE tm.organization_id = {table}.organization_id
          AND tm.team_id::text = COALESCE({table}.scope_id::text, '')
          AND tm.user_id::text = current_setting('app.current_user_id', true)
      )
    )
    OR (
      {table}.visibility_scope IN ('department', 'division')
      AND EXISTS (
        SELECT 1
        FROM team_members tm
        JOIN teams t ON t.id = tm.team_id
        JOIN organization_hierarchy oh_team ON oh_team.id = t.hierarchy_node_id
        JOIN organization_hierarchy oh_scope ON oh_scope.id::text = COALESCE({table}.scope_id::text, '')
        WHERE tm.organization_id = {table}.organization_id
          AND tm.user_id::text = current_setting('app.current_user_id', true)
          AND oh_team.organization_id = {table}.organization_id
          AND oh_scope.organization_id = {table}.organization_id
          AND oh_team.path <@ oh_scope.path
      )
    )
    OR (
      string_to_array(current_setting('app.current_roles', true), ',') && ARRAY['org_admin','system_admin']
    )
  )
)
"""


def upgrade() -> None:
    # ---------------------------------------------------------------------
    # goals
    # ---------------------------------------------------------------------
    op.create_table(
        "goals",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "created_by_user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("owner_type", sa.String(length=30), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("goal_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column("visibility_scope", sa.String(length=30), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=False, server_default=sa.text("'{}'::text[]")),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_goals_confidence"),
        sa.CheckConstraint(
            "priority >= 0 AND priority <= 5",
            name="ck_goals_priority",
        ),
        sa.CheckConstraint(
            "owner_type IN ('user','team','department','organization')",
            name="ck_goals_owner_type",
        ),
        sa.CheckConstraint(
            "goal_type IN ('task','project','objective','policy','research')",
            name="ck_goals_goal_type",
        ),
        sa.CheckConstraint(
            "status IN ('proposed','active','blocked','completed','abandoned')",
            name="ck_goals_status",
        ),
        sa.CheckConstraint(
            "visibility_scope IN ('personal','team','department','division','organization')",
            name="ck_goals_visibility_scope",
        ),
    )

    op.create_index("ix_goals_org_status", "goals", ["organization_id", "status"], unique=False)
    op.create_index("ix_goals_org_due_at", "goals", ["organization_id", "due_at"], unique=False)
    op.create_index("ix_goals_tags_gin", "goals", ["tags"], postgresql_using="gin")
    op.create_index("ix_goals_metadata_gin", "goals", ["metadata"], postgresql_using="gin")

    # ---------------------------------------------------------------------
    # goal_nodes
    # ---------------------------------------------------------------------
    op.create_table(
        "goal_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "goal_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "parent_node_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goal_nodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("node_type", sa.String(length=20), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default=sa.text("'todo'")),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("assigned_to_user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("assigned_to_team_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column("ordering", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "expected_outputs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "success_criteria",
            postgresql.ARRAY(sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "blockers",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_goal_nodes_confidence"),
        sa.CheckConstraint(
            "priority >= 0 AND priority <= 5",
            name="ck_goal_nodes_priority",
        ),
        sa.CheckConstraint(
            "node_type IN ('subgoal','task','milestone')",
            name="ck_goal_nodes_node_type",
        ),
        sa.CheckConstraint(
            "status IN ('todo','in_progress','blocked','done','cancelled')",
            name="ck_goal_nodes_status",
        ),
    )

    op.create_index("ix_goal_nodes_goal_ordering", "goal_nodes", ["goal_id", "ordering"], unique=False)

    # ---------------------------------------------------------------------
    # goal_edges
    # ---------------------------------------------------------------------
    op.create_table(
        "goal_edges",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "from_node_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goal_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_node_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goal_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("edge_type", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("from_node_id", "to_node_id", "edge_type", name="uq_goal_edges_triplet"),
        sa.CheckConstraint(
            "edge_type IN ('depends_on','blocks','related_to')",
            name="ck_goal_edges_edge_type",
        ),
    )

    op.create_index("ix_goal_edges_from", "goal_edges", ["from_node_id"], unique=False)
    op.create_index("ix_goal_edges_to", "goal_edges", ["to_node_id"], unique=False)

    # ---------------------------------------------------------------------
    # goal_memory_links
    # ---------------------------------------------------------------------
    op.create_table(
        "goal_memory_links",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "goal_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "node_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goal_nodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("link_type", sa.String(length=20), nullable=False),
        sa.Column("linked_by", sa.String(length=10), nullable=False, server_default=sa.text("'user'")),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint(
            "organization_id",
            "goal_id",
            "memory_id",
            name="uq_goal_memory_links_goal_memory",
        ),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_goal_memory_links_confidence"),
        sa.CheckConstraint(
            "link_type IN ('evidence','progress','blocker','reference')",
            name="ck_goal_memory_links_link_type",
        ),
        sa.CheckConstraint(
            "linked_by IN ('auto','user','agent')",
            name="ck_goal_memory_links_linked_by",
        ),
    )

    op.create_index("ix_goal_memory_links_goal_memory", "goal_memory_links", ["goal_id", "memory_id"], unique=False)

    # ---------------------------------------------------------------------
    # goal_activity_log
    # ---------------------------------------------------------------------
    op.create_table(
        "goal_activity_log",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "goal_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goals.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "node_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("goal_nodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("actor_type", sa.String(length=10), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint(
            "actor_type IN ('user','agent','system')",
            name="ck_goal_activity_actor_type",
        ),
    )

    op.create_index("ix_goal_activity_log_org_created", "goal_activity_log", ["organization_id", "created_at"], unique=False)

    # ---------------------------------------------------------------------
    # RLS policies
    # ---------------------------------------------------------------------
    op.execute("ALTER TABLE goals ENABLE ROW LEVEL SECURITY")
    goals_visibility = _GOAL_VISIBILITY_CHECK_SQL.format(table="goals")
    op.execute(
        sa.text(
            f"""
            CREATE POLICY goals_visibility_policy ON goals
            USING ({goals_visibility})
            """
        )
    )

    for table in ("goal_nodes", "goal_edges", "goal_memory_links", "goal_activity_log"):
        op.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))

    g_visibility = _GOAL_VISIBILITY_CHECK_SQL.format(table="g")

    # goal_nodes: user must be able to see the parent goal
    op.execute(sa.text("DROP POLICY IF EXISTS goal_nodes_visibility_policy ON goal_nodes"))
    op.execute(
        sa.text(
            f"""
            CREATE POLICY goal_nodes_visibility_policy ON goal_nodes
            USING (
              goal_nodes.organization_id::text = current_setting('app.current_org_id', true)
              AND EXISTS (
                SELECT 1 FROM goals g
                WHERE g.id = goal_nodes.goal_id
                  AND g.organization_id = goal_nodes.organization_id
                  AND ({g_visibility})
              )
            )
            """
        )
    )

    # goal_edges: user must be able to see the from_node via its goal
    op.execute(sa.text("DROP POLICY IF EXISTS goal_edges_visibility_policy ON goal_edges"))
    op.execute(
        sa.text(
            f"""
            CREATE POLICY goal_edges_visibility_policy ON goal_edges
            USING (
              goal_edges.organization_id::text = current_setting('app.current_org_id', true)
              AND EXISTS (
                SELECT 1
                FROM goal_nodes n
                JOIN goals g ON g.id = n.goal_id
                WHERE n.id = goal_edges.from_node_id
                  AND n.organization_id = goal_edges.organization_id
                  AND g.organization_id = goal_edges.organization_id
                  AND ({g_visibility})
              )
            )
            """
        )
    )

    # goal_memory_links: user must be able to see the goal
    op.execute(sa.text("DROP POLICY IF EXISTS goal_memory_links_visibility_policy ON goal_memory_links"))
    op.execute(
        sa.text(
            f"""
            CREATE POLICY goal_memory_links_visibility_policy ON goal_memory_links
            USING (
              goal_memory_links.organization_id::text = current_setting('app.current_org_id', true)
              AND EXISTS (
                SELECT 1 FROM goals g
                WHERE g.id = goal_memory_links.goal_id
                  AND g.organization_id = goal_memory_links.organization_id
                  AND ({g_visibility})
              )
            )
            """
        )
    )

    # goal_activity_log: user must be able to see the goal
    op.execute(sa.text("DROP POLICY IF EXISTS goal_activity_log_visibility_policy ON goal_activity_log"))
    op.execute(
        sa.text(
            f"""
            CREATE POLICY goal_activity_log_visibility_policy ON goal_activity_log
            USING (
              goal_activity_log.organization_id::text = current_setting('app.current_org_id', true)
              AND EXISTS (
                SELECT 1 FROM goals g
                WHERE g.id = goal_activity_log.goal_id
                  AND g.organization_id = goal_activity_log.organization_id
                  AND ({g_visibility})
              )
            )
            """
        )
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS goal_activity_log_visibility_policy ON goal_activity_log")
    op.execute("DROP POLICY IF EXISTS goal_memory_links_visibility_policy ON goal_memory_links")
    op.execute("DROP POLICY IF EXISTS goal_edges_visibility_policy ON goal_edges")
    op.execute("DROP POLICY IF EXISTS goal_nodes_visibility_policy ON goal_nodes")
    op.execute("DROP POLICY IF EXISTS goals_visibility_policy ON goals")

    for table in ("goal_activity_log", "goal_memory_links", "goal_edges", "goal_nodes", "goals"):
        op.execute(sa.text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))

    op.drop_table("goal_activity_log")
    op.drop_table("goal_memory_links")
    op.drop_table("goal_edges")
    op.drop_index("ix_goal_nodes_goal_ordering", table_name="goal_nodes")
    op.drop_table("goal_nodes")
    op.drop_index("ix_goals_metadata_gin", table_name="goals")
    op.drop_index("ix_goals_tags_gin", table_name="goals")
    op.drop_index("ix_goals_org_due_at", table_name="goals")
    op.drop_index("ix_goals_org_status", table_name="goals")
    op.drop_table("goals")
