"""enforce row-level security on all tenant tables

Revision ID: 2026_04_10_002
Revises: 2026_04_10_001
Create Date: 2026-04-10
"""
from alembic import op
import sqlalchemy as sa

revision = "2026_04_10_002"
down_revision = "2026_04_10_001"
branch_labels = None
depends_on = None

# Every table that stores per-tenant rows.
# If you add a new table with organization_id, add it here.
TENANT_TABLES = [
    "memories",
    "knowledge_items",
    "knowledge_item_versions",
    "goals",
    "goal_hierarchy_nodes",
    "episodes",
    "episode_events",
    "episode_links",
    "agent_runs",
    "agent_run_events",
    "agent_processes",
    "audit_logs",
    "users",
    "user_roles",
    "roles",
    "api_keys",
    "webhooks",
    "org_subscriptions",
    "org_feature_flags",
    "org_llm_configs",
    "playbooks",
    "events",
    "backups",
    "snapshots",
    "causal_edges",
    "ontology_entries",
    "social_graph_edges",
    "strategy_library",
    "teams",
]


def upgrade() -> None:
    conn = op.get_bind()
    for table in TENANT_TABLES:
        # Check table exists before touching it — graceful if a table was removed
        result = conn.execute(
            sa.text("SELECT to_regclass(:tname)"),
            {"tname": f"public.{table}"},
        ).scalar()
        if result is None:
            continue

        conn.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        conn.execute(sa.text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

        # Drop policy if it already exists (idempotent re-run)
        conn.execute(
            sa.text(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        )
        conn.execute(
            sa.text(f"""
                CREATE POLICY tenant_isolation ON {table}
                  USING (
                    organization_id = current_setting('app.current_org_id', TRUE)::uuid
                  )
            """)
        )


def downgrade() -> None:
    conn = op.get_bind()
    for table in TENANT_TABLES:
        result = conn.execute(
            sa.text("SELECT to_regclass(:tname)"), {"tname": f"public.{table}"}
        ).scalar()
        if result is None:
            continue
        conn.execute(sa.text(f"DROP POLICY IF EXISTS tenant_isolation ON {table}"))
        conn.execute(sa.text(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY"))
