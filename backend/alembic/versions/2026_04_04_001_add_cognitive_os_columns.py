"""add cognitive OS columns and system_cognition_states table

Adds:
  - cognitive_sessions.is_autonomous (bool, default False)
  - system_cognition_states table (per-org live cognitive state)

Revision ID: 20260404_cognitive_os
Revises: 20260403_benchmark_runs
Create Date: 2026-04-04 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260404_cognitive_os"
down_revision = "20260403_benchmark_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_autonomous to cognitive_sessions
    op.add_column(
        "cognitive_sessions",
        sa.Column(
            "is_autonomous",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_cognitive_sessions_autonomy",
        "cognitive_sessions",
        ["organization_id", "is_autonomous"],
    )

    # Create system_cognition_states table
    op.create_table(
        "system_cognition_states",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("current_focus", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "active_session_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("unresolved_anomalies_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stale_goals_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cognitive_load", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_scheduled_action", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", name="uq_system_cognition_state_org"),
    )
    op.create_index(
        "ix_system_cognition_states_org_id",
        "system_cognition_states",
        ["organization_id"],
    )
    op.create_index(
        "ix_system_cognition_states_updated_at",
        "system_cognition_states",
        ["updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_system_cognition_states_updated_at", table_name="system_cognition_states")
    op.drop_index("ix_system_cognition_states_org_id", table_name="system_cognition_states")
    op.drop_table("system_cognition_states")
    op.drop_index("ix_cognitive_sessions_autonomy", table_name="cognitive_sessions")
    op.drop_column("cognitive_sessions", "is_autonomous")
