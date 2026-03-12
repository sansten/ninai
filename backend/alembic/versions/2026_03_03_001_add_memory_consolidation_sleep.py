"""Add PR-2 consolidation sessions and memory arcs.

Revision ID: 003_add_memory_consolidation_sleep
Revises: 002_add_causal_edges
Create Date: 2026-03-03 09:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '20260303_pr2_sleep'
down_revision = '20260302_causal_edges'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "consolidation_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("session_type", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("operations", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("memory_quality_before", sa.Float(), nullable=True),
        sa.Column("memory_quality_after", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_consolidation_sessions_org_started",
        "consolidation_sessions",
        ["organization_id", "started_at"],
    )

    op.create_table(
        "memory_arcs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("measurements", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("trend", sa.String(32), nullable=False),
        sa.Column("trajectory_type", sa.String(64), nullable=False),
        sa.Column("prediction_next_access", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "idx_memory_arcs_org_memory",
        "memory_arcs",
        ["organization_id", "memory_id"],
    )


def downgrade():
    op.drop_index("idx_memory_arcs_org_memory", table_name="memory_arcs")
    op.drop_table("memory_arcs")

    op.drop_index("idx_consolidation_sessions_org_started", table_name="consolidation_sessions")
    op.drop_table("consolidation_sessions")
