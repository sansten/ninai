"""add_strategy_library

Outcome-Driven Strategy Learning table.
Stores per-org EMA-smoothed success rates for (goal_type, domain) strategy fingerprints.

Revision ID: 20260307_strategy_library
Revises: 20260307_pr7_comp_gen
Create Date: 2026-03-07 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260307_strategy_library"
down_revision = "20260307_pr7_comp_gen"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "strategy_library_entries",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("goal_type", sa.String(100), nullable=False),
        sa.Column("domain", sa.String(100), nullable=False, server_default="general"),
        sa.Column(
            "tool_sequence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "evidence_pattern",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("avg_plan_confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("avg_iterations", sa.Float(), nullable=False, server_default="2.0"),
        sa.Column("success_rate", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "last_updated",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id", "goal_type", "domain",
            name="uq_strategy_library_org_goaltype_domain",
        ),
    )
    op.create_index(
        "ix_strategy_library_entries_org_id",
        "strategy_library_entries",
        ["organization_id"],
    )
    op.create_index(
        "ix_strategy_library_entries_org_goaltype_domain",
        "strategy_library_entries",
        ["organization_id", "goal_type", "domain"],
    )


def downgrade() -> None:
    op.drop_index("ix_strategy_library_entries_org_goaltype_domain", table_name="strategy_library_entries")
    op.drop_index("ix_strategy_library_entries_org_id", table_name="strategy_library_entries")
    op.drop_table("strategy_library_entries")
