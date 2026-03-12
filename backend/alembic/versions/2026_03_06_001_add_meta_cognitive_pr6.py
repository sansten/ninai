"""add_meta_cognitive_pr6

PR-6 Meta-Cognitive Planning tables for strategy allocation and epistemic state tracking.

Revision ID: 2026_03_06_001_add_meta_cognitive_pr6
Revises: 2026_03_05_001_add_temporal_reasoning_pr5
Create Date: 2026-03-06 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260306_pr6_metacog"
down_revision = "20260305_pr5_temporal"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cognitive_strategies",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("query_id", sa.String(255), nullable=False),
        sa.Column("complexity_estimated", sa.Float(), server_default="0.5"),
        sa.Column("strategy_selected", sa.String(50), nullable=False),
        sa.Column("retrieval_budget", sa.Integer(), server_default="20"),
        sa.Column("reasoning_depth", sa.Integer(), server_default="2"),
        sa.Column("verification_required", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("confidence_threshold", sa.Float(), server_default="0.7"),
        sa.Column("time_budget_seconds", sa.Integer(), server_default="30"),
        sa.Column("expected_answer_quality", sa.Float(), server_default="0.7"),
        sa.Column("started_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(), nullable=True),
        sa.Column("actual_confidence", sa.Float(), nullable=True),
        sa.Column("strategy_effectiveness", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_cognitive_strategies_org_query",
        "cognitive_strategies",
        ["organization_id", "query_id"],
    )
    op.create_index(
        "idx_cognitive_strategies_org_strategy",
        "cognitive_strategies",
        ["organization_id", "strategy_selected"],
    )

    op.create_table(
        "epistemic_states",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("known_domains", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("uncertain_domains", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("unknown_domains", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("confidence_calibration", sa.Float(), nullable=True),
        sa.Column("surprise_frequency", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_epistemic_states_org_timestamp",
        "epistemic_states",
        ["organization_id", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("idx_epistemic_states_org_timestamp", table_name="epistemic_states")
    op.drop_index("idx_cognitive_strategies_org_strategy", table_name="cognitive_strategies")
    op.drop_index("idx_cognitive_strategies_org_query", table_name="cognitive_strategies")

    op.drop_table("epistemic_states")
    op.drop_table("cognitive_strategies")
