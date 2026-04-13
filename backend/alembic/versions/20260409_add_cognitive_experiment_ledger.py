"""add cognitive_experiment_ledger table

Revision ID: 20260409_auto_research
Revises: 20260407_002
Create Date: 2026-04-09 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260409_auto_research"
down_revision = "20260407_002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cognitive_experiment_ledger",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("agent_name", sa.String(length=128), nullable=False),
        sa.Column("parameter_key", sa.String(length=128), nullable=False),
        sa.Column("baseline_value", sa.Float(), nullable=False),
        sa.Column("candidate_value", sa.Float(), nullable=False),
        sa.Column("baseline_score", sa.Float(), nullable=False),
        sa.Column("candidate_score", sa.Float(), nullable=False),
        sa.Column("score_delta", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="reverted"),
        sa.Column("benchmark_summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("rationale", sa.String(length=2048), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_cognitive_experiment_ledger_org_param",
        "cognitive_experiment_ledger",
        ["org_id", "parameter_key"],
    )
    op.create_index("ix_cognitive_experiment_ledger_org_id", "cognitive_experiment_ledger", ["org_id"])
    op.create_index(
        "ix_cognitive_experiment_ledger_agent_name",
        "cognitive_experiment_ledger",
        ["agent_name"],
    )
    op.create_index(
        "ix_cognitive_experiment_ledger_parameter_key",
        "cognitive_experiment_ledger",
        ["parameter_key"],
    )
    op.create_index("ix_cognitive_experiment_ledger_status", "cognitive_experiment_ledger", ["status"])
    op.create_index(
        "ix_cognitive_experiment_ledger_created_at",
        "cognitive_experiment_ledger",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_cognitive_experiment_ledger_created_at", table_name="cognitive_experiment_ledger")
    op.drop_index("ix_cognitive_experiment_ledger_status", table_name="cognitive_experiment_ledger")
    op.drop_index("ix_cognitive_experiment_ledger_parameter_key", table_name="cognitive_experiment_ledger")
    op.drop_index("ix_cognitive_experiment_ledger_agent_name", table_name="cognitive_experiment_ledger")
    op.drop_index("ix_cognitive_experiment_ledger_org_id", table_name="cognitive_experiment_ledger")
    op.drop_index("ix_cognitive_experiment_ledger_org_param", table_name="cognitive_experiment_ledger")
    op.drop_table("cognitive_experiment_ledger")