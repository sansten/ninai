"""add_benchmark_runs table

Adds benchmark_runs table for persisting COSE benchmark run results.

Revision ID: 20260403_benchmark_runs
Revises: 20260306_pr6_metacog
Create Date: 2026-04-03 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260403_benchmark_runs"
down_revision = "20260306_pr6_metacog"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mode", sa.String(64), nullable=False),
        sa.Column("strategy", sa.String(64), nullable=False),
        sa.Column("dataset", sa.String(64), nullable=False),
        sa.Column("ollama_model", sa.String(128), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("composite_score", sa.Float(), nullable=False),
        sa.Column("results", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_benchmark_runs_run_at", "benchmark_runs", ["run_at"])


def downgrade() -> None:
    op.drop_index("ix_benchmark_runs_run_at", table_name="benchmark_runs")
    op.drop_table("benchmark_runs")
