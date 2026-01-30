"""Add agent_run_events.summary_text + FTS index.

Revision ID: 2026_01_23_0002
Revises: 2026_01_23_0001
Create Date: 2026-01-23
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "2026_01_23_0002"
down_revision = "2026_01_23_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_run_events",
        sa.Column(
            "summary_text",
            sa.Text(),
            nullable=False,
            server_default=sa.text("''"),
        ),
    )

    # Full-text index for fast trajectory search.
    op.execute(
        "CREATE INDEX ix_agent_run_events_summary_fts ON agent_run_events USING gin (to_tsvector('simple', coalesce(summary_text,'')))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_run_events_summary_fts")
    op.drop_column("agent_run_events", "summary_text")
