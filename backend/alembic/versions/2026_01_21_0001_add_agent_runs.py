"""Add agent_runs table

Revision ID: 9f2d0e3b4c1a
Revises: 9f0c2a9c4b11
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "9f2d0e3b4c1a"
down_revision: Union[str, None] = "9f0c2a9c4b11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("agent_name", sa.String(length=255), nullable=False),
        sa.Column("agent_version", sa.String(length=50), nullable=False),
        sa.Column("inputs_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("outputs", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("warnings", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("errors", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_agent_runs_organization_id"), "agent_runs", ["organization_id"], unique=False)
    op.create_index(op.f("ix_agent_runs_memory_id"), "agent_runs", ["memory_id"], unique=False)
    op.create_index("ix_agent_runs_lookup", "agent_runs", ["organization_id", "memory_id", "agent_name"], unique=False)
    op.create_index(
        "ux_agent_runs_idempotency",
        "agent_runs",
        ["organization_id", "memory_id", "agent_name", "agent_version"],
        unique=True,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE agent_runs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_runs FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_agent_runs ON agent_runs
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_agent_runs ON agent_runs")
    op.drop_index("ux_agent_runs_idempotency", table_name="agent_runs")
    op.drop_index("ix_agent_runs_lookup", table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_memory_id"), table_name="agent_runs")
    op.drop_index(op.f("ix_agent_runs_organization_id"), table_name="agent_runs")
    op.drop_table("agent_runs")
