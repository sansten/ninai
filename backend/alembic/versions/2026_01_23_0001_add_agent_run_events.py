"""Add agent_run_events table.

Revision ID: 2026_01_23_0001
Revises: 2026_01_22_1300
Create Date: 2026-01-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "2026_01_23_0001"
down_revision: Union[str, None] = "2026_01_22_1300"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_run_events",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("agent_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_agent_run_events_organization_id"), "agent_run_events", ["organization_id"], unique=False)
    op.create_index(op.f("ix_agent_run_events_agent_run_id"), "agent_run_events", ["agent_run_id"], unique=False)
    op.create_index(op.f("ix_agent_run_events_memory_id"), "agent_run_events", ["memory_id"], unique=False)
    op.create_index(
        "ix_agent_run_events_lookup",
        "agent_run_events",
        ["organization_id", "agent_run_id", "step_index"],
        unique=False,
    )
    op.create_index(
        "ix_agent_run_events_memory_lookup",
        "agent_run_events",
        ["organization_id", "memory_id", "created_at"],
        unique=False,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE agent_run_events ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_run_events FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_agent_run_events ON agent_run_events
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_agent_run_events ON agent_run_events")
    op.drop_index("ix_agent_run_events_memory_lookup", table_name="agent_run_events")
    op.drop_index("ix_agent_run_events_lookup", table_name="agent_run_events")
    op.drop_index(op.f("ix_agent_run_events_memory_id"), table_name="agent_run_events")
    op.drop_index(op.f("ix_agent_run_events_agent_run_id"), table_name="agent_run_events")
    op.drop_index(op.f("ix_agent_run_events_organization_id"), table_name="agent_run_events")
    op.drop_table("agent_run_events")
