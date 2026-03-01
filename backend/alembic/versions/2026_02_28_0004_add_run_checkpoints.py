"""Add run_checkpoints table (PR5: Replayability)

Revision ID: 20260228_checkpoints
Revises: 20260228_playbooks
Create Date: 2026-02-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260228_checkpoints"
down_revision: Union[str, None] = "20260228_playbooks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "run_checkpoints",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("input_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("retrieval_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("model_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("output_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_run_checkpoints_org_run", "run_checkpoints", ["organization_id", "agent_run_id"], unique=False)
    op.create_index("ix_run_checkpoints_lookup", "run_checkpoints", ["agent_run_id", "step_index"], unique=False)

    op.execute("ALTER TABLE run_checkpoints ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY run_checkpoints_org_isolation ON run_checkpoints
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS run_checkpoints_org_isolation ON run_checkpoints")
    op.drop_index("ix_run_checkpoints_lookup", table_name="run_checkpoints")
    op.drop_index("ix_run_checkpoints_org_run", table_name="run_checkpoints")
    op.drop_table("run_checkpoints")
