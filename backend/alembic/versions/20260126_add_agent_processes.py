"""add agent processes table

Revision ID: 20260126_add_agent_processes
Revises: 
Create Date: 2026-01-26
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260126_add_agent_processes"
down_revision = "2026_01_25_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_processes",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False, index=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("agent_run_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("agent_name", sa.String(length=255), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("quota_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("quota_storage_mb", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=False, server_default=""),
        sa.Column("process_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index(
        "ix_agent_processes_org_status_prio",
        "agent_processes",
        ["organization_id", "status", "priority", "created_at"],
    )
    op.create_index(
        "ix_agent_processes_session_lookup",
        "agent_processes",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_processes_session_lookup", table_name="agent_processes")
    op.drop_index("ix_agent_processes_org_status_prio", table_name="agent_processes")
    op.drop_table("agent_processes")
