"""Add playbooks table (PR4)

Revision ID: 20260228_playbooks
Revises: 20260228_facts
Create Date: 2026-02-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260228_playbooks"
down_revision: Union[str, None] = "20260228_facts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "playbooks",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scope_type", sa.String(length=30), nullable=False, server_default="personal"),
        sa.Column("scope_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("problem_signature", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("signature_hash", sa.String(length=64), nullable=False),
        sa.Column("steps", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("constraints", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("success_rate", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_playbooks_organization_id", "playbooks", ["organization_id"], unique=False)
    op.create_index("ix_playbooks_scope_type", "playbooks", ["scope_type"], unique=False)
    op.create_index("ix_playbooks_scope_id", "playbooks", ["scope_id"], unique=False)
    op.create_index("ix_playbooks_signature_hash", "playbooks", ["signature_hash"], unique=False)
    op.create_index("ix_playbooks_org_scope", "playbooks", ["organization_id", "scope_type"], unique=False)
    op.create_index("ix_playbooks_org_signature", "playbooks", ["organization_id", "signature_hash"], unique=False)

    op.execute("ALTER TABLE playbooks ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY playbooks_org_isolation ON playbooks
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS playbooks_org_isolation ON playbooks")
    op.drop_index("ix_playbooks_org_signature", table_name="playbooks")
    op.drop_index("ix_playbooks_org_scope", table_name="playbooks")
    op.drop_index("ix_playbooks_signature_hash", table_name="playbooks")
    op.drop_index("ix_playbooks_scope_id", table_name="playbooks")
    op.drop_index("ix_playbooks_scope_type", table_name="playbooks")
    op.drop_index("ix_playbooks_organization_id", table_name="playbooks")
    op.drop_table("playbooks")
