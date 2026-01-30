"""Add memory_logseq_exports table

Revision ID: c1d2e3f4a5b6
Revises: b1c2d3e4f5a6
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "b1c2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_logseq_exports",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column(
            "graph",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("agent_version", sa.String(length=50), nullable=True),
        sa.Column("trace_id", sa.String(length=255), nullable=True),
        sa.Column("created_by", sa.String(length=50), nullable=False, server_default=sa.text("'agent'")),
        sa.Column("updated_by_user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_memory_logseq_exports_organization_id"), "memory_logseq_exports", ["organization_id"], unique=False)
    op.create_index(op.f("ix_memory_logseq_exports_memory_id"), "memory_logseq_exports", ["memory_id"], unique=False)
    op.create_index("ix_memory_logseq_exports_lookup", "memory_logseq_exports", ["organization_id", "memory_id"], unique=False)
    op.create_index("ux_memory_logseq_exports_org_memory", "memory_logseq_exports", ["organization_id", "memory_id"], unique=True)

    op.execute("ALTER TABLE memory_logseq_exports ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_logseq_exports FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY memory_logseq_exports_isolation
        ON memory_logseq_exports
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS memory_logseq_exports_isolation ON memory_logseq_exports")
    op.drop_index("ux_memory_logseq_exports_org_memory", table_name="memory_logseq_exports")
    op.drop_index("ix_memory_logseq_exports_lookup", table_name="memory_logseq_exports")
    op.drop_index(op.f("ix_memory_logseq_exports_memory_id"), table_name="memory_logseq_exports")
    op.drop_index(op.f("ix_memory_logseq_exports_organization_id"), table_name="memory_logseq_exports")
    op.drop_table("memory_logseq_exports")
