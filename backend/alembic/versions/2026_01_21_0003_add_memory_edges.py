"""Add memory_edges table

Revision ID: d4e5f6a7b8c9
Revises: c3a1d2e4f5b6
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3a1d2e4f5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_edges",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("from_node", sa.String(length=512), nullable=False),
        sa.Column("to_node", sa.String(length=512), nullable=False),
        sa.Column("relation", sa.String(length=128), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False, server_default=sa.text("1")),
        sa.Column("explanation", sa.String(length=1000), nullable=True),
        sa.Column("created_by", sa.String(length=50), nullable=False, server_default=sa.text("'agent'")),
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

    op.create_index(op.f("ix_memory_edges_organization_id"), "memory_edges", ["organization_id"], unique=False)
    op.create_index(op.f("ix_memory_edges_memory_id"), "memory_edges", ["memory_id"], unique=False)
    op.create_index(
        "ix_memory_edges_lookup",
        "memory_edges",
        ["organization_id", "memory_id", "relation"],
        unique=False,
    )
    op.create_index(
        "ux_memory_edges_dedupe",
        "memory_edges",
        ["organization_id", "memory_id", "from_node", "to_node", "relation"],
        unique=True,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE memory_edges ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_edges FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_edges ON memory_edges
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_edges ON memory_edges")
    op.drop_index("ux_memory_edges_dedupe", table_name="memory_edges")
    op.drop_index("ix_memory_edges_lookup", table_name="memory_edges")
    op.drop_index(op.f("ix_memory_edges_memory_id"), table_name="memory_edges")
    op.drop_index(op.f("ix_memory_edges_organization_id"), table_name="memory_edges")
    op.drop_table("memory_edges")
