"""Add memory_promotion_history table

Revision ID: e1f2a3b4c5d6
Revises: d4e5f6a7b8c9
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_promotion_history",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("from_stm_id", sa.String(length=255), nullable=False),
        sa.Column("to_memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("from_type", sa.String(length=50), nullable=False, server_default=sa.text("'short_term'")),
        sa.Column("to_type", sa.String(length=50), nullable=False, server_default=sa.text("'long_term'")),
        sa.Column("promotion_reason", sa.String(length=255), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
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
        sa.ForeignKeyConstraint(["to_memory_id"], ["memory_metadata.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_memory_promotion_history_organization_id"),
        "memory_promotion_history",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_memory_promotion_history_to_memory_id"),
        "memory_promotion_history",
        ["to_memory_id"],
        unique=False,
    )
    op.create_index(
        "ux_promotion_once",
        "memory_promotion_history",
        ["organization_id", "from_stm_id"],
        unique=True,
    )
    op.create_index(
        "ix_promotion_lookup",
        "memory_promotion_history",
        ["organization_id", "to_memory_id"],
        unique=False,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE memory_promotion_history ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_promotion_history FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_promotion_history ON memory_promotion_history
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_promotion_history ON memory_promotion_history")
    op.drop_index("ix_promotion_lookup", table_name="memory_promotion_history")
    op.drop_index("ux_promotion_once", table_name="memory_promotion_history")
    op.drop_index(op.f("ix_memory_promotion_history_to_memory_id"), table_name="memory_promotion_history")
    op.drop_index(op.f("ix_memory_promotion_history_organization_id"), table_name="memory_promotion_history")
    op.drop_table("memory_promotion_history")
