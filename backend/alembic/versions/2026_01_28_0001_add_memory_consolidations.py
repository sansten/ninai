"""Add memory_consolidations table

Revision ID: 20260128_mem_consol
Revises: fix_admin_settings_types
Create Date: 2026-01-28

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260128_mem_consol"
down_revision: Union[str, None] = "fix_admin_settings_types"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_consolidations",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "consolidated_memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "source_memory_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("created_by", sa.String(length=50), nullable=False, server_default="manual"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_memory_consolidations_organization_id",
        "memory_consolidations",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_consolidations_user_id",
        "memory_consolidations",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_consolidations_consolidated_memory_id",
        "memory_consolidations",
        ["consolidated_memory_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_consolidations_org_created_at",
        "memory_consolidations",
        ["organization_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_memory_consolidations_org_created_at", table_name="memory_consolidations")
    op.drop_index("ix_memory_consolidations_consolidated_memory_id", table_name="memory_consolidations")
    op.drop_index("ix_memory_consolidations_user_id", table_name="memory_consolidations")
    op.drop_index("ix_memory_consolidations_organization_id", table_name="memory_consolidations")
    op.drop_table("memory_consolidations")
