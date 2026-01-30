"""Fix admin_settings column types

Revision ID: 2026_01_27_0004_fix_admin_settings_types
Revises: 2026_01_27_0003_add_backup_models
Create Date: 2026-01-27 22:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fix_admin_settings_types'
down_revision: Union[str, None] = 'add_fts_hybrid_search'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fix admin_settings id and updated_by column types from VARCHAR to UUID."""
    
    # Drop and recreate the table with correct types
    op.drop_table('admin_settings')
    
    op.create_table(
        "admin_settings",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_secret", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", postgresql.UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("category", "key", name="uq_admin_settings_category_key"),
    )
    op.create_index("ix_admin_settings_category", "admin_settings", ["category"], unique=False)
    op.create_index("ix_admin_settings_category_key", "admin_settings", ["category", "key"], unique=False)
    op.create_index("ix_admin_settings_updated_by", "admin_settings", ["updated_by"], unique=False)


def downgrade() -> None:
    """Revert to original schema."""
    
    op.drop_table('admin_settings')
    
    # Recreate with UUID types (original schema from admin_ui_foundation)
    op.create_table(
        "admin_settings",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_secret", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", postgresql.UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("category", "key"),
    )
    op.create_index("ix_admin_settings_category", "admin_settings", ["category"], unique=False)
    op.create_index("ix_admin_settings_category_key", "admin_settings", ["category", "key"], unique=False)
    op.create_index("ix_admin_settings_updated_by", "admin_settings", ["updated_by"], unique=False)
