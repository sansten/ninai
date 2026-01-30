"""
Alembic Migration Script Template
=================================

This is the Mako template for generating new migration scripts.
"""

"""add_app_settings_key_index

Revision ID: 22a29f3b8904
Revises: 20260126_add_memory_snapshots
Create Date: 2026-01-26 22:42:41.293903+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "22a29f3b8904"
down_revision: Union[str, None] = "20260126_add_memory_snapshots"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Add index on app_settings.key for faster admin config lookups
    op.create_index(
        "ix_app_settings_key",
        "app_settings",
        ["key"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("ix_app_settings_key", table_name="app_settings")
