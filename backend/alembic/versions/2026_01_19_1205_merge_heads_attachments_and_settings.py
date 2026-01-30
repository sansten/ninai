"""Merge heads: attachments and app_settings

Revision ID: 9f0c2a9c4b11
Revises: 4aa2f0c6a72e, 4d3a9d9e1c2b
Create Date: 2026-01-19
"""

from typing import Sequence, Union

# Alembic revision identifiers
revision: str = "9f0c2a9c4b11"
down_revision: Union[str, Sequence[str], None] = ("4aa2f0c6a72e", "4d3a9d9e1c2b")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Merge revision: no schema changes.
    pass


def downgrade() -> None:
    # Merge revision: no schema changes.
    pass
