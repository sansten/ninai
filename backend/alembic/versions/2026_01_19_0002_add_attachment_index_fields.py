"""Add attachment indexing fields

Revision ID: 4aa2f0c6a72e
Revises: 8c1f7b2d8b3a
Create Date: 2026-01-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "4aa2f0c6a72e"
down_revision: Union[str, None] = "8c1f7b2d8b3a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("memory_attachments", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("memory_attachments", sa.Column("index_error", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("memory_attachments", "index_error")
    op.drop_column("memory_attachments", "indexed_at")
