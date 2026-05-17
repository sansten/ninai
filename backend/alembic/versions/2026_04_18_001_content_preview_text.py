"""extend content_preview from varchar(500) to text

Revision ID: 2026_04_18_001
Revises: 2026_04_15_001_actor_identity_policy
Create Date: 2026-04-18
"""
from alembic import op
import sqlalchemy as sa

revision = "2026_04_18_001"
down_revision = "2026_04_15_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "memory_metadata",
        "content_preview",
        type_=sa.Text(),
        existing_type=sa.String(500),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "memory_metadata",
        "content_preview",
        type_=sa.String(500),
        existing_type=sa.Text(),
        existing_nullable=False,
        postgresql_using="content_preview::varchar(500)",
    )
