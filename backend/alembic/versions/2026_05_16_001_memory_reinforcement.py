"""add ingest reinforcement columns to memory_metadata

Revision ID: 2026_05_16_001
Revises: 2026_04_18_001
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision = "2026_05_16_001"
down_revision = "2026_04_18_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memory_metadata",
        sa.Column("ingest_count", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "memory_metadata",
        sa.Column("last_ingested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "memory_metadata",
        sa.Column(
            "unique_sources",
            ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade() -> None:
    op.drop_column("memory_metadata", "unique_sources")
    op.drop_column("memory_metadata", "last_ingested_at")
    op.drop_column("memory_metadata", "ingest_count")
