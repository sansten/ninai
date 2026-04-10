"""add org saas fields: status, pinned_api_version, signup_ref

Revision ID: 2026_04_10_001
Revises: 20260404_cognitive_os
Create Date: 2026-04-10
"""
from alembic import op
import sqlalchemy as sa

revision = "2026_04_10_001"
down_revision = "20260404_cognitive_os"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("status", sa.String(30), nullable=False, server_default="active"),
    )
    op.add_column(
        "organizations",
        sa.Column("pinned_api_version", sa.String(10), nullable=True),
    )
    op.add_column(
        "organizations",
        sa.Column("signup_ref", sa.String(100), nullable=True),
    )
    op.create_index("ix_organizations_status", "organizations", ["status"])


def downgrade() -> None:
    op.drop_index("ix_organizations_status", table_name="organizations")
    op.drop_column("organizations", "signup_ref")
    op.drop_column("organizations", "pinned_api_version")
    op.drop_column("organizations", "status")
