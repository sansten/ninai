"""add org_subscriptions table

Revision ID: 20260407_001
Revises: 20260404_cognitive_os
Create Date: 2026-04-07 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260407_001"
down_revision = "20260404_cognitive_os"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "org_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("plan", sa.String(50), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="active"),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(255), nullable=True),
        sa.Column("seat_limit", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("seat_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("trial_ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("license_token", sa.String(2000), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", name="uq_org_subscriptions_org_id"),
    )
    op.create_index("ix_org_subscriptions_org_id", "org_subscriptions", ["organization_id"])
    op.create_index("ix_org_subscriptions_stripe_customer", "org_subscriptions", ["stripe_customer_id"])
    op.create_index("ix_org_subscriptions_stripe_sub", "org_subscriptions", ["stripe_subscription_id"])


def downgrade() -> None:
    op.drop_index("ix_org_subscriptions_stripe_sub", table_name="org_subscriptions")
    op.drop_index("ix_org_subscriptions_stripe_customer", table_name="org_subscriptions")
    op.drop_index("ix_org_subscriptions_org_id", table_name="org_subscriptions")
    op.drop_table("org_subscriptions")
