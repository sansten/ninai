"""add usage_events table

Revision ID: 20260407_002
Revises: 20260407_001
Create Date: 2026-04-07 01:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260407_002"
down_revision = "20260407_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("metric", sa.String(length=100), nullable=False),
        sa.Column("value", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "date", "metric", name="uq_usage_event_org_date_metric"),
    )
    op.create_index("ix_usage_events_org_id", "usage_events", ["organization_id"])
    op.create_index("ix_usage_events_metric", "usage_events", ["metric"])
    op.create_index("ix_usage_events_date", "usage_events", ["date"])


def downgrade() -> None:
    op.drop_index("ix_usage_events_date", table_name="usage_events")
    op.drop_index("ix_usage_events_metric", table_name="usage_events")
    op.drop_index("ix_usage_events_org_id", table_name="usage_events")
    op.drop_table("usage_events")
