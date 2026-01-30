"""Add API keys and outgoing webhooks.

Revision ID: 2026_01_22_1200
Revises: 2026_01_22_0001
Create Date: 2026-01-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_22_1200"
down_revision = "2026_01_22_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
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
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("prefix", sa.String(length=24), nullable=False),
        sa.Column("key_hash", sa.String(length=255), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_api_keys_organization_id", "api_keys", ["organization_id"], unique=False)
    op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"], unique=False)
    op.create_index("ix_api_keys_prefix", "api_keys", ["prefix"], unique=False)
    op.create_index("ux_api_keys_org_name", "api_keys", ["organization_id", "name"], unique=True)

    op.create_table(
        "webhook_subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("url", sa.String(length=2000), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "event_types",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("secret_encrypted", sa.Text(), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
    )
    op.create_index(
        "ix_webhook_subscriptions_organization_id",
        "webhook_subscriptions",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_subscriptions_org_active",
        "webhook_subscriptions",
        ["organization_id", "is_active"],
        unique=False,
    )

    op.create_table(
        "webhook_outbox_events",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=200), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index(
        "ix_webhook_outbox_events_organization_id",
        "webhook_outbox_events",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_outbox_events_event_type",
        "webhook_outbox_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_outbox_org_type",
        "webhook_outbox_events",
        ["organization_id", "event_type"],
        unique=False,
    )

    op.create_table(
        "webhook_deliveries",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "subscription_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("webhook_subscriptions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "outbox_event_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("webhook_outbox_events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_http_status", sa.Integer(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_webhook_deliveries_organization_id",
        "webhook_deliveries",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_deliveries_subscription_id",
        "webhook_deliveries",
        ["subscription_id"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_deliveries_outbox_event_id",
        "webhook_deliveries",
        ["outbox_event_id"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_deliveries_status",
        "webhook_deliveries",
        ["status"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_deliveries_due",
        "webhook_deliveries",
        ["status", "next_attempt_at"],
        unique=False,
    )
    op.create_index(
        "ix_webhook_deliveries_org_sub",
        "webhook_deliveries",
        ["organization_id", "subscription_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_webhook_deliveries_org_sub", table_name="webhook_deliveries")
    op.drop_index("ix_webhook_deliveries_due", table_name="webhook_deliveries")
    op.drop_index("ix_webhook_deliveries_status", table_name="webhook_deliveries")
    op.drop_index("ix_webhook_deliveries_outbox_event_id", table_name="webhook_deliveries")
    op.drop_index("ix_webhook_deliveries_subscription_id", table_name="webhook_deliveries")
    op.drop_index("ix_webhook_deliveries_organization_id", table_name="webhook_deliveries")
    op.drop_table("webhook_deliveries")

    op.drop_index("ix_webhook_outbox_org_type", table_name="webhook_outbox_events")
    op.drop_index("ix_webhook_outbox_events_event_type", table_name="webhook_outbox_events")
    op.drop_index("ix_webhook_outbox_events_organization_id", table_name="webhook_outbox_events")
    op.drop_table("webhook_outbox_events")

    op.drop_index("ix_webhook_subscriptions_org_active", table_name="webhook_subscriptions")
    op.drop_index("ix_webhook_subscriptions_organization_id", table_name="webhook_subscriptions")
    op.drop_table("webhook_subscriptions")

    op.drop_index("ux_api_keys_org_name", table_name="api_keys")
    op.drop_index("ix_api_keys_prefix", table_name="api_keys")
    op.drop_index("ix_api_keys_user_id", table_name="api_keys")
    op.drop_index("ix_api_keys_organization_id", table_name="api_keys")
    op.drop_table("api_keys")
