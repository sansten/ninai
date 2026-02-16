"""
Migration for Phase 7: Event Publishing, Webhooks, and Snapshots

Adds tables for:
- events (system events for webhooks)
- webhook_subscriptions (webhook endpoints)
- snapshots (exports and backups)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260127_add_event_publishing'
down_revision = '02c1596033c4'
branch_labels = None
depends_on = None


def upgrade():
    # Create events table
    op.create_table(
        'events',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('event_type', sa.String(128), nullable=False),
        sa.Column('event_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('resource_type', sa.String(64), nullable=False),
        sa.Column('resource_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('payload', postgresql.JSON(), nullable=True),
        sa.Column('actor_user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('actor_agent_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('trace_id', sa.String(256), nullable=True),
        sa.Column('request_id', sa.String(256), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_event_org_type', 'events', ['organization_id', 'event_type'])
    op.create_index('idx_event_resource', 'events', ['organization_id', 'resource_type', 'resource_id'])
    op.create_index('ix_events_created_at', 'events', ['created_at'])
    op.create_index('ix_events_event_type', 'events', ['event_type'])
    op.create_index('ix_events_organization_id', 'events', ['organization_id'])
    op.create_index('ix_events_resource_id', 'events', ['resource_id'])

    # Create webhook_subscriptions table if not already present (earlier branch adds it)
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table('webhook_subscriptions'):
        op.create_table(
            'webhook_subscriptions',
            sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
            sa.Column('organization_id', postgresql.UUID(as_uuid=False), nullable=False),
            sa.Column('url', sa.String(512), nullable=False),
            sa.Column('event_types', sa.String(512), nullable=False),
            sa.Column('resource_types', sa.String(256), nullable=True),
            sa.Column('secret', sa.String(256), nullable=False),
            sa.Column('signing_algorithm', sa.String(32), nullable=False, server_default='sha256'),
            sa.Column('active', sa.Boolean(), nullable=False, server_default='true'),
            sa.Column('paused_at', sa.DateTime(), nullable=True),
            sa.Column('paused_reason', sa.Text(), nullable=True),
            sa.Column('max_retries', sa.Integer(), nullable=False, server_default='5'),
            sa.Column('retry_delay_seconds', sa.Integer(), nullable=False, server_default='60'),
            sa.Column('rate_limit_per_minute', sa.Integer(), nullable=True),
            sa.Column('delivered_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('failed_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('last_delivery_at', sa.DateTime(), nullable=True),
            sa.Column('last_error', sa.Text(), nullable=True),
            sa.Column('description', sa.Text(), nullable=True),
            sa.Column('custom_headers', postgresql.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column('created_by_user_id', postgresql.UUID(as_uuid=False), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index('idx_webhook_org_active', 'webhook_subscriptions', ['organization_id', 'active'])
        op.create_index('ix_webhook_subscriptions_created_at', 'webhook_subscriptions', ['created_at'])

    # Create snapshots table
    op.create_table(
        'snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(256), nullable=False),
        sa.Column('description', sa.String(512), nullable=True),
        sa.Column('format', sa.String(32), nullable=False),
        sa.Column('compression', sa.String(32), nullable=True),
        sa.Column('resource_type', sa.String(64), nullable=True),
        sa.Column('filters', postgresql.JSON(), nullable=True),
        sa.Column('storage_path', sa.String(512), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('item_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(32), nullable=False, server_default='pending'),
        sa.Column('progress_percent', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.String(512), nullable=True),
        sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('download_token', sa.String(256), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('checksum', sa.String(256), nullable=True),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_snapshot_org', 'snapshots', ['organization_id'])
    op.create_index('idx_snapshot_status', 'snapshots', ['status'])
    op.create_index('ix_snapshots_created_at', 'snapshots', ['created_at'])


def downgrade():
    op.drop_table('snapshots')
    op.drop_table('webhook_subscriptions')
    op.drop_table('events')
