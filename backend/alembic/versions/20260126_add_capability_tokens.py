"""Add Capability Tokens table - Phase 2 Memory Syscall Surface

Revision ID: 20260126_add_capability_tokens
Revises: 20260126_add_agent_processes
Create Date: 2026-01-26 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260126_add_capability_tokens'
down_revision = '20260126_add_agent_processes'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create capability_tokens table
    op.create_table(
        'capability_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('token', sa.String(60), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('scopes', sa.String(256), nullable=False, server_default='read'),
        sa.Column('quota_tokens_per_month', sa.Integer(), nullable=False, server_default='1000000'),
        sa.Column('quota_storage_bytes', sa.Integer(), nullable=False, server_default='104857600'),
        sa.Column('quota_requests_per_minute', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('tokens_used_this_month', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('storage_used_bytes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('requests_this_minute', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_request_at', sa.DateTime(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('quota_exceeded', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('revocation_reason', sa.Text(), nullable=True),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('revoked_by_user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('token_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token')
    )

    # Create indices
    op.create_index('idx_capability_token_org_active', 'capability_tokens', ['organization_id', 'active'])
    op.create_index('idx_capability_token_agent', 'capability_tokens', ['organization_id', 'session_id'])
    op.create_index('idx_capability_token_expires', 'capability_tokens', ['expires_at'])


def downgrade() -> None:
    op.drop_index('idx_capability_token_expires', table_name='capability_tokens')
    op.drop_index('idx_capability_token_agent', table_name='capability_tokens')
    op.drop_index('idx_capability_token_org_active', table_name='capability_tokens')
    op.drop_table('capability_tokens')
