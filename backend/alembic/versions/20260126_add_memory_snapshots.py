"""Add memory_snapshots table

Revision ID: 20260126_add_memory_snapshots
Revises: 20260126_add_policy_versions
Create Date: 2026-01-26 12:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260126_add_memory_snapshots'
down_revision: Union[str, None] = '20260126_add_policy_versions'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create memory_snapshots table
    op.create_table(
        'memory_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('snapshot_name', sa.String(length=100), nullable=False),
        sa.Column('snapshot_type', sa.String(length=20), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('snapshot_size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('memory_count', sa.BigInteger(), nullable=False),
        sa.Column('embedding_count', sa.BigInteger(), nullable=False),
        sa.Column('snapshot_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('storage_location', sa.Text(), nullable=True),
        sa.Column('compression_format', sa.String(length=20), nullable=True),
        sa.Column('parent_snapshot_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('retention_days', sa.BigInteger(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('failed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('verified', sa.Boolean(), nullable=False),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('checksum', sa.String(length=64), nullable=True),
        sa.Column('replicated', sa.Boolean(), nullable=False),
        sa.Column('replication_targets', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for efficient queries
    op.create_index('ix_memory_snapshots_organization_id', 'memory_snapshots', ['organization_id'], unique=False)
    op.create_index('ix_memory_snapshots_snapshot_name', 'memory_snapshots', ['snapshot_name'], unique=False)
    op.create_index('ix_memory_snapshots_snapshot_type', 'memory_snapshots', ['snapshot_type'], unique=False)
    op.create_index('ix_memory_snapshots_status', 'memory_snapshots', ['status'], unique=False)
    op.create_index('ix_memory_snapshots_org_status', 'memory_snapshots', ['organization_id', 'status'], unique=False)
    op.create_index('ix_memory_snapshots_org_type', 'memory_snapshots', ['organization_id', 'snapshot_type'], unique=False)
    op.create_index('ix_memory_snapshots_parent', 'memory_snapshots', ['parent_snapshot_id'], unique=False)
    op.create_index('ix_memory_snapshots_expires', 'memory_snapshots', ['expires_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_memory_snapshots_expires', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_parent', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_org_type', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_org_status', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_status', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_snapshot_type', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_snapshot_name', table_name='memory_snapshots')
    op.drop_index('ix_memory_snapshots_organization_id', table_name='memory_snapshots')
    op.drop_table('memory_snapshots')
