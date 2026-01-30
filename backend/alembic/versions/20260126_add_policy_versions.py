"""Add policy_versions table

Revision ID: 20260126_add_policy_versions
Revises: 20260126_add_resource_budgets
Create Date: 2026-01-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260126_add_policy_versions'
down_revision: Union[str, None] = '20260126_add_resource_budgets'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create policy_versions table
    op.create_table(
        'policy_versions',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('policy_name', sa.String(length=100), nullable=False),
        sa.Column('policy_type', sa.String(length=50), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('policy_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('validation_schema', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('rollout_status', sa.String(length=20), nullable=False),
        sa.Column('rollout_percentage', sa.Float(), nullable=False),
        sa.Column('canary_group_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('success_count', sa.Integer(), nullable=False),
        sa.Column('failure_count', sa.Integer(), nullable=False),
        sa.Column('error_rate', sa.Float(), nullable=False),
        sa.Column('activated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('superseded_by_version', sa.Integer(), nullable=True),
        sa.Column('rolled_back_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rollback_reason', sa.Text(), nullable=True),
        sa.Column('rolled_back_to_version', sa.Integer(), nullable=True),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('change_notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for efficient queries
    op.create_index('ix_policy_versions_organization_id', 'policy_versions', ['organization_id'], unique=False)
    op.create_index('ix_policy_versions_policy_name', 'policy_versions', ['policy_name'], unique=False)
    op.create_index('ix_policy_versions_policy_type', 'policy_versions', ['policy_type'], unique=False)
    op.create_index(
        'ix_policy_versions_org_name_version',
        'policy_versions',
        ['organization_id', 'policy_name', 'version'],
        unique=True
    )
    op.create_index(
        'ix_policy_versions_org_name_status',
        'policy_versions',
        ['organization_id', 'policy_name', 'rollout_status'],
        unique=False
    )


def downgrade() -> None:
    op.drop_index('ix_policy_versions_org_name_status', table_name='policy_versions')
    op.drop_index('ix_policy_versions_org_name_version', table_name='policy_versions')
    op.drop_index('ix_policy_versions_policy_type', table_name='policy_versions')
    op.drop_index('ix_policy_versions_policy_name', table_name='policy_versions')
    op.drop_index('ix_policy_versions_organization_id', table_name='policy_versions')
    op.drop_table('policy_versions')
