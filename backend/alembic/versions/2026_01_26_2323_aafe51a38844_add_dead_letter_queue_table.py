"""
Alembic Migration Script Template
=================================

This is the Mako template for generating new migration scripts.
"""

"""add_dead_letter_queue_table

Revision ID: aafe51a38844
Revises: 22a29f3b8904
Create Date: 2026-01-26 23:23:10.908638+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'aafe51a38844'
down_revision: Union[str, None] = '22a29f3b8904'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create dead_letter_queue table
    op.create_table(
        'dead_letter_queue',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=False),
        sa.Column('original_task_id', sa.UUID(), nullable=False),
        sa.Column('task_type', sa.String(50), nullable=False),
        sa.Column('failure_reason', sa.String(100), nullable=False),
        sa.Column('total_attempts', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('last_error', sa.Text(), nullable=True),
        sa.Column('error_pattern', sa.String(200), nullable=True),
        sa.Column('task_payload', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('quarantined_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('reviewed_by', sa.UUID(), nullable=True),
        sa.Column('resolution', sa.String(50), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('review_priority', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['original_task_id'], ['pipeline_tasks.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id'], ondelete='SET NULL'),
    )
    
    # Create indexes for efficient queries
    op.create_index('ix_dlq_organization_id', 'dead_letter_queue', ['organization_id'])
    op.create_index('ix_dlq_original_task_id', 'dead_letter_queue', ['original_task_id'])
    op.create_index('ix_dlq_task_type', 'dead_letter_queue', ['task_type'])
    op.create_index('ix_dlq_quarantined_at', 'dead_letter_queue', ['quarantined_at'])
    op.create_index('ix_dlq_is_resolved', 'dead_letter_queue', ['is_resolved'])
    
    # Composite index for common query patterns
    op.create_index(
        'ix_dlq_org_resolved',
        'dead_letter_queue',
        ['organization_id', 'is_resolved', 'review_priority'],
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_dlq_org_resolved', table_name='dead_letter_queue')
    op.drop_index('ix_dlq_is_resolved', table_name='dead_letter_queue')
    op.drop_index('ix_dlq_quarantined_at', table_name='dead_letter_queue')
    op.drop_index('ix_dlq_task_type', table_name='dead_letter_queue')
    op.drop_index('ix_dlq_original_task_id', table_name='dead_letter_queue')
    op.drop_index('ix_dlq_organization_id', table_name='dead_letter_queue')
    op.drop_table('dead_letter_queue')
