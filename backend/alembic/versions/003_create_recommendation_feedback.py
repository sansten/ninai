"""Create recommendation_feedback table

Revision ID: 003_rec_fb
Revises: 002_graph_rel
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003_rec_fb'
down_revision = '002_graph_rel'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create recommendation_feedback table."""
    op.create_table(
        'recommendation_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('base_memory_id', sa.String(36), nullable=False),
        sa.Column('recommended_memory_id', sa.String(36), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('helpful', sa.Boolean(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    
    # Create indexes
    op.create_index(
        'idx_rec_feedback_org_base',
        'recommendation_feedback',
        ['organization_id', 'base_memory_id']
    )
    op.create_index(
        'idx_rec_feedback_user_org',
        'recommendation_feedback',
        ['user_id', 'organization_id']
    )
    op.create_index(
        'idx_rec_feedback_helpful',
        'recommendation_feedback',
        ['helpful']
    )
    op.create_index(
        'idx_rec_feedback_created',
        'recommendation_feedback',
        ['created_at']
    )


def downgrade() -> None:
    """Drop recommendation_feedback table."""
    # Drop indexes
    op.drop_index('idx_rec_feedback_created', table_name='recommendation_feedback')
    op.drop_index('idx_rec_feedback_helpful', table_name='recommendation_feedback')
    op.drop_index('idx_rec_feedback_user_org', table_name='recommendation_feedback')
    op.drop_index('idx_rec_feedback_org_base', table_name='recommendation_feedback')
    
    # Drop table
    op.drop_table('recommendation_feedback')
