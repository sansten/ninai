"""Create graph_relationships table

Revision ID: 002_graph_rel
Revises: 2026_01_27_0005_add_fts_hybrid_search
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_graph_rel'
down_revision = 'add_fts_hybrid_search'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create graph_relationships table."""
    op.create_table(
        'graph_relationships',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('from_memory_id', sa.String(36), nullable=False),
        sa.Column('to_memory_id', sa.String(36), nullable=False),
        sa.Column('relationship_type', sa.String(50), nullable=False, server_default='RELATES_TO'),
        sa.Column('similarity_score', sa.Float(), nullable=True),
        sa.Column('auto_created', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_by_user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'organization_id', 'from_memory_id', 'to_memory_id', 'relationship_type',
            name='uq_graph_rel_unique'
        )
    )
    
    # Create indexes
    op.create_index(
        'idx_graph_rel_org_from',
        'graph_relationships',
        ['organization_id', 'from_memory_id']
    )
    op.create_index(
        'idx_graph_rel_org_to',
        'graph_relationships',
        ['organization_id', 'to_memory_id']
    )
    op.create_index(
        'idx_graph_rel_type',
        'graph_relationships',
        ['organization_id', 'relationship_type']
    )
    op.create_index(
        'idx_graph_rel_similarity',
        'graph_relationships',
        ['similarity_score']
    )
    op.create_index(
        'idx_graph_rel_auto_created',
        'graph_relationships',
        ['auto_created']
    )
    op.create_index(
        'idx_graph_rel_created_at',
        'graph_relationships',
        ['created_at']
    )


def downgrade() -> None:
    """Drop graph_relationships table."""
    # Drop indexes
    op.drop_index('idx_graph_rel_created_at', table_name='graph_relationships')
    op.drop_index('idx_graph_rel_auto_created', table_name='graph_relationships')
    op.drop_index('idx_graph_rel_similarity', table_name='graph_relationships')
    op.drop_index('idx_graph_rel_type', table_name='graph_relationships')
    op.drop_index('idx_graph_rel_org_to', table_name='graph_relationships')
    op.drop_index('idx_graph_rel_org_from', table_name='graph_relationships')
    
    # Drop table
    op.drop_table('graph_relationships')
