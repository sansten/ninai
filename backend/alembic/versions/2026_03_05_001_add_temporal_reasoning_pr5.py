"""add_temporal_reasoning_pr5

Temporal Reasoning Engine models with fact validity, sequence patterns, and trajectories.

Revision ID: 2026_03_05_001_add_temporal_reasoning_pr5
Revises: 2026_03_04_001_add_tool_capability_pr4
Create Date: 2026-03-05 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '2026_03_05_001_add_temporal_reasoning_pr5'
down_revision = '2026_03_04_001_add_tool_capability_pr4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create temporal_facts table
    op.create_table(
        'temporal_facts',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('organization_id', sa.String(36), nullable=False),
        sa.Column('fact_id', sa.String(36), nullable=False),
        sa.Column('valid_from', sa.TIMESTAMP(), nullable=False),
        sa.Column('valid_to', sa.TIMESTAMP(), nullable=True),
        sa.Column('confidence_at_time', sa.Float(), server_default='0.8'),
        sa.Column('change_type', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_temporal_facts_org_fact', 'temporal_facts', ['organization_id', 'fact_id'])
    op.create_index('idx_temporal_facts_valid_range', 'temporal_facts', ['valid_from', 'valid_to'])
    
    # Create temporal_sequences table
    op.create_table(
        'temporal_sequences',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('organization_id', sa.String(36), nullable=False),
        sa.Column('sequence_type', sa.String(50), nullable=False),
        sa.Column('entities', postgresql.JSON(astext_type=sa.Text()), server_default='[]'),
        sa.Column('temporal_gaps', postgresql.JSON(astext_type=sa.Text()), server_default='[]'),
        sa.Column('pattern_type', sa.String(50), nullable=False),
        sa.Column('pattern_strength', sa.Float(), server_default='0.5'),
        sa.Column('last_observed_at', sa.DateTime(), nullable=False),
        sa.Column('predicted_next_event', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('observation_count', sa.Integer(), server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_temporal_sequences_org_type', 'temporal_sequences', ['organization_id', 'pattern_type'])
    op.create_index('idx_temporal_sequences_strength', 'temporal_sequences', ['organization_id', 'pattern_strength'])
    
    # Create temporal_trajectories table
    op.create_table(
        'temporal_trajectories',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('organization_id', sa.String(36), nullable=False),
        sa.Column('entity_id', sa.String(36), nullable=False),
        sa.Column('quantity', sa.String(100), nullable=False),
        sa.Column('measurements', postgresql.JSON(astext_type=sa.Text()), server_default='[]'),
        sa.Column('trend_direction', sa.String(50), nullable=True),
        sa.Column('trend_strength', sa.Float(), nullable=True),
        sa.Column('predicted_future', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('inflection_points', postgresql.JSON(astext_type=sa.Text()), server_default='[]'),
        sa.Column('seasonality', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('last_computed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('idx_temporal_trajectories_org_entity', 'temporal_trajectories', ['organization_id', 'entity_id'])
    op.create_index('idx_temporal_trajectories_quantity', 'temporal_trajectories', ['organization_id', 'quantity'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_temporal_trajectories_quantity', table_name='temporal_trajectories')
    op.drop_index('idx_temporal_trajectories_org_entity', table_name='temporal_trajectories')
    op.drop_index('idx_temporal_sequences_strength', table_name='temporal_sequences')
    op.drop_index('idx_temporal_sequences_org_type', table_name='temporal_sequences')
    op.drop_index('idx_temporal_facts_valid_range', table_name='temporal_facts')
    op.drop_index('idx_temporal_facts_org_fact', table_name='temporal_facts')
    
    # Drop tables
    op.drop_table('temporal_trajectories')
    op.drop_table('temporal_sequences')
    op.drop_table('temporal_facts')
