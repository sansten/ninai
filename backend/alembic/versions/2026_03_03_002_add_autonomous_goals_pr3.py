"""
Create autonomous goals, knowledge gaps, and outcomes tables for PR-3.

Revision ID: 2026_03_03_002
Revises: 2026_03_03_001
Create Date: 2026-03-03 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260303_pr3_goals'
down_revision = '20260303_pr2_sleep'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create PR-3 autonomous goals tables."""
    
    # Create autonomous_goals table
    op.create_table(
        'autonomous_goals',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('initiator', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('trigger_memory_ids', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('expected_value', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('urgency', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.7'),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('activated_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('completion_evidence', postgresql.JSON(), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=True, server_default='{}'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_autonomous_goals_organization_id', 'organization_id'),
        sa.Index('ix_autonomous_goals_status', 'status'),
        sa.Index('ix_autonomous_goals_activated_at', 'activated_at'),
        sa.Index('ix_autonomous_goals_completed_at', 'completed_at'),
    )
    
    # Create knowledge_gaps table
    op.create_table(
        'knowledge_gaps',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('gap_type', sa.String(), nullable=False),
        sa.Column('domain', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('confidence_in_gap', sa.Float(), nullable=False, server_default='0.7'),
        sa.Column('related_memories', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('suggested_learning_approach', sa.String(), nullable=True),
        sa.Column('discovered_at', sa.DateTime(), nullable=False),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=True, server_default='{}'),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_knowledge_gaps_organization_id', 'organization_id'),
        sa.Index('ix_knowledge_gaps_discovered_at', 'discovered_at'),
        sa.Index('ix_knowledge_gaps_resolved_at', 'resolved_at'),
        sa.Index('ix_knowledge_gaps_domain', 'domain'),
    )
    
    # Create autonomous_goal_outcomes table
    op.create_table(
        'autonomous_goal_outcomes',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('goal_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('outcome_type', sa.String(), nullable=False),
        sa.Column('impact_description', sa.String(), nullable=True),
        sa.Column('feedback_from', sa.String(), nullable=True, server_default='auto_detection'),
        sa.Column('was_user_expecting', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('metadata', postgresql.JSON(), nullable=True, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.Index('ix_autonomous_goal_outcomes_organization_id', 'organization_id'),
        sa.Index('ix_autonomous_goal_outcomes_goal_id', 'goal_id'),
        sa.Index('ix_autonomous_goal_outcomes_created_at', 'created_at'),
    )


def downgrade() -> None:
    """Drop PR-3 autonomous goals tables."""
    op.drop_table('autonomous_goal_outcomes')
    op.drop_table('knowledge_gaps')
    op.drop_table('autonomous_goals')
