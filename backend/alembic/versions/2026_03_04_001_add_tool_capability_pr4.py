"""
Alembic migration for PR-4: Tool Capability Learning & Adaptive Strategy Selection

Creates tables for tracking tool capabilities, strategy adaptations, and capability discoveries.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '2026_03_04_001_add_tool_capability_pr4'
down_revision = '2026_03_03_002_add_autonomous_goals_pr3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create PR-4 tables."""
    
    # Create ToolType enum
    tool_type = postgresql.ENUM(
        'DATA_ANALYSIS', 'API_CALL', 'CODE_EXECUTION', 'KNOWLEDGE_RETRIEVAL',
        'TEXT_GENERATION', 'IMAGE_GENERATION', 'WEB_SEARCH', 'MEMORY_QUERY',
        'SYSTEM_COMMAND',
        name='tooltype'
    )
    tool_type.create(op.get_bind(), checkfirst=True)
    
    # Create tool_capabilities table
    op.create_table(
        'tool_capabilities',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('tool_name', sa.String(255), nullable=False, index=True),
        sa.Column('tool_type', tool_type, nullable=False, index=True),
        sa.Column('description', sa.String(1000)),
        sa.Column('success_rate', sa.Float, nullable=False, default=0.0),
        sa.Column('avg_execution_time', sa.Float, nullable=False, default=0.0),
        sa.Column('avg_cost', sa.Float, nullable=False, default=0.0),
        sa.Column('reliability_score', sa.Float, nullable=False, default=0.5),
        sa.Column('total_uses', sa.Integer, nullable=False, default=0),
        sa.Column('successful_uses', sa.Integer, nullable=False, default=0),
        sa.Column('failed_uses', sa.Integer, nullable=False, default=0),
        sa.Column('last_used', sa.DateTime),
        sa.Column('supported_goal_types', postgresql.JSON, default=[]),
        sa.Column('supported_domains', postgresql.JSON, default=[]),
        sa.Column('known_limitations', postgresql.JSON, default={}),
        sa.Column('meta', postgresql.JSON, default={}),
        sa.Column('discovered_at', sa.DateTime, nullable=False),
        sa.Column('last_evaluated', sa.DateTime),
        sa.Column('created_at', sa.DateTime, nullable=False, index=True),
    )
    
    # Create indexes on tool_capabilities
    op.create_index(
        'idx_tool_capability_org_type',
        'tool_capabilities',
        ['organization_id', 'tool_type']
    )
    op.create_index(
        'idx_tool_capability_reliability',
        'tool_capabilities',
        ['organization_id', 'reliability_score']
    )
    op.create_index(
        'idx_tool_capability_org_name',
        'tool_capabilities',
        ['organization_id', 'tool_name']
    )
    
    # Create strategy_adaptations table
    op.create_table(
        'strategy_adaptations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('goal_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('goal_type', sa.String(100), nullable=False),
        sa.Column('previous_strategy', sa.String(255)),
        sa.Column('new_strategy', sa.String(255), nullable=False),
        sa.Column('adaptation_reason', sa.String(500), nullable=False),
        sa.Column('previous_success_rate', sa.Float),
        sa.Column('new_success_rate', sa.Float),
        sa.Column('predicted_improvement', sa.Float),
        sa.Column('actual_improvement', sa.Float),
        sa.Column('triggered_by', sa.String(100), nullable=False),
        sa.Column('confidence_in_adaptation', sa.Float, nullable=False, default=0.5),
        sa.Column('meta', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, index=True),
        sa.Column('evaluated_at', sa.DateTime),
    )
    
    # Create indexes on strategy_adaptations
    op.create_index(
        'idx_adaptation_org_goal',
        'strategy_adaptations',
        ['organization_id', 'goal_id']
    )
    op.create_index(
        'idx_adaptation_trigger',
        'strategy_adaptations',
        ['organization_id', 'triggered_by']
    )
    
    # Create capability_discoveries table
    op.create_table(
        'capability_discoveries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('discovery_type', sa.String(100), nullable=False),
        sa.Column('tool_or_capability', sa.String(255), nullable=False),
        sa.Column('description', sa.String(1000), nullable=False),
        sa.Column('potential_value', sa.Float, nullable=False, default=0.5),
        sa.Column('required_investigation', sa.String(1000)),
        sa.Column('can_exploit_now', postgresql.JSON, default=[]),
        sa.Column('discovered_when', sa.String(100), nullable=False),
        sa.Column('validation_status', sa.String(100), nullable=False, default='unvalidated'),
        sa.Column('meta', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime, nullable=False, index=True),
        sa.Column('last_investigated', sa.DateTime),
    )
    
    # Create indexes on capability_discoveries
    op.create_index(
        'idx_discovery_org_type',
        'capability_discoveries',
        ['organization_id', 'discovery_type']
    )
    op.create_index(
        'idx_discovery_status',
        'capability_discoveries',
        ['organization_id', 'validation_status']
    )


def downgrade() -> None:
    """Drop PR-4 tables."""
    
    op.drop_table('capability_discoveries')
    op.drop_table('strategy_adaptations')
    op.drop_table('tool_capabilities')
    
    # Drop enum type
    op.execute('DROP TYPE tooltype')
