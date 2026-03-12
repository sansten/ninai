"""Add causal edges and counterfactual scenarios tables

Revision ID: 002_add_causal_edges
Revises: 001_initial_schema
Create Date: 2026-03-02 10:00:00.000000

This migration adds support for causal reasoning to Ninai:
- causal_edges: Represents A → B causal relationships
- counterfactual_scenarios: Tracks hypothetical predictions and outcomes

Why This Matters:
================

Level 1 (Observation): "What if I observe Y?" ← Current Ninai
Level 2 (Intervention): "What if I DO X?" ← NEW
Level 3 (Counterfactual): "What if I HAD done X?" ← NEW

General intelligence requires moving beyond pattern matching to causal reasoning:
- Explanation: "Why did X happen?" (root cause analysis)
- Planning: "If I do X, what happens?" (interventional reasoning)
- Learning: "What would've happened if..." (counterfactual learning)
- Debugging: "What caused the failure?" (causal trace)

This migration implements these capabilities via:

1. CausalEdge: Represents discovered/learned causal relationships
   - mechanism: how causation works (direct, temporal, correlative, hypothetical)
   - strength: 0-1 confidence in the relationship
   - evidence tracking: which memories support this edge
   - validation counts: how many times confirmed/contradicted

2. CounterfactualScenario: Hypothetical "what-if" scenarios
   - Predictions made before observing outcomes
   - Updated when reality manifests
   - Accuracy tracking for causal model improvement

All tables use org_id for RLS (Row-Level Security) and have proper indexes
for efficient querying across causal relationships.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = '20260302_causal_edges'
down_revision = '20260228_eval_harness'
branch_labels = None
depends_on = None


def upgrade():
    # Create causal_edges table
    op.create_table(
        "causal_edges",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("cause_entity_id", sa.String(256), nullable=False),
        sa.Column("cause_entity_type", sa.String(50), nullable=False),
        sa.Column("effect_entity_id", sa.String(256), nullable=False),
        sa.Column("effect_entity_type", sa.String(50), nullable=False),
        sa.Column(
            "mechanism",
            sa.String(50),
            nullable=False,
            comment="direct|temporal|correlative|hypothetical",
        ),
        sa.Column(
            "strength",
            sa.Float,
            nullable=False,
            default=0.5,
            comment="0-1 confidence in causation",
        ),
        sa.Column(
            "latency_hours",
            sa.Integer,
            nullable=True,
            comment="typical delay between cause-effect",
        ),
        sa.Column(
            "evidence_memory_ids",
            postgresql.ARRAY(sa.String),
            nullable=False,
            default=[],
            comment="memory IDs supporting this edge",
        ),
        sa.Column(
            "counterfactual_evidence_ids",
            postgresql.ARRAY(sa.String),
            nullable=False,
            default=[],
            comment="counterfactual IDs confirming/contradicting",
        ),
        sa.Column(
            "validation_count",
            sa.Integer,
            nullable=False,
            default=0,
            comment="times this edge was confirmed",
        ),
        sa.Column(
            "invalidation_count",
            sa.Integer,
            nullable=False,
            default=0,
            comment="times this edge was contradicted",
        ),
        sa.Column("discovered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for efficient querying
    op.create_index(
        "idx_causal_edges_org_cause_effect",
        "causal_edges",
        ["organization_id", "cause_entity_id", "effect_entity_id"],
    )

    op.create_index(
        "idx_causal_edges_org_cause",
        "causal_edges",
        ["organization_id", "cause_entity_id"],
    )

    op.create_index(
        "idx_causal_edges_org_effect",
        "causal_edges",
        ["organization_id", "effect_entity_id"],
    )

    # Create counterfactual_scenarios table
    op.create_table(
        "counterfactual_scenarios",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("causal_edge_id", sa.String(36), nullable=True),
        sa.Column(
            "scenario_type",
            sa.String(50),
            nullable=False,
            comment="prediction|causal_explanation|planning",
        ),
        sa.Column("condition", sa.Text, nullable=False, comment="if X, what about Y?"),
        sa.Column(
            "predicted_outcomes",
            postgresql.JSON,
            nullable=False,
            default={},
            comment="entity: probability",
        ),
        sa.Column(
            "actual_outcome",
            postgresql.JSON,
            nullable=True,
            comment="reality when scenario manifests",
        ),
        sa.Column(
            "accuracy",
            sa.Float,
            nullable=True,
            comment="(1 - |predicted - actual|) / magnitude",
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, default=sa.func.now()
        ),
        sa.Column(
            "resolved_at", sa.DateTime(timezone=True), nullable=True, comment="when actual_outcome set"
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["causal_edge_id"],
            ["causal_edges.id"],
            ondelete="SET NULL",
        ),
    )

    # Create indexes for counterfactual queries
    op.create_index(
        "idx_counterfactual_scenarios_org_edge",
        "counterfactual_scenarios",
        ["organization_id", "causal_edge_id"],
    )

    op.create_index(
        "idx_counterfactual_scenarios_org_type",
        "counterfactual_scenarios",
        ["organization_id", "scenario_type"],
    )

    # Enable RLS on causal_edges
    op.execute(
        """
        ALTER TABLE causal_edges ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY org_isolation_causal_edges ON causal_edges
            USING (organization_id = current_setting('app.current_org_id'))
            WITH CHECK (organization_id = current_setting('app.current_org_id'));
    """
    )

    # Enable RLS on counterfactual_scenarios
    op.execute(
        """
        ALTER TABLE counterfactual_scenarios ENABLE ROW LEVEL SECURITY;
        
        CREATE POLICY org_isolation_counterfactual_scenarios ON counterfactual_scenarios
            USING (organization_id = current_setting('app.current_org_id'))
            WITH CHECK (organization_id = current_setting('app.current_org_id'));
    """
    )


def downgrade():
    # Drop RLS policies
    op.execute("DROP POLICY IF EXISTS org_isolation_causal_edges ON causal_edges")
    op.execute(
        "DROP POLICY IF EXISTS org_isolation_counterfactual_scenarios ON counterfactual_scenarios"
    )

    # Drop tables (foreign key will be handled by cascade)
    op.drop_table("counterfactual_scenarios")
    op.drop_table("causal_edges")
