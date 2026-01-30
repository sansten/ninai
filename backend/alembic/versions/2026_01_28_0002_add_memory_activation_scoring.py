"""Add memory activation scoring tables and RLS policies

Revision ID: 20260128_mem_activation
Revises: 20260128_mem_consol, 20260127_add_event_publishing
Create Date: 2026-01-28

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "20260128_mem_activation"
down_revision: Union[str, tuple[str, str], None] = (
    "20260128_mem_consol",
    "20260127_add_event_publishing",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ==================== 1. memory_activation_state ====================
    op.create_table(
        "memory_activation_state",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("base_importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.8"),
        sa.Column("contradicted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("risk_factor", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "last_accessed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_memory_activation_state_org_memory",
        "memory_activation_state",
        ["organization_id", "memory_id"],
        unique=True,
    )
    op.create_index(
        "ix_memory_activation_state_org_accessed",
        "memory_activation_state",
        ["organization_id", "last_accessed_at"],
        unique=False,
    )

    # ==================== 2. memory_coactivation_edges ====================
    op.create_table(
        "memory_coactivation_edges",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "memory_id_a",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "memory_id_b",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("coactivation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("edge_weight", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "last_coactivated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_memory_coactivation_edges_org_a_b",
        "memory_coactivation_edges",
        ["organization_id", "memory_id_a", "memory_id_b"],
        unique=True,
    )
    op.create_index(
        "ix_memory_coactivation_edges_org_a",
        "memory_coactivation_edges",
        ["organization_id", "memory_id_a"],
        unique=False,
    )
    op.create_index(
        "ix_memory_coactivation_edges_org_b",
        "memory_coactivation_edges",
        ["organization_id", "memory_id_b"],
        unique=False,
    )

    # ==================== 3. memory_retrieval_explanations ====================
    op.create_table(
        "memory_retrieval_explanations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=False),
            nullable=False,
        ),
        sa.Column("query_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "retrieved_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("top_k", sa.Integer(), nullable=False, server_default="10"),
        sa.Column(
            "results",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_memory_retrieval_explanations_org_user",
        "memory_retrieval_explanations",
        ["organization_id", "user_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_retrieval_explanations_org_query",
        "memory_retrieval_explanations",
        ["organization_id", "query_hash"],
        unique=False,
    )
    op.create_index(
        "ix_memory_retrieval_explanations_org_timestamp",
        "memory_retrieval_explanations",
        ["organization_id", "retrieved_at"],
        unique=False,
    )

    # ==================== 4. causal_hypotheses ====================
    op.create_table(
        "causal_hypotheses",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=False),
            primary_key=True,
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "episode_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column(
            "from_event_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column(
            "to_event_id",
            postgresql.UUID(as_uuid=False),
            nullable=True,
        ),
        sa.Column("relation", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column(
            "evidence_memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=False)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="proposed"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_causal_hypotheses_org_episode",
        "causal_hypotheses",
        ["organization_id", "episode_id"],
        unique=False,
    )
    op.create_index(
        "ix_causal_hypotheses_org_status",
        "causal_hypotheses",
        ["organization_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_causal_hypotheses_org_created",
        "causal_hypotheses",
        ["organization_id", "created_at"],
        unique=False,
    )

    op.create_index(
        "ix_causal_hypotheses_evidence_memory_ids_gin",
        "causal_hypotheses",
        ["evidence_memory_ids"],
        unique=False,
        postgresql_using="gin",
    )

    # ==================== 5. RLS Policies ====================
    # Enable RLS on all tables
    op.execute("ALTER TABLE memory_activation_state ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_activation_state FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_coactivation_edges ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_coactivation_edges FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_retrieval_explanations ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_retrieval_explanations FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE causal_hypotheses ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE causal_hypotheses FORCE ROW LEVEL SECURITY;")

    # Create RLS policies for memory_activation_state
    op.execute("""
        CREATE POLICY org_isolation_activation_state ON memory_activation_state
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid);
    """)

    # Create RLS policies for memory_coactivation_edges
    op.execute("""
        CREATE POLICY org_isolation_coactivation_edges ON memory_coactivation_edges
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid);
    """)

    # Create RLS policies for memory_retrieval_explanations
    op.execute("""
        CREATE POLICY org_isolation_retrieval_explanations ON memory_retrieval_explanations
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid);
    """)

    # Create RLS policies for causal_hypotheses
    op.execute("""
        CREATE POLICY org_isolation_causal_hypotheses ON causal_hypotheses
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid);
    """)


def downgrade() -> None:
    # Drop RLS policies
    op.execute("DROP POLICY IF EXISTS org_isolation_activation_state ON memory_activation_state;")
    op.execute("DROP POLICY IF EXISTS org_isolation_coactivation_edges ON memory_coactivation_edges;")
    op.execute("DROP POLICY IF EXISTS org_isolation_retrieval_explanations ON memory_retrieval_explanations;")
    op.execute("DROP POLICY IF EXISTS org_isolation_causal_hypotheses ON causal_hypotheses;")

    # Disable RLS
    op.execute("ALTER TABLE memory_activation_state DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_coactivation_edges DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE memory_retrieval_explanations DISABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE causal_hypotheses DISABLE ROW LEVEL SECURITY;")

    # Drop tables
    op.drop_index("ix_causal_hypotheses_evidence_memory_ids_gin", table_name="causal_hypotheses")
    op.drop_index("ix_causal_hypotheses_org_created", table_name="causal_hypotheses")
    op.drop_index("ix_causal_hypotheses_org_status", table_name="causal_hypotheses")
    op.drop_index("ix_causal_hypotheses_org_episode", table_name="causal_hypotheses")
    op.drop_table("causal_hypotheses")

    op.drop_index("ix_memory_retrieval_explanations_org_timestamp", table_name="memory_retrieval_explanations")
    op.drop_index("ix_memory_retrieval_explanations_org_query", table_name="memory_retrieval_explanations")
    op.drop_index("ix_memory_retrieval_explanations_org_user", table_name="memory_retrieval_explanations")
    op.drop_table("memory_retrieval_explanations")

    op.drop_index("ix_memory_coactivation_edges_org_b", table_name="memory_coactivation_edges")
    op.drop_index("ix_memory_coactivation_edges_org_a", table_name="memory_coactivation_edges")
    op.drop_index("ix_memory_coactivation_edges_org_a_b", table_name="memory_coactivation_edges")
    op.drop_table("memory_coactivation_edges")

    op.drop_index("ix_memory_activation_state_org_accessed", table_name="memory_activation_state")
    op.drop_index("ix_memory_activation_state_org_memory", table_name="memory_activation_state")
    op.drop_table("memory_activation_state")
