"""Add memory facts and contradictions tables (PR3)

Revision ID: 20260228_facts
Revises: 20260228_episodes
Create Date: 2026-02-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260228_facts"
down_revision: Union[str, None] = "20260228_episodes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_facts",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("subject", sa.String(length=255), nullable=False),
        sa.Column("predicate", sa.String(length=255), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("source_memory_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes_fact_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("memory_facts.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("contradiction_group_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_memory_facts_organization_id", "memory_facts", ["organization_id"], unique=False)
    op.create_index("ix_memory_facts_subject", "memory_facts", ["subject"], unique=False)
    op.create_index("ix_memory_facts_predicate", "memory_facts", ["predicate"], unique=False)
    op.create_index("ix_memory_facts_status", "memory_facts", ["status"], unique=False)
    op.create_index("ix_memory_facts_source_memory_id", "memory_facts", ["source_memory_id"], unique=False)
    op.create_index("ix_memory_facts_contradiction_group_id", "memory_facts", ["contradiction_group_id"], unique=False)
    op.create_index(
        "ix_memory_facts_org_subject_pred_status",
        "memory_facts",
        ["organization_id", "subject", "predicate", "status"],
        unique=False,
        postgresql_using="btree",
    )

    op.create_table(
        "contradictions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fact_a", postgresql.UUID(as_uuid=False), sa.ForeignKey("memory_facts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fact_b", postgresql.UUID(as_uuid=False), sa.ForeignKey("memory_facts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False, server_default="medium"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_index("ix_contradictions_organization_id", "contradictions", ["organization_id"], unique=False)
    op.create_index("ix_contradictions_fact_a", "contradictions", ["fact_a"], unique=False)
    op.create_index("ix_contradictions_fact_b", "contradictions", ["fact_b"], unique=False)
    op.create_index("ix_contradictions_severity", "contradictions", ["severity"], unique=False)
    op.create_index(
        "ix_contradictions_org_created",
        "contradictions",
        ["organization_id", "created_at"],
        unique=False,
        postgresql_using="btree",
    )

    op.execute("ALTER TABLE memory_facts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE contradictions ENABLE ROW LEVEL SECURITY")

    op.execute(
        """
        CREATE POLICY memory_facts_org_isolation ON memory_facts
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )

    op.execute(
        """
        CREATE POLICY contradictions_org_isolation ON contradictions
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS contradictions_org_isolation ON contradictions")
    op.execute("DROP POLICY IF EXISTS memory_facts_org_isolation ON memory_facts")

    op.drop_index("ix_contradictions_org_created", table_name="contradictions")
    op.drop_index("ix_contradictions_severity", table_name="contradictions")
    op.drop_index("ix_contradictions_fact_b", table_name="contradictions")
    op.drop_index("ix_contradictions_fact_a", table_name="contradictions")
    op.drop_index("ix_contradictions_organization_id", table_name="contradictions")
    op.drop_table("contradictions")

    op.drop_index("ix_memory_facts_org_subject_pred_status", table_name="memory_facts")
    op.drop_index("ix_memory_facts_contradiction_group_id", table_name="memory_facts")
    op.drop_index("ix_memory_facts_source_memory_id", table_name="memory_facts")
    op.drop_index("ix_memory_facts_status", table_name="memory_facts")
    op.drop_index("ix_memory_facts_predicate", table_name="memory_facts")
    op.drop_index("ix_memory_facts_subject", table_name="memory_facts")
    op.drop_index("ix_memory_facts_organization_id", table_name="memory_facts")
    op.drop_table("memory_facts")
