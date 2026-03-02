"""add_compositional_generalization_pr7

PR-7 Compositional Generalization Engine tables.

Revision ID: 20260307_pr7_comp_gen
Revises: 20260202_hierarchy_knn
Create Date: 2026-03-07 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260307_pr7_comp_gen"
down_revision = "20260202_hierarchy_knn"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "abstract_procedures",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("concrete_playbook_id", sa.String(36), nullable=False),
        sa.Column("abstraction_level", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("parameters", postgresql.JSON(astext_type=sa.Text()), server_default="{}"),
        sa.Column("prerequisites", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("postconditions", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("invariants", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("instances", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_abstract_procedures_org_level",
        "abstract_procedures",
        ["organization_id", "abstraction_level"],
    )

    op.create_table(
        "analogies",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("organization_id", sa.String(36), nullable=False),
        sa.Column("source_domain", sa.String(255), nullable=False),
        sa.Column("target_domain", sa.String(255), nullable=False),
        sa.Column("structural_similarity", sa.Float(), nullable=True),
        sa.Column("mapped_concepts", postgresql.JSON(astext_type=sa.Text()), server_default="{}"),
        sa.Column("constraints", postgresql.JSON(astext_type=sa.Text()), server_default="[]"),
        sa.Column("applicability", sa.String(50), nullable=True),
        sa.Column("discovered_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_analogies_org_domains",
        "analogies",
        ["organization_id", "source_domain", "target_domain"],
    )


def downgrade() -> None:
    op.drop_index("idx_analogies_org_domains", table_name="analogies")
    op.drop_index("idx_abstract_procedures_org_level", table_name="abstract_procedures")
    op.drop_table("analogies")
    op.drop_table("abstract_procedures")
