"""Add Federated Memory & Collective Intelligence models (PR-10)

Revision ID: federated_memory_pr10
Revises: multimodal_memory_pr9
Create Date: 2026-02-02 10:00:00.000000+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "federated_memory_pr10"
down_revision: Union[str, None] = "multimodal_memory_pr9"  # Adjust to actual previous revision
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""

    # Create federated_knowledge_summaries table
    op.create_table(
        "federated_knowledge_summaries",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("summary_type", sa.String(length=50), nullable=False),
        sa.Column("domain", sa.String(length=100), nullable=False),
        sa.Column("problem_type", sa.String(length=100), nullable=True),
        sa.Column("pattern_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("anonymized_org_ids", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("aggregated_content", postgresql.JSON(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("validation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("helpful_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_federated_knowledge_summaries_summary_type",
        "federated_knowledge_summaries",
        ["summary_type"],
        unique=False,
    )
    op.create_index(
        "ix_federated_knowledge_summaries_domain",
        "federated_knowledge_summaries",
        ["domain"],
        unique=False,
    )
    op.create_index(
        "ix_federated_knowledge_summaries_quality_score",
        "federated_knowledge_summaries",
        ["quality_score"],
        unique=False,
    )

    # Create org_benchmarks table
    op.create_table(
        "org_benchmarks",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", sa.String(length=128), nullable=False),
        sa.Column("metric_type", sa.String(length=50), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("percentile", sa.Float(), nullable=False),
        sa.Column("peer_count", sa.Integer(), nullable=False),
        sa.Column("measured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trend", sa.String(length=20), nullable=True),
        sa.Column("metadata", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "metric_type", name="uq_org_metric"),
    )
    op.create_index(
        "ix_org_benchmarks_organization_id",
        "org_benchmarks",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_org_benchmarks_metric_type",
        "org_benchmarks",
        ["metric_type"],
        unique=False,
    )
    op.create_index(
        "ix_org_benchmarks_measured_at",
        "org_benchmarks",
        ["measured_at"],
        unique=False,
    )

    # Create privacy_policies table
    op.create_table(
        "privacy_policies",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", sa.String(length=128), nullable=False, unique=True),
        sa.Column("allow_pattern_sharing", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("allow_playbook_contribution", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("allow_benchmark_participation", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("minimum_anonymization_level", sa.String(length=20), nullable=False, server_default="high"),
        sa.Column("excluded_domains", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("max_privacy_budget_epsilon", sa.Float(), nullable=False, server_default="5.0"),
        sa.Column("k_anonymity_threshold", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_privacy_policies_organization_id",
        "privacy_policies",
        ["organization_id"],
        unique=True,
    )

    # Create federated_contributions table
    op.create_table(
        "federated_contributions",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", sa.String(length=128), nullable=False),
        sa.Column("federated_knowledge_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("contributed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("contribution_type", sa.String(length=50), nullable=False),
        sa.Column("impact_score", sa.Float(), nullable=True),
        sa.Column("times_accessed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("average_helpfulness", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["federated_knowledge_id"],
            ["federated_knowledge_summaries.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_federated_contributions_organization_id",
        "federated_contributions",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ix_federated_contributions_federated_knowledge_id",
        "federated_contributions",
        ["federated_knowledge_id"],
        unique=False,
    )
    op.create_index(
        "ix_federated_contributions_contribution_type",
        "federated_contributions",
        ["contribution_type"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_table("federated_contributions")
    op.drop_table("privacy_policies")
    op.drop_table("org_benchmarks")
    op.drop_table("federated_knowledge_summaries")
