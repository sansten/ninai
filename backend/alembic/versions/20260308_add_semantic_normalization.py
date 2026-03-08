"""Phase 6: Add semantic_intent and business_domain to memory_metadata.

These columns are written by SemanticNormalizationAgent and used by:
  - Phase 7 (Ontology): cross-department entity resolution keyed on business_domain
  - Phase 9 (Silo Propagation): route signals to teams matching business_domain

Revision ID: 20260308_semantic_normalization
Revises: 20260307_strategy_library
Create Date: 2026-03-08
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260308_semantic_normalization"
down_revision = "20260307_strategy_library"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memory_metadata",
        sa.Column("semantic_intent", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "memory_metadata",
        sa.Column("business_domain", sa.String(length=100), nullable=True),
    )
    op.create_index(
        "ix_memory_business_domain",
        "memory_metadata",
        ["organization_id", "business_domain"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_business_domain", table_name="memory_metadata")
    op.drop_column("memory_metadata", "business_domain")
    op.drop_column("memory_metadata", "semantic_intent")
