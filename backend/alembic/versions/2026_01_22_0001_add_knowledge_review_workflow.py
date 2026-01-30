"""Add HITL knowledge review workflow tables.

Revision ID: 2026_01_22_0001
Revises: 2026_01_21_0012
Create Date: 2026-01-22

Adds:
- knowledge_items
- knowledge_item_versions
- knowledge_review_requests

These tables support a human-in-the-loop submission -> review -> publish flow,
with immutable versions and audit-friendly review requests.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "2026_01_22_0001"
down_revision = "2026_01_21_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_items",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=True),
        sa.Column("is_published", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("published_version_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_knowledge_items_organization_id", "knowledge_items", ["organization_id"], unique=False)
    op.create_index("ix_knowledge_items_published_version_id", "knowledge_items", ["published_version_id"], unique=False)
    op.create_index("ix_knowledge_items_org_title", "knowledge_items", ["organization_id", "title"], unique=False)
    op.create_index("ux_knowledge_items_org_key", "knowledge_items", ["organization_id", "key"], unique=True)

    op.create_table(
        "knowledge_item_versions",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "item_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("knowledge_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("extra_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "created_by_user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("provenance", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.create_index("ix_knowledge_item_versions_organization_id", "knowledge_item_versions", ["organization_id"], unique=False)
    op.create_index("ix_knowledge_item_versions_item_id", "knowledge_item_versions", ["item_id"], unique=False)
    op.create_index("ix_knowledge_item_versions_created_by_user_id", "knowledge_item_versions", ["created_by_user_id"], unique=False)
    op.create_index("ix_knowledge_item_versions_org_item", "knowledge_item_versions", ["organization_id", "item_id"], unique=False)
    op.create_index(
        "ux_knowledge_item_versions_item_vn",
        "knowledge_item_versions",
        ["item_id", "version_number"],
        unique=True,
    )

    op.create_foreign_key(
        "fk_knowledge_items_published_version_id",
        source_table="knowledge_items",
        referent_table="knowledge_item_versions",
        local_cols=["published_version_id"],
        remote_cols=["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "knowledge_review_requests",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "item_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("knowledge_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "item_version_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("knowledge_item_versions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column(
            "requested_by_user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "reviewed_by_user_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision_comment", sa.Text(), nullable=True),
    )
    op.create_index("ix_knowledge_review_requests_organization_id", "knowledge_review_requests", ["organization_id"], unique=False)
    op.create_index("ix_knowledge_review_requests_item_id", "knowledge_review_requests", ["item_id"], unique=False)
    op.create_index("ix_knowledge_review_requests_item_version_id", "knowledge_review_requests", ["item_version_id"], unique=False)
    op.create_index("ix_knowledge_review_requests_status", "knowledge_review_requests", ["status"], unique=False)
    op.create_index(
        "ix_knowledge_review_requests_org_status",
        "knowledge_review_requests",
        ["organization_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_review_requests_org_item",
        "knowledge_review_requests",
        ["organization_id", "item_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_review_requests_org_item", table_name="knowledge_review_requests")
    op.drop_index("ix_knowledge_review_requests_org_status", table_name="knowledge_review_requests")
    op.drop_index("ix_knowledge_review_requests_status", table_name="knowledge_review_requests")
    op.drop_index("ix_knowledge_review_requests_item_version_id", table_name="knowledge_review_requests")
    op.drop_index("ix_knowledge_review_requests_item_id", table_name="knowledge_review_requests")
    op.drop_index("ix_knowledge_review_requests_organization_id", table_name="knowledge_review_requests")
    op.drop_table("knowledge_review_requests")

    op.drop_index("ux_knowledge_item_versions_item_vn", table_name="knowledge_item_versions")
    op.drop_index("ix_knowledge_item_versions_org_item", table_name="knowledge_item_versions")
    op.drop_index("ix_knowledge_item_versions_created_by_user_id", table_name="knowledge_item_versions")
    op.drop_index("ix_knowledge_item_versions_item_id", table_name="knowledge_item_versions")
    op.drop_index("ix_knowledge_item_versions_organization_id", table_name="knowledge_item_versions")
    op.drop_table("knowledge_item_versions")

    op.drop_constraint("fk_knowledge_items_published_version_id", "knowledge_items", type_="foreignkey")
    op.drop_index("ux_knowledge_items_org_key", table_name="knowledge_items")
    op.drop_index("ix_knowledge_items_org_title", table_name="knowledge_items")
    op.drop_index("ix_knowledge_items_published_version_id", table_name="knowledge_items")
    op.drop_index("ix_knowledge_items_organization_id", table_name="knowledge_items")
    op.drop_table("knowledge_items")
