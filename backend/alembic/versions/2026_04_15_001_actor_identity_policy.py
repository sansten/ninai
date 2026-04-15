"""actor identity policy tables

Revision ID: 2026_04_15_001
Revises: 2026_04_10_002
Create Date: 2026-04-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "2026_04_15_001"
down_revision = "2026_04_10_002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # org_identity_policies
    # ------------------------------------------------------------------
    op.create_table(
        "org_identity_policies",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("org_id", sa.String(255), nullable=False),
        sa.Column("mandate_actor_identity", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("allowed_modes", JSONB, nullable=False, server_default='["full","role_only","anonymous"]'),
        sa.Column("enrich_from_directory", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("audit_trail_always", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_org_identity_policies_org_id",
        "org_identity_policies",
        ["org_id"],
        unique=True,
    )

    # ------------------------------------------------------------------
    # user_identity_preferences
    # ------------------------------------------------------------------
    op.create_table(
        "user_identity_preferences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("org_id", sa.String(255), nullable=False),
        sa.Column("preference", sa.String(20), nullable=False, server_default="full"),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("changed_by", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_user_identity_preferences_user_id",
        "user_identity_preferences",
        ["user_id"],
        unique=True,
    )
    op.create_index(
        "ix_user_identity_preferences_org_id",
        "user_identity_preferences",
        ["org_id"],
    )

    # ------------------------------------------------------------------
    # memory_identity_audits
    # ------------------------------------------------------------------
    op.create_table(
        "memory_identity_audits",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("memory_id", sa.String(255), nullable=False),
        sa.Column("org_id", sa.String(255), nullable=False),
        sa.Column("actual_actor_id", sa.String(255), nullable=True),
        sa.Column("actual_actor_type", sa.String(50), nullable=True),
        sa.Column("actual_role", sa.String(100), nullable=True),
        sa.Column("actual_department", sa.String(255), nullable=True),
        sa.Column("mode_applied", sa.String(20), nullable=False),
        sa.Column("mandate_was_active", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("identity_confidence", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("ad_enriched", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_memory_identity_audits_memory_id",
        "memory_identity_audits",
        ["memory_id"],
    )
    op.create_index(
        "ix_memory_identity_audits_org_id",
        "memory_identity_audits",
        ["org_id"],
    )


def downgrade() -> None:
    op.drop_table("memory_identity_audits")
    op.drop_table("user_identity_preferences")
    op.drop_table("org_identity_policies")
