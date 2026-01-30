"""Add agent_result_cache table

Revision ID: a1b2c3d4e5f6
Revises: f1a2b3c4d5e6
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_result_cache",
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agent_name", sa.String(length=100), nullable=False),
        sa.Column("agent_version", sa.String(length=50), nullable=False),
        sa.Column("strategy", sa.String(length=20), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column(
            "outputs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_agent_result_cache_organization_id"),
        "agent_result_cache",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ux_agent_result_cache_key",
        "agent_result_cache",
        ["organization_id", "agent_name", "agent_version", "strategy", "model", "cache_key"],
        unique=True,
    )
    op.create_index(
        "ix_agent_result_cache_lookup",
        "agent_result_cache",
        ["organization_id", "agent_name", "agent_version", "updated_at"],
        unique=False,
    )
    op.create_index(
        "ix_agent_result_cache_expires",
        "agent_result_cache",
        ["organization_id", "expires_at"],
        unique=False,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE agent_result_cache ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE agent_result_cache FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_agent_result_cache ON agent_result_cache
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_agent_result_cache ON agent_result_cache")
    op.drop_index("ix_agent_result_cache_expires", table_name="agent_result_cache")
    op.drop_index("ix_agent_result_cache_lookup", table_name="agent_result_cache")
    op.drop_index("ux_agent_result_cache_key", table_name="agent_result_cache")
    op.drop_index(op.f("ix_agent_result_cache_organization_id"), table_name="agent_result_cache")
    op.drop_table("agent_result_cache")
