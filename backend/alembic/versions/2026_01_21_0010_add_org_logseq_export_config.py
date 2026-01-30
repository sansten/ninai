"""Add org_logseq_export_config table

Revision ID: f1a2b3c4d5e6
Revises: d1e2f3a4b5c6
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "org_logseq_export_config",
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("export_base_dir", sa.String(length=1024), nullable=True),
        sa.Column("last_nightly_export_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_by_user_id", postgresql.UUID(as_uuid=False), nullable=True),
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
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_org_logseq_export_config_organization_id"),
        "org_logseq_export_config",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ux_org_logseq_export_config_org",
        "org_logseq_export_config",
        ["organization_id"],
        unique=True,
    )

    op.execute("ALTER TABLE org_logseq_export_config ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE org_logseq_export_config FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_org_logseq_export_config ON org_logseq_export_config
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_org_logseq_export_config ON org_logseq_export_config")
    op.drop_index("ux_org_logseq_export_config_org", table_name="org_logseq_export_config")
    op.drop_index(op.f("ix_org_logseq_export_config_organization_id"), table_name="org_logseq_export_config")
    op.drop_table("org_logseq_export_config")
