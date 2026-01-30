"""Add logseq_export_files table

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "logseq_export_files",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("relative_path", sa.String(length=500), nullable=False),
        sa.Column("bytes_written", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("requested_by_user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("trace_id", sa.String(length=255), nullable=True),
        sa.Column(
            "options",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
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

    op.create_index(op.f("ix_logseq_export_files_organization_id"), "logseq_export_files", ["organization_id"], unique=False)
    op.create_index("ix_logseq_export_files_lookup", "logseq_export_files", ["organization_id", "requested_by_user_id", "created_at"], unique=False)
    op.create_index("ux_logseq_export_files_org_path", "logseq_export_files", ["organization_id", "relative_path"], unique=True)

    op.execute("ALTER TABLE logseq_export_files ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE logseq_export_files FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY logseq_export_files_isolation
        ON logseq_export_files
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS logseq_export_files_isolation ON logseq_export_files")
    op.drop_index("ux_logseq_export_files_org_path", table_name="logseq_export_files")
    op.drop_index("ix_logseq_export_files_lookup", table_name="logseq_export_files")
    op.drop_index(op.f("ix_logseq_export_files_organization_id"), table_name="logseq_export_files")
    op.drop_table("logseq_export_files")
