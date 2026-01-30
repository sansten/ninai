"""Add export jobs.

Revision ID: 2026_01_22_1300
Revises: 2026_01_22_1200
Create Date: 2026-01-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "2026_01_22_1300"
down_revision = "2026_01_22_1200"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "export_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("job_type", sa.String(length=50), nullable=False, server_default="snapshot"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("file_path", sa.String(length=2048), nullable=True),
        sa.Column("file_bytes", sa.Integer(), nullable=True),
        sa.Column("file_sha256", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.String(length=2000), nullable=True),
    )

    op.create_index("ix_export_jobs_organization_id", "export_jobs", ["organization_id"], unique=False)
    op.create_index("ix_export_jobs_created_by_user_id", "export_jobs", ["created_by_user_id"], unique=False)
    op.create_index("ix_export_jobs_expires_at", "export_jobs", ["expires_at"], unique=False)
    op.create_index("ix_export_jobs_org_status", "export_jobs", ["organization_id", "status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_export_jobs_org_status", table_name="export_jobs")
    op.drop_index("ix_export_jobs_expires_at", table_name="export_jobs")
    op.drop_index("ix_export_jobs_created_by_user_id", table_name="export_jobs")
    op.drop_index("ix_export_jobs_organization_id", table_name="export_jobs")
    op.drop_table("export_jobs")
