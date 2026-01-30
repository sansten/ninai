"""Add Backup Automation models

Revision ID: backup_models
Revises: mfa_models
Create Date: 2026-01-27 14:15:00.000000+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "backup_models"
down_revision: Union[str, None] = "mfa_models"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Create backup_task table
    op.create_table(
        "backup_task",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("backup_type", sa.String(length=20), nullable=False, server_default="full"),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("s3_path", sa.String(length=500), nullable=True),
        sa.Column("s3_object_key", sa.String(length=500), nullable=True),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=True),
        sa.Column("retention_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("backup_metadata", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_backup_task_status", "backup_task", ["status"], unique=False)
    op.create_index("ix_backup_task_created_at", "backup_task", ["created_at"], unique=False)
    op.create_index("ix_backup_task_backup_type", "backup_task", ["backup_type"], unique=False)
    op.create_index("ix_backup_task_retention_until", "backup_task", ["retention_until"], unique=False)
    
    # Create backup_schedule table
    op.create_table(
        "backup_schedule",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("frequency", sa.String(length=20), nullable=False),
        sa.Column("retention_days", sa.Integer(), nullable=False),
        sa.Column("backup_time", sa.String(length=5), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("s3_bucket", sa.String(length=255), nullable=False),
        sa.Column("s3_prefix", sa.String(length=500), nullable=True),
        sa.Column("max_backup_size_gb", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("enable_incremental", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_backup_schedule_enabled", "backup_schedule", ["enabled"], unique=False)
    op.create_index("ix_backup_schedule_frequency", "backup_schedule", ["frequency"], unique=False)
    op.create_index("ix_backup_schedule_next_run_at", "backup_schedule", ["next_run_at"], unique=False)
    
    # Create backup_restore table
    op.create_table(
        "backup_restore",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("backup_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("initiated_by", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("duration_seconds", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["backup_id"], ["backup_task.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["initiated_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_backup_restore_backup_id", "backup_restore", ["backup_id"], unique=False)
    op.create_index("ix_backup_restore_initiated_by", "backup_restore", ["initiated_by"], unique=False)
    op.create_index("ix_backup_restore_status", "backup_restore", ["status"], unique=False)
    op.create_index("ix_backup_restore_created_at", "backup_restore", ["created_at"], unique=False)


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_table("backup_restore")
    op.drop_table("backup_schedule")
    op.drop_table("backup_task")
