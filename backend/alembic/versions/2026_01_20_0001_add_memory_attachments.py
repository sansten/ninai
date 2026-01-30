"""Add memory attachments

Revision ID: 8c1f7b2d8b3a
Revises: 57af38c507fd
Create Date: 2026-01-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "8c1f7b2d8b3a"
down_revision: Union[str, None] = "57af38c507fd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_attachments",
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "uploaded_by",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("storage_path", sa.String(length=1024), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_memory_attachments_organization_id"),
        "memory_attachments",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_memory_attachments_memory_id"),
        "memory_attachments",
        ["memory_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_memory_attachments_uploaded_by"),
        "memory_attachments",
        ["uploaded_by"],
        unique=False,
    )
    op.create_index(
        op.f("ix_memory_attachments_sha256"),
        "memory_attachments",
        ["sha256"],
        unique=False,
    )
    op.create_index(
        "ix_memory_attachments_org_memory",
        "memory_attachments",
        ["organization_id", "memory_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_memory_attachments_org_memory", table_name="memory_attachments")
    op.drop_index(op.f("ix_memory_attachments_sha256"), table_name="memory_attachments")
    op.drop_index(op.f("ix_memory_attachments_uploaded_by"), table_name="memory_attachments")
    op.drop_index(op.f("ix_memory_attachments_memory_id"), table_name="memory_attachments")
    op.drop_index(op.f("ix_memory_attachments_organization_id"), table_name="memory_attachments")
    op.drop_table("memory_attachments")
