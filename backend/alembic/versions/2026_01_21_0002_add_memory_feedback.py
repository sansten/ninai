"""Add memory_feedback table

Revision ID: c3a1d2e4f5b6
Revises: 9f2d0e3b4c1a
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c3a1d2e4f5b6"
down_revision: Union[str, None] = "9f2d0e3b4c1a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_feedback",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("actor_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("feedback_type", sa.String(length=50), nullable=False),
        sa.Column("target_agent", sa.String(length=255), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("is_applied", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("applied_by", postgresql.UUID(as_uuid=False), nullable=True),
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
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_memory_feedback_organization_id"), "memory_feedback", ["organization_id"], unique=False)
    op.create_index(op.f("ix_memory_feedback_memory_id"), "memory_feedback", ["memory_id"], unique=False)
    op.create_index(op.f("ix_memory_feedback_actor_id"), "memory_feedback", ["actor_id"], unique=False)
    op.create_index("ix_memory_feedback_lookup", "memory_feedback", ["organization_id", "memory_id", "is_applied"], unique=False)
    op.create_index("ix_memory_feedback_actor", "memory_feedback", ["organization_id", "actor_id", "created_at"], unique=False)

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE memory_feedback ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_feedback FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_feedback ON memory_feedback
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_feedback ON memory_feedback")
    op.drop_index("ix_memory_feedback_actor", table_name="memory_feedback")
    op.drop_index("ix_memory_feedback_lookup", table_name="memory_feedback")
    op.drop_index(op.f("ix_memory_feedback_actor_id"), table_name="memory_feedback")
    op.drop_index(op.f("ix_memory_feedback_memory_id"), table_name="memory_feedback")
    op.drop_index(op.f("ix_memory_feedback_organization_id"), table_name="memory_feedback")
    op.drop_table("memory_feedback")
