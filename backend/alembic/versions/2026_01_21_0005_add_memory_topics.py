"""Add memory_topics and memory_topic_memberships tables

Revision ID: f0a1b2c3d4e5
Revises: e1f2a3b4c5d6
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_topics",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("scope", sa.String(length=50), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("scope_key", sa.String(length=120), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("label_normalized", sa.String(length=200), nullable=False),
        sa.Column(
            "keywords",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("created_by", sa.String(length=50), nullable=False, server_default=sa.text("'agent'")),
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

    op.create_index(op.f("ix_memory_topics_organization_id"), "memory_topics", ["organization_id"], unique=False)
    op.create_index("ix_memory_topics_lookup", "memory_topics", ["organization_id", "scope", "scope_id"], unique=False)
    op.create_index(
        "ux_memory_topics_scope_label",
        "memory_topics",
        ["organization_id", "scope_key", "label_normalized"],
        unique=True,
    )

    op.create_table(
        "memory_topic_memberships",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("topic_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("weight", sa.Float(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_by", sa.String(length=50), nullable=False, server_default=sa.text("'agent'")),
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
        sa.ForeignKeyConstraint(["topic_id"], ["memory_topics.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_memory_topic_memberships_organization_id"),
        "memory_topic_memberships",
        ["organization_id"],
        unique=False,
    )
    op.create_index(op.f("ix_memory_topic_memberships_memory_id"), "memory_topic_memberships", ["memory_id"], unique=False)
    op.create_index(op.f("ix_memory_topic_memberships_topic_id"), "memory_topic_memberships", ["topic_id"], unique=False)
    op.create_index(
        "ux_memory_topic_membership",
        "memory_topic_memberships",
        ["organization_id", "memory_id", "topic_id"],
        unique=True,
    )
    op.create_index(
        "ix_memory_topic_membership_lookup",
        "memory_topic_memberships",
        ["organization_id", "memory_id", "is_primary"],
        unique=False,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE memory_topics ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_topics FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_topics ON memory_topics
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )

    op.execute("ALTER TABLE memory_topic_memberships ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_topic_memberships FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_topic_memberships ON memory_topic_memberships
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_topic_memberships ON memory_topic_memberships")
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_topics ON memory_topics")

    op.drop_index("ix_memory_topic_membership_lookup", table_name="memory_topic_memberships")
    op.drop_index("ux_memory_topic_membership", table_name="memory_topic_memberships")
    op.drop_index(op.f("ix_memory_topic_memberships_topic_id"), table_name="memory_topic_memberships")
    op.drop_index(op.f("ix_memory_topic_memberships_memory_id"), table_name="memory_topic_memberships")
    op.drop_index(op.f("ix_memory_topic_memberships_organization_id"), table_name="memory_topic_memberships")
    op.drop_table("memory_topic_memberships")

    op.drop_index("ux_memory_topics_scope_label", table_name="memory_topics")
    op.drop_index("ix_memory_topics_lookup", table_name="memory_topics")
    op.drop_index(op.f("ix_memory_topics_organization_id"), table_name="memory_topics")
    op.drop_table("memory_topics")
