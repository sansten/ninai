"""Add org_feedback_learning_config table

Revision ID: b1c2d3e4f5a6
Revises: 2026_01_21_0006
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, None] = "2026_01_21_0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "org_feedback_learning_config",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "updated_thresholds",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "stopwords",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "heuristic_weights",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "calibration_delta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("updated_by_user_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("last_source_memory_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("last_agent_version", sa.String(length=50), nullable=True),
        sa.Column("last_trace_id", sa.String(length=255), nullable=True),
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
        op.f("ix_org_feedback_learning_config_organization_id"),
        "org_feedback_learning_config",
        ["organization_id"],
        unique=False,
    )
    op.create_index(
        "ux_org_feedback_learning_config_org",
        "org_feedback_learning_config",
        ["organization_id"],
        unique=True,
    )

    op.execute("ALTER TABLE org_feedback_learning_config ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE org_feedback_learning_config FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_feedback_learning_config_isolation
        ON org_feedback_learning_config
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_feedback_learning_config_isolation ON org_feedback_learning_config")
    op.drop_index("ux_org_feedback_learning_config_org", table_name="org_feedback_learning_config")
    op.drop_index(op.f("ix_org_feedback_learning_config_organization_id"), table_name="org_feedback_learning_config")
    op.drop_table("org_feedback_learning_config")
