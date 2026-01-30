"""Add SelfModel tables.

Revision ID: 2026_01_24_0004
Revises: 2026_01_24_0003
Create Date: 2026-01-24

Adds:
- self_model_profiles
- self_model_events

All tables are protected by Postgres RLS and scoped by organization_id.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_24_0004"
down_revision = "2026_01_24_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "self_model_profiles",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column(
            "domain_confidence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "tool_reliability",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "agent_accuracy",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("organization_id"),
    )

    op.create_table(
        "self_model_events",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=True),
        sa.Column("agent_name", sa.Text(), nullable=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('tool_success','tool_failure','agent_corrected','agent_confirmed','policy_denial')",
            name="ck_self_model_events_event_type",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["cognitive_sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_self_model_events_organization_id"), "self_model_events", ["organization_id"], unique=False)
    op.create_index(op.f("ix_self_model_events_event_type"), "self_model_events", ["event_type"], unique=False)
    op.create_index(op.f("ix_self_model_events_created_at"), "self_model_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_self_model_events_tool_name"), "self_model_events", ["tool_name"], unique=False)

    for table in ["self_model_profiles", "self_model_events"]:
        op.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        op.execute(sa.text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))
        op.execute(
            sa.text(
                f"""
                CREATE POLICY org_isolation_{table} ON {table}
                USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
                WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
                """
            )
        )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_self_model_events ON self_model_events")
    op.execute("DROP POLICY IF EXISTS org_isolation_self_model_profiles ON self_model_profiles")

    op.drop_index(op.f("ix_self_model_events_tool_name"), table_name="self_model_events")
    op.drop_index(op.f("ix_self_model_events_created_at"), table_name="self_model_events")
    op.drop_index(op.f("ix_self_model_events_event_type"), table_name="self_model_events")
    op.drop_index(op.f("ix_self_model_events_organization_id"), table_name="self_model_events")

    op.drop_table("self_model_events")
    op.drop_table("self_model_profiles")
