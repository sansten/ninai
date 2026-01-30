"""Add simulation_reports.

Revision ID: 2026_01_24_0005
Revises: 2026_01_24_0004
Create Date: 2026-01-24

Adds:
- simulation_reports

All tables are protected by Postgres RLS and scoped by organization_id.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_24_0005"
down_revision = "2026_01_24_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "simulation_reports",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column(
            "report",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["session_id"], ["cognitive_sessions.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_simulation_reports_organization_id"), "simulation_reports", ["organization_id"], unique=False)
    op.create_index(op.f("ix_simulation_reports_session_id"), "simulation_reports", ["session_id"], unique=False)
    op.create_index(op.f("ix_simulation_reports_memory_id"), "simulation_reports", ["memory_id"], unique=False)
    op.create_index(op.f("ix_simulation_reports_created_at"), "simulation_reports", ["created_at"], unique=False)
    op.create_index("ix_simulation_reports_org_created", "simulation_reports", ["organization_id", "created_at"], unique=False)
    op.create_index("ix_simulation_reports_session_created", "simulation_reports", ["session_id", "created_at"], unique=False)

    op.execute(sa.text("ALTER TABLE simulation_reports ENABLE ROW LEVEL SECURITY"))
    op.execute(sa.text("ALTER TABLE simulation_reports FORCE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            """
            CREATE POLICY org_isolation_simulation_reports ON simulation_reports
            USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
            WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
            """
        )
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_simulation_reports ON simulation_reports")
    op.execute(sa.text("ALTER TABLE simulation_reports DISABLE ROW LEVEL SECURITY"))

    op.drop_index("ix_simulation_reports_session_created", table_name="simulation_reports")
    op.drop_index("ix_simulation_reports_org_created", table_name="simulation_reports")
    op.drop_index(op.f("ix_simulation_reports_created_at"), table_name="simulation_reports")
    op.drop_index(op.f("ix_simulation_reports_memory_id"), table_name="simulation_reports")
    op.drop_index(op.f("ix_simulation_reports_session_id"), table_name="simulation_reports")
    op.drop_index(op.f("ix_simulation_reports_organization_id"), table_name="simulation_reports")

    op.drop_table("simulation_reports")
