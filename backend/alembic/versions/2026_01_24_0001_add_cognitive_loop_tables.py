"""Add Cognitive Loop tables.

Revision ID: 2026_01_24_0001
Revises: 2026_01_23_0002
Create Date: 2026-01-24

Implements:
- cognitive_sessions
- cognitive_iterations
- tool_call_logs
- evaluation_reports

All tables are protected by Postgres RLS:
- cognitive_sessions is tenant-scoped via organization_id.
- child tables enforce org isolation via their session_id -> cognitive_sessions.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_24_0001"
down_revision = "2026_01_23_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cognitive_sessions",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("agent_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column(
            "context_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("trace_id", sa.String(length=100), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "status IN ('running','succeeded','failed','aborted')",
            name="ck_cognitive_sessions_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_cognitive_sessions_organization_id"), "cognitive_sessions", ["organization_id"], unique=False)
    op.create_index(op.f("ix_cognitive_sessions_user_id"), "cognitive_sessions", ["user_id"], unique=False)
    op.create_index(op.f("ix_cognitive_sessions_status"), "cognitive_sessions", ["status"], unique=False)
    op.create_index(op.f("ix_cognitive_sessions_trace_id"), "cognitive_sessions", ["trace_id"], unique=False)

    op.execute("ALTER TABLE cognitive_sessions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE cognitive_sessions FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_cognitive_sessions ON cognitive_sessions
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )

    op.create_table(
        "cognitive_iterations",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("iteration_num", sa.Integer(), nullable=False),
        sa.Column(
            "plan_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "execution_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "critique_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("evaluation", sa.String(length=20), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.CheckConstraint(
            "evaluation IN ('pass','fail','retry','needs_evidence')",
            name="ck_cognitive_iterations_evaluation",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "iteration_num", name="ux_cognitive_iterations_session_iteration"),
    )

    op.create_index(op.f("ix_cognitive_iterations_session_id"), "cognitive_iterations", ["session_id"], unique=False)
    op.create_index(op.f("ix_cognitive_iterations_evaluation"), "cognitive_iterations", ["evaluation"], unique=False)

    op.execute("ALTER TABLE cognitive_iterations ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE cognitive_iterations FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_cognitive_iterations ON cognitive_iterations
        USING (
            EXISTS (
                SELECT 1
                FROM cognitive_sessions s
                WHERE s.id = cognitive_iterations.session_id
                  AND s.organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM cognitive_sessions s
                WHERE s.id = cognitive_iterations.session_id
                  AND s.organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid
            )
        )
        """
    )

    op.create_table(
        "tool_call_logs",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "iteration_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("cognitive_iterations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column(
            "tool_input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "tool_output_summary",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("denial_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.CheckConstraint(
            "status IN ('success','denied','failed')",
            name="ck_tool_call_logs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_tool_call_logs_session_id"), "tool_call_logs", ["session_id"], unique=False)
    op.create_index(op.f("ix_tool_call_logs_iteration_id"), "tool_call_logs", ["iteration_id"], unique=False)
    op.create_index(op.f("ix_tool_call_logs_tool_name"), "tool_call_logs", ["tool_name"], unique=False)
    op.create_index(op.f("ix_tool_call_logs_status"), "tool_call_logs", ["status"], unique=False)

    op.execute("ALTER TABLE tool_call_logs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE tool_call_logs FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_tool_call_logs ON tool_call_logs
        USING (
            EXISTS (
                SELECT 1
                FROM cognitive_sessions s
                WHERE s.id = tool_call_logs.session_id
                  AND s.organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM cognitive_sessions s
                WHERE s.id = tool_call_logs.session_id
                  AND s.organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid
            )
        )
        """
    )

    op.create_table(
        "evaluation_reports",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("cognitive_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "report",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("final_decision", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.CheckConstraint(
            "final_decision IN ('pass','fail','needs_evidence','contested')",
            name="ck_evaluation_reports_final_decision",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_evaluation_reports_session_id"), "evaluation_reports", ["session_id"], unique=False)
    op.create_index(op.f("ix_evaluation_reports_final_decision"), "evaluation_reports", ["final_decision"], unique=False)

    op.execute("ALTER TABLE evaluation_reports ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE evaluation_reports FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_evaluation_reports ON evaluation_reports
        USING (
            EXISTS (
                SELECT 1
                FROM cognitive_sessions s
                WHERE s.id = evaluation_reports.session_id
                  AND s.organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid
            )
        )
        WITH CHECK (
            EXISTS (
                SELECT 1
                FROM cognitive_sessions s
                WHERE s.id = evaluation_reports.session_id
                  AND s.organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid
            )
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_evaluation_reports ON evaluation_reports")
    op.drop_index(op.f("ix_evaluation_reports_final_decision"), table_name="evaluation_reports")
    op.drop_index(op.f("ix_evaluation_reports_session_id"), table_name="evaluation_reports")
    op.drop_table("evaluation_reports")

    op.execute("DROP POLICY IF EXISTS org_isolation_tool_call_logs ON tool_call_logs")
    op.drop_index(op.f("ix_tool_call_logs_status"), table_name="tool_call_logs")
    op.drop_index(op.f("ix_tool_call_logs_tool_name"), table_name="tool_call_logs")
    op.drop_index(op.f("ix_tool_call_logs_iteration_id"), table_name="tool_call_logs")
    op.drop_index(op.f("ix_tool_call_logs_session_id"), table_name="tool_call_logs")
    op.drop_table("tool_call_logs")

    op.execute("DROP POLICY IF EXISTS org_isolation_cognitive_iterations ON cognitive_iterations")
    op.drop_index(op.f("ix_cognitive_iterations_evaluation"), table_name="cognitive_iterations")
    op.drop_index(op.f("ix_cognitive_iterations_session_id"), table_name="cognitive_iterations")
    op.drop_table("cognitive_iterations")

    op.execute("DROP POLICY IF EXISTS org_isolation_cognitive_sessions ON cognitive_sessions")
    op.drop_index(op.f("ix_cognitive_sessions_trace_id"), table_name="cognitive_sessions")
    op.drop_index(op.f("ix_cognitive_sessions_status"), table_name="cognitive_sessions")
    op.drop_index(op.f("ix_cognitive_sessions_user_id"), table_name="cognitive_sessions")
    op.drop_index(op.f("ix_cognitive_sessions_organization_id"), table_name="cognitive_sessions")
    op.drop_table("cognitive_sessions")
