"""Add eval_suites, eval_runs, drift_reports tables (PR6: Eval Harness + Drift Detection)

Revision ID: 20260228_eval_harness
Revises: 20260228_checkpoints
Create Date: 2026-02-28
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260228_eval_harness"
down_revision: Union[str, None] = "20260228_checkpoints"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


eval_run_status_enum = postgresql.ENUM(
    "running",
    "success",
    "failure",
    "cancelled",
    name="eval_run_status",
    create_type=False,
)

drift_severity_enum = postgresql.ENUM(
    "none",
    "low",
    "medium",
    "high",
    "critical",
    name="drift_severity",
    create_type=False,
)


def upgrade() -> None:
    # Resolve legacy name collision: older meta-agent migration created a
    # different drift_reports table shape.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'drift_reports'
            )
            AND EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'drift_reports'
                  AND column_name = 'metric_name'
            )
            AND NOT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'drift_reports'
                  AND column_name = 'baseline_run_id'
            )
            THEN
                ALTER TABLE drift_reports RENAME TO meta_drift_reports;
            END IF;
        END
        $$;
        """
    )

    # Create eval_suites table
    op.create_table(
        "eval_suites",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("queries", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("expected", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    op.create_index("ix_eval_suites_org_active", "eval_suites", ["organization_id", "is_active"], unique=False)
    op.create_index("ix_eval_suites_org_created", "eval_suites", ["organization_id", "created_at"], unique=False)
    
    op.execute("ALTER TABLE eval_suites ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY eval_suites_org_isolation ON eval_suites
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )

    # Create eval_run_status enum
    eval_run_status_enum.create(op.get_bind(), checkfirst=True)

    # Create eval_runs table
    op.create_table(
        "eval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("suite_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("eval_suites.id", ondelete="CASCADE"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", eval_run_status_enum, nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    op.create_index("ix_eval_runs_org_suite", "eval_runs", ["organization_id", "suite_id"], unique=False)
    op.create_index("ix_eval_runs_org_started", "eval_runs", ["organization_id", "started_at"], unique=False)
    op.create_index("ix_eval_runs_status", "eval_runs", ["status"], unique=False)
    
    op.execute("ALTER TABLE eval_runs ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY eval_runs_org_isolation ON eval_runs
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )

    # Create drift_severity enum
    drift_severity_enum.create(op.get_bind(), checkfirst=True)

    # Create drift_reports table
    op.create_table(
        "drift_reports",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("baseline_run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("current_run_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("delta", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("severity", drift_severity_enum, nullable=False, server_default="none"),
        sa.Column("flagged_issues", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    op.create_index("ix_drift_reports_org_severity", "drift_reports", ["organization_id", "severity"], unique=False)
    op.create_index("ix_drift_reports_org_created", "drift_reports", ["organization_id", "created_at"], unique=False)
    op.create_index("ix_drift_reports_baseline", "drift_reports", ["baseline_run_id"], unique=False)
    op.create_index("ix_drift_reports_current", "drift_reports", ["current_run_id"], unique=False)
    
    op.execute("ALTER TABLE drift_reports ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY drift_reports_org_isolation ON drift_reports
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )


def downgrade() -> None:
    # Drop drift_reports
    op.execute("DROP POLICY IF EXISTS drift_reports_org_isolation ON drift_reports")
    op.drop_index("ix_drift_reports_current", table_name="drift_reports")
    op.drop_index("ix_drift_reports_baseline", table_name="drift_reports")
    op.drop_index("ix_drift_reports_org_created", table_name="drift_reports")
    op.drop_index("ix_drift_reports_org_severity", table_name="drift_reports")
    op.drop_table("drift_reports")
    drift_severity_enum.drop(op.get_bind(), checkfirst=True)

    # Drop eval_runs
    op.execute("DROP POLICY IF EXISTS eval_runs_org_isolation ON eval_runs")
    op.drop_index("ix_eval_runs_status", table_name="eval_runs")
    op.drop_index("ix_eval_runs_org_started", table_name="eval_runs")
    op.drop_index("ix_eval_runs_org_suite", table_name="eval_runs")
    op.drop_table("eval_runs")
    eval_run_status_enum.drop(op.get_bind(), checkfirst=True)

    # Drop eval_suites
    op.execute("DROP POLICY IF EXISTS eval_suites_org_isolation ON eval_suites")
    op.drop_index("ix_eval_suites_org_created", table_name="eval_suites")
    op.drop_index("ix_eval_suites_org_active", table_name="eval_suites")
    op.drop_table("eval_suites")
