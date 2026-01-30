"""Add pipeline tasks table for SLA-based scheduling.

Revision ID: 20260126_add_pipeline_tasks
Revises: 20260126_add_agent_processes
Create Date: 2026-01-26 16:45:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260126_add_pipeline_tasks"
down_revision = "20260126_add_agent_processes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create pipeline_tasks table
    op.create_table(
        "pipeline_tasks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
            nullable=False,
        ),
        sa.Column("org_id", sa.UUID(), nullable=False),
        sa.Column(
            "task_type",
            sa.String(50),
            nullable=False,
            comment="Task type: consolidation, critique, evaluation, feedback_loop, embedding_refresh",
        ),
        sa.Column(
            "status",
            sa.String(20),
            server_default="queued",
            nullable=False,
            comment="Current status: queued, running, blocked, succeeded, failed",
        ),
        sa.Column(
            "priority",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Priority for SLA ordering (higher = sooner)",
        ),
        sa.Column(
            "sla_deadline",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="SLA deadline for this task",
        ),
        sa.Column(
            "sla_category",
            sa.String(50),
            nullable=True,
            comment="SLA category: critical, high, normal, low",
        ),
        sa.Column(
            "attempts",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Current attempt number",
        ),
        sa.Column(
            "max_attempts",
            sa.Integer(),
            server_default="3",
            nullable=False,
            comment="Max retry attempts",
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When execution started",
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When execution finished",
        ),
        sa.Column(
            "duration_ms",
            sa.Integer(),
            nullable=True,
            comment="Execution duration in milliseconds",
        ),
        sa.Column(
            "estimated_tokens",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Est. token cost",
        ),
        sa.Column(
            "estimated_latency_ms",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Est. latency in ms",
        ),
        sa.Column(
            "actual_tokens",
            sa.Integer(),
            nullable=True,
            comment="Actual token cost",
        ),
        sa.Column(
            "actual_latency_ms",
            sa.Integer(),
            nullable=True,
            comment="Actual latency in ms",
        ),
        sa.Column(
            "input_session_id",
            sa.String(100),
            nullable=False,
            comment="Source cognitive session ID",
        ),
        sa.Column(
            "target_resource_id",
            sa.String(100),
            nullable=False,
            comment="Target resource (memory/run/session ID)",
        ),
        sa.Column(
            "task_metadata",
            sa.JSON(),
            server_default="{}",
            nullable=False,
            comment="Task-specific metadata",
        ),
        sa.Column(
            "blocks_on_task_id",
            sa.String(100),
            nullable=True,
            comment="Task ID that blocks this one (dependency tracking)",
        ),
        sa.Column(
            "blocked_by_quota",
            sa.Boolean(),
            server_default="false",
            nullable=False,
            comment="Is this blocked by quota?",
        ),
        sa.Column(
            "last_error",
            sa.Text(),
            nullable=True,
            comment="Error message from last failure",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indices for efficient querying
    op.create_index(
        "ix_pipeline_tasks_task_type",
        "pipeline_tasks",
        ["task_type"],
    )
    op.create_index(
        "ix_pipeline_tasks_status",
        "pipeline_tasks",
        ["status"],
    )
    op.create_index(
        "ix_pipeline_tasks_org_status_priority",
        "pipeline_tasks",
        ["org_id", "status", "priority"],
        postgresql_where="status = 'queued'",
    )
    op.create_index(
        "ix_pipeline_tasks_sla_deadline",
        "pipeline_tasks",
        ["sla_deadline"],
        postgresql_where="status = 'queued'",
    )
    op.create_index(
        "ix_pipeline_tasks_session",
        "pipeline_tasks",
        ["input_session_id", "org_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_pipeline_tasks_session", table_name="pipeline_tasks")
    op.drop_index("ix_pipeline_tasks_sla_deadline", table_name="pipeline_tasks")
    op.drop_index("ix_pipeline_tasks_org_status_priority", table_name="pipeline_tasks")
    op.drop_index("ix_pipeline_tasks_status", table_name="pipeline_tasks")
    op.drop_index("ix_pipeline_tasks_task_type", table_name="pipeline_tasks")
    op.drop_table("pipeline_tasks")
