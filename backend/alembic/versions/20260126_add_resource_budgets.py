"""Add resource budgets table for quota tracking.

Revision ID: 20260126_add_resource_budgets
Revises: 20260126_add_pipeline_tasks
Create Date: 2026-01-26 17:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20260126_add_resource_budgets"
down_revision = "20260126_add_pipeline_tasks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create resource_budgets table
    op.create_table(
        "resource_budgets",
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
        sa.Column("organization_id", sa.UUID(), nullable=False),
        sa.Column(
            "period",
            sa.String(20),
            server_default="daily",
            nullable=False,
            comment="Budget period: hourly, daily, monthly",
        ),
        sa.Column(
            "period_start",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="Start of current budget period",
        ),
        sa.Column(
            "period_end",
            sa.DateTime(timezone=True),
            nullable=False,
            comment="End of current budget period",
        ),
        sa.Column(
            "token_budget",
            sa.BigInteger(),
            server_default="1000000",
            nullable=False,
            comment="Total token budget for period",
        ),
        sa.Column(
            "tokens_used",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
            comment="Tokens consumed in current period",
        ),
        sa.Column(
            "tokens_reserved",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
            comment="Tokens reserved but not yet consumed",
        ),
        sa.Column(
            "storage_budget_mb",
            sa.Integer(),
            server_default="10000",
            nullable=False,
            comment="Total storage budget in MB",
        ),
        sa.Column(
            "storage_used_mb",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Storage consumed in MB",
        ),
        sa.Column(
            "latency_budget_ms",
            sa.Integer(),
            server_default="30000",
            nullable=False,
            comment="Total latency budget in ms per period",
        ),
        sa.Column(
            "latency_consumed_ms",
            sa.BigInteger(),
            server_default="0",
            nullable=False,
            comment="Latency consumed in current period",
        ),
        sa.Column(
            "slo_breach_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Number of SLO breaches in period",
        ),
        sa.Column(
            "request_budget",
            sa.Integer(),
            server_default="10000",
            nullable=False,
            comment="Total request budget for period",
        ),
        sa.Column(
            "requests_used",
            sa.Integer(),
            server_default="0",
            nullable=False,
            comment="Requests made in current period",
        ),
        sa.Column(
            "admission_blocked",
            sa.Boolean(),
            server_default="false",
            nullable=False,
            comment="Is admission currently blocked due to quota exhaustion?",
        ),
        sa.Column(
            "degraded_mode",
            sa.Boolean(),
            server_default="false",
            nullable=False,
            comment="Is tenant in degraded mode (reduced quota)?",
        ),
        sa.Column(
            "throttle_rate",
            sa.Float(),
            server_default="1.0",
            nullable=False,
            comment="Current throttle rate (1.0 = no throttle, 0.5 = 50% throttle)",
        ),
        sa.Column(
            "last_throttle_update",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="When throttle rate was last updated",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indices for efficient querying
    op.create_index(
        "ix_resource_budgets_period",
        "resource_budgets",
        ["period"],
    )
    op.create_index(
        "ix_resource_budgets_period_start",
        "resource_budgets",
        ["period_start"],
    )
    op.create_index(
        "ix_resource_budgets_admission_blocked",
        "resource_budgets",
        ["admission_blocked"],
    )
    op.create_index(
        "ix_resource_budgets_org_period",
        "resource_budgets",
        ["organization_id", "period", "period_start", "period_end"],
    )


def downgrade() -> None:
    op.drop_index("ix_resource_budgets_org_period", table_name="resource_budgets")
    op.drop_index("ix_resource_budgets_admission_blocked", table_name="resource_budgets")
    op.drop_index("ix_resource_budgets_period_start", table_name="resource_budgets")
    op.drop_index("ix_resource_budgets_period", table_name="resource_budgets")
    op.drop_table("resource_budgets")
