"""Add goal_id to cognitive_sessions.

Revision ID: 2026_01_24_0006
Revises: 2026_01_24_0005
Create Date: 2026-01-24

Adds an optional foreign key from cognitive_sessions -> goals to attach
cognitive runs to GoalGraph.

RLS note:
- cognitive_sessions is already protected by org-scoped RLS.
- goals is protected by org + visibility policies.
- This FK is safe because inserts/updates remain subject to RLS checks.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_24_0006"
down_revision = "2026_01_24_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "cognitive_sessions",
        sa.Column("goal_id", postgresql.UUID(as_uuid=False), nullable=True),
    )
    op.create_index(op.f("ix_cognitive_sessions_goal_id"), "cognitive_sessions", ["goal_id"], unique=False)
    op.create_foreign_key(
        "fk_cognitive_sessions_goal_id_goals",
        "cognitive_sessions",
        "goals",
        ["goal_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_cognitive_sessions_goal_id_goals", "cognitive_sessions", type_="foreignkey")
    op.drop_index(op.f("ix_cognitive_sessions_goal_id"), table_name="cognitive_sessions")
    op.drop_column("cognitive_sessions", "goal_id")
