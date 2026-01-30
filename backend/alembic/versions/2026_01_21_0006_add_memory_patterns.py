"""Add memory_patterns and memory_pattern_evidence tables

Revision ID: 2026_01_21_0006
Revises: f0a1b2c3d4e5
Create Date: 2026-01-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "2026_01_21_0006"
down_revision: Union[str, None] = "f0a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_patterns",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("scope", sa.String(length=50), nullable=False),
        sa.Column("scope_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("scope_key", sa.String(length=120), nullable=False),
        sa.Column("pattern_key", sa.String(length=200), nullable=False),
        sa.Column("pattern_type", sa.String(length=100), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
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

    op.create_index(op.f("ix_memory_patterns_organization_id"), "memory_patterns", ["organization_id"], unique=False)
    op.create_index("ix_memory_patterns_lookup", "memory_patterns", ["organization_id", "scope", "scope_id"], unique=False)
    op.create_index(
        "ux_memory_patterns_scope_key",
        "memory_patterns",
        ["organization_id", "scope_key", "pattern_key"],
        unique=True,
    )

    op.create_table(
        "memory_pattern_evidence",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("pattern_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column(
            "evidence",
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
        sa.ForeignKeyConstraint(["memory_id"], ["memory_metadata.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["pattern_id"], ["memory_patterns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_memory_pattern_evidence_organization_id"), "memory_pattern_evidence", ["organization_id"], unique=False)
    op.create_index(op.f("ix_memory_pattern_evidence_memory_id"), "memory_pattern_evidence", ["memory_id"], unique=False)
    op.create_index(op.f("ix_memory_pattern_evidence_pattern_id"), "memory_pattern_evidence", ["pattern_id"], unique=False)
    op.create_index(
        "ux_memory_pattern_evidence",
        "memory_pattern_evidence",
        ["organization_id", "memory_id", "pattern_id"],
        unique=True,
    )
    op.create_index(
        "ix_memory_pattern_evidence_lookup",
        "memory_pattern_evidence",
        ["organization_id", "memory_id"],
        unique=False,
    )

    # RLS: org isolation (tenant)
    op.execute("ALTER TABLE memory_patterns ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_patterns FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_patterns ON memory_patterns
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )

    op.execute("ALTER TABLE memory_pattern_evidence ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_pattern_evidence FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY org_isolation_memory_pattern_evidence ON memory_pattern_evidence
        USING (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        WITH CHECK (organization_id = nullif(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_pattern_evidence ON memory_pattern_evidence")
    op.execute("DROP POLICY IF EXISTS org_isolation_memory_patterns ON memory_patterns")

    op.drop_index("ix_memory_pattern_evidence_lookup", table_name="memory_pattern_evidence")
    op.drop_index("ux_memory_pattern_evidence", table_name="memory_pattern_evidence")
    op.drop_index(op.f("ix_memory_pattern_evidence_pattern_id"), table_name="memory_pattern_evidence")
    op.drop_index(op.f("ix_memory_pattern_evidence_memory_id"), table_name="memory_pattern_evidence")
    op.drop_index(op.f("ix_memory_pattern_evidence_organization_id"), table_name="memory_pattern_evidence")
    op.drop_table("memory_pattern_evidence")

    op.drop_index("ux_memory_patterns_scope_key", table_name="memory_patterns")
    op.drop_index("ix_memory_patterns_lookup", table_name="memory_patterns")
    op.drop_index(op.f("ix_memory_patterns_organization_id"), table_name="memory_patterns")
    op.drop_table("memory_patterns")
