"""Add Meta Agent Supervision & Calibration tables.

Revision ID: 2026_01_24_0002
Revises: 2026_01_24_0001
Create Date: 2026-01-24

Implements:
- meta_agent_runs
- meta_conflict_registry
- belief_store
- calibration_profiles

All tables are protected by Postgres RLS and scoped by organization_id.
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_01_24_0002"
down_revision = "2026_01_24_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "meta_agent_runs",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("supervision_type", sa.String(length=30), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("final_confidence", sa.Float(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("reasoning_summary", sa.Text(), nullable=True),
        sa.Column(
            "evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "resource_type IN ('memory','cognitive_session')",
            name="ck_meta_agent_runs_resource_type",
        ),
        sa.CheckConstraint(
            "supervision_type IN ('review','arbitration','calibration_update','drift_check')",
            name="ck_meta_agent_runs_supervision_type",
        ),
        sa.CheckConstraint(
            "status IN ('accepted','modified','rejected','contested','escalated')",
            name="ck_meta_agent_runs_status",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(op.f("ix_meta_agent_runs_organization_id"), "meta_agent_runs", ["organization_id"], unique=False)
    op.create_index(op.f("ix_meta_agent_runs_resource_type"), "meta_agent_runs", ["resource_type"], unique=False)
    op.create_index(op.f("ix_meta_agent_runs_resource_id"), "meta_agent_runs", ["resource_id"], unique=False)

    op.create_table(
        "meta_conflict_registry",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("conflict_type", sa.String(length=50), nullable=False),
        sa.Column("candidates", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("resolution", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("resolved_by", sa.String(length=20), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "resource_type IN ('memory','cognitive_session')",
            name="ck_meta_conflict_registry_resource_type",
        ),
        sa.CheckConstraint(
            "conflict_type IN ('classification','topic','entity','promotion','tool','belief')",
            name="ck_meta_conflict_registry_conflict_type",
        ),
        sa.CheckConstraint(
            "status IN ('open','resolved','ignored')",
            name="ck_meta_conflict_registry_status",
        ),
        sa.CheckConstraint(
            "resolved_by IN ('meta_auto','human_admin')",
            name="ck_meta_conflict_registry_resolved_by",
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        op.f("ix_meta_conflict_registry_organization_id"),
        "meta_conflict_registry",
        ["organization_id"],
        unique=False,
    )
    op.create_index(op.f("ix_meta_conflict_registry_resource_id"), "meta_conflict_registry", ["resource_id"], unique=False)
    op.create_index(
        op.f("ix_meta_conflict_registry_status"),
        "meta_conflict_registry",
        ["status"],
        unique=False,
    )

    op.create_table(
        "belief_store",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("belief_key", sa.String(length=200), nullable=False),
        sa.Column("belief_value", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "evidence_memory_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=False)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column(
            "contradiction_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=False)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_belief_store_confidence"),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "memory_id", "belief_key", name="uq_belief_store_org_memory_key"),
    )

    op.create_index(op.f("ix_belief_store_organization_id"), "belief_store", ["organization_id"], unique=False)
    op.create_index(op.f("ix_belief_store_memory_id"), "belief_store", ["memory_id"], unique=False)
    op.create_index(op.f("ix_belief_store_belief_key"), "belief_store", ["belief_key"], unique=False)

    op.create_table(
        "calibration_profiles",
        sa.Column("organization_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("promotion_threshold", sa.Float(), nullable=False, server_default=sa.text("0.75")),
        sa.Column("conflict_escalation_threshold", sa.Float(), nullable=False, server_default=sa.text("0.60")),
        sa.Column("drift_threshold", sa.Float(), nullable=False, server_default=sa.text("0.20")),
        sa.Column("signal_weights", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("learning_rate", sa.Float(), nullable=False, server_default=sa.text("0.05")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("organization_id"),
    )

    # RLS policies
    for table in ["meta_agent_runs", "meta_conflict_registry", "belief_store", "calibration_profiles"]:
        op.execute(sa.text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"))
        op.execute(sa.text(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY"))

    # Org isolation: direct organization_id match
    for table in ["meta_agent_runs", "meta_conflict_registry", "belief_store", "calibration_profiles"]:
        op.execute(
            sa.text(
                f"""
                CREATE POLICY {table}_org_isolation ON {table}
                USING (organization_id::text = current_setting('app.current_org_id', true))
                WITH CHECK (organization_id::text = current_setting('app.current_org_id', true));
                """
            )
        )


def downgrade() -> None:
    op.drop_table("calibration_profiles")

    op.drop_index(op.f("ix_belief_store_belief_key"), table_name="belief_store")
    op.drop_index(op.f("ix_belief_store_memory_id"), table_name="belief_store")
    op.drop_index(op.f("ix_belief_store_organization_id"), table_name="belief_store")
    op.drop_table("belief_store")

    op.drop_index(op.f("ix_meta_conflict_registry_status"), table_name="meta_conflict_registry")
    op.drop_index(op.f("ix_meta_conflict_registry_resource_id"), table_name="meta_conflict_registry")
    op.drop_index(op.f("ix_meta_conflict_registry_organization_id"), table_name="meta_conflict_registry")
    op.drop_table("meta_conflict_registry")

    op.drop_index(op.f("ix_meta_agent_runs_resource_id"), table_name="meta_agent_runs")
    op.drop_index(op.f("ix_meta_agent_runs_resource_type"), table_name="meta_agent_runs")
    op.drop_index(op.f("ix_meta_agent_runs_organization_id"), table_name="meta_agent_runs")
    op.drop_table("meta_agent_runs")
