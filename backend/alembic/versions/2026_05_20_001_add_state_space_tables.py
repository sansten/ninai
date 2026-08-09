"""add state space stream and snapshot tables

Revision ID: 2026_05_20_001
Revises: 2026_05_16_001
Create Date: 2026-05-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "2026_05_20_001"
down_revision = "2026_05_16_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_state_stream",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scope_type", sa.String(length=50), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False, server_default="memory_write"),
        sa.Column("event_summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("event_features", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("fact_delta", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.create_index("ix_memory_state_stream_organization_id", "memory_state_stream", ["organization_id"], unique=False)
    op.create_index("ix_memory_state_stream_source_memory_id", "memory_state_stream", ["source_memory_id"], unique=False)
    op.create_index("ix_memory_state_stream_scope_type", "memory_state_stream", ["scope_type"], unique=False)
    op.create_index("ix_memory_state_stream_scope_key", "memory_state_stream", ["scope_key"], unique=False)
    op.create_index(
        "ix_memory_state_stream_org_scope_created",
        "memory_state_stream",
        ["organization_id", "scope_type", "scope_key", "created_at"],
        unique=False,
    )

    op.create_table(
        "memory_state_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("scope_type", sa.String(length=50), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("symbolic_state", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("summary_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("salience_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("staleness_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "last_memory_id",
            postgresql.UUID(as_uuid=False),
            sa.ForeignKey("memory_metadata.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("organization_id", "scope_type", "scope_key", name="uq_memory_state_snapshot_scope"),
    )
    op.create_index("ix_memory_state_snapshots_organization_id", "memory_state_snapshots", ["organization_id"], unique=False)
    op.create_index("ix_memory_state_snapshots_scope_type", "memory_state_snapshots", ["scope_type"], unique=False)
    op.create_index("ix_memory_state_snapshots_scope_key", "memory_state_snapshots", ["scope_key"], unique=False)
    op.create_index(
        "ix_memory_state_snapshots_org_scope",
        "memory_state_snapshots",
        ["organization_id", "scope_type", "scope_key"],
        unique=False,
    )

    op.execute("ALTER TABLE memory_state_stream ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memory_state_snapshots ENABLE ROW LEVEL SECURITY")

    op.execute(
        """
        CREATE POLICY memory_state_stream_org_isolation ON memory_state_stream
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )
    op.execute(
        """
        CREATE POLICY memory_state_snapshots_org_isolation ON memory_state_snapshots
        USING (organization_id = current_setting('app.current_org_id', TRUE)::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS memory_state_snapshots_org_isolation ON memory_state_snapshots")
    op.execute("DROP POLICY IF EXISTS memory_state_stream_org_isolation ON memory_state_stream")

    op.drop_index("ix_memory_state_snapshots_org_scope", table_name="memory_state_snapshots")
    op.drop_index("ix_memory_state_snapshots_scope_key", table_name="memory_state_snapshots")
    op.drop_index("ix_memory_state_snapshots_scope_type", table_name="memory_state_snapshots")
    op.drop_index("ix_memory_state_snapshots_organization_id", table_name="memory_state_snapshots")
    op.drop_table("memory_state_snapshots")

    op.drop_index("ix_memory_state_stream_org_scope_created", table_name="memory_state_stream")
    op.drop_index("ix_memory_state_stream_scope_key", table_name="memory_state_stream")
    op.drop_index("ix_memory_state_stream_scope_type", table_name="memory_state_stream")
    op.drop_index("ix_memory_state_stream_source_memory_id", table_name="memory_state_stream")
    op.drop_index("ix_memory_state_stream_organization_id", table_name="memory_state_stream")
    op.drop_table("memory_state_stream")
