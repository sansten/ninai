"""Make memory_access_log.timestamp timezone-aware.

Revision ID: 2026_01_25_0001
Revises: 2026_01_24_0006
Create Date: 2026-01-25
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_01_25_0001"
down_revision = "2026_01_24_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Postgres: convert TIMESTAMP -> TIMESTAMPTZ, interpreting existing values as UTC.
    op.execute(
        "ALTER TABLE memory_access_log "
        "ALTER COLUMN timestamp TYPE TIMESTAMPTZ "
        "USING (timestamp AT TIME ZONE 'UTC')"
    )


def downgrade() -> None:
    # Convert back to TIMESTAMP WITHOUT TIME ZONE, preserving UTC wall time.
    op.execute(
        "ALTER TABLE memory_access_log "
        "ALTER COLUMN timestamp TYPE TIMESTAMP "
        "USING (timestamp AT TIME ZONE 'UTC')"
    )
