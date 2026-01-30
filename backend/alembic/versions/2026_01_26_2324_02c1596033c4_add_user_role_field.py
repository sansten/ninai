"""
Alembic Migration Script Template
=================================

This is the Mako template for generating new migration scripts.
"""

"""add_user_role_field

Revision ID: 02c1596033c4
Revises: aafe51a38844
Create Date: 2026-01-26 23:24:19.314862+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '02c1596033c4'
down_revision: Union[str, None] = 'aafe51a38844'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Add role column to users table
    op.add_column(
        'users',
        sa.Column('role', sa.String(50), nullable=False, server_default='user')
    )
    
    # Set admins based on is_superuser flag
    op.execute("UPDATE users SET role = 'admin' WHERE is_superuser = true")
    
    # Create index for role-based queries
    op.create_index('ix_users_role', 'users', ['role'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_users_role', table_name='users')
    op.drop_column('users', 'role')
