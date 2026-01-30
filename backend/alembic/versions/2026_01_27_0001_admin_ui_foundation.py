"""Admin UI foundation - roles, permissions, settings, audit

Revision ID: admin_ui_foundation
Revises: 22a29f3b8904
Create Date: 2026-01-27 10:00:00.000000+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "admin_ui_foundation"
down_revision: Union[str, None] = "22a29f3b8904"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Create admin_roles table
    op.create_table(
        "admin_roles",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("permissions", postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column("is_system", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("created_by", sa.UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_admin_roles_name", "admin_roles", ["name"], unique=False)
    op.create_index("ix_admin_roles_created_by", "admin_roles", ["created_by"], unique=False)
    
    # Create admin_settings table
    op.create_table(
        "admin_settings",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_secret", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_by", sa.UUID(as_uuid=False), nullable=True),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("category", "key"),
    )
    op.create_index("ix_admin_settings_category", "admin_settings", ["category"], unique=False)
    op.create_index("ix_admin_settings_category_key", "admin_settings", ["category", "key"], unique=False)
    op.create_index("ix_admin_settings_updated_by", "admin_settings", ["updated_by"], unique=False)
    
    # Create admin_audit_logs table
    op.create_table(
        "admin_audit_logs",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("admin_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=True),
        sa.Column("old_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", postgresql.INET(), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["admin_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_admin_audit_logs_admin_id", "admin_audit_logs", ["admin_id"], unique=False)
    op.create_index("ix_admin_audit_logs_created_at", "admin_audit_logs", ["created_at"], unique=False)
    op.create_index("ix_admin_audit_logs_resource", "admin_audit_logs", ["resource_type", "resource_id"], unique=False)
    op.create_index("ix_admin_audit_logs_action", "admin_audit_logs", ["action"], unique=False)
    
    # Create admin_sessions table
    op.create_table(
        "admin_sessions",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("admin_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("token_hash", sa.String(length=255), nullable=False),
        sa.Column("ip_address", postgresql.INET(), nullable=False),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_activity", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["admin_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index("ix_admin_sessions_admin_id", "admin_sessions", ["admin_id"], unique=False)
    op.create_index("ix_admin_sessions_expires_at", "admin_sessions", ["expires_at"], unique=False)
    
    # Create admin_ip_whitelist table
    op.create_table(
        "admin_ip_whitelist",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("ip_address", postgresql.INET(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_by", sa.UUID(as_uuid=False), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ip_address"),
    )
    op.create_index("ix_admin_ip_whitelist_ip_address", "admin_ip_whitelist", ["ip_address"], unique=False)
    op.create_index("ix_admin_ip_whitelist_created_by", "admin_ip_whitelist", ["created_by"], unique=False)
    
    # Extend users table with admin fields
    op.add_column("users", sa.Column("admin_role_id", sa.UUID(as_uuid=False), nullable=True))
    op.add_column("users", sa.Column("admin_notes", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("last_admin_action_by", sa.UUID(as_uuid=False), nullable=True))
    op.add_column("users", sa.Column("last_admin_action_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("users", sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="false"))
    
    # Add foreign key constraint for admin_role_id
    op.create_foreign_key(
        "fk_users_admin_role_id",
        "users", "admin_roles",
        ["admin_role_id"], ["id"]
    )
    op.create_foreign_key(
        "fk_users_last_admin_action_by",
        "users", "users",
        ["last_admin_action_by"], ["id"]
    )
    
    # Create indexes for new columns
    op.create_index("ix_users_admin_role_id", "users", ["admin_role_id"], unique=False)
    op.create_index("ix_users_is_admin", "users", ["is_admin"], unique=False)
    op.create_index("ix_users_last_admin_action_by", "users", ["last_admin_action_by"], unique=False)


def downgrade() -> None:
    """Downgrade database schema."""
    
    # Drop indexes
    op.drop_index("ix_users_last_admin_action_by", table_name="users")
    op.drop_index("ix_users_is_admin", table_name="users")
    op.drop_index("ix_users_admin_role_id", table_name="users")
    
    # Drop foreign keys
    op.drop_constraint("fk_users_last_admin_action_by", "users", type_="foreignkey")
    op.drop_constraint("fk_users_admin_role_id", "users", type_="foreignkey")
    
    # Drop columns from users
    op.drop_column("users", "is_admin")
    op.drop_column("users", "last_admin_action_at")
    op.drop_column("users", "last_admin_action_by")
    op.drop_column("users", "admin_notes")
    op.drop_column("users", "admin_role_id")
    
    # Drop admin tables
    op.drop_index("ix_admin_ip_whitelist_created_by", table_name="admin_ip_whitelist")
    op.drop_index("ix_admin_ip_whitelist_ip_address", table_name="admin_ip_whitelist")
    op.drop_table("admin_ip_whitelist")
    
    op.drop_index("ix_admin_sessions_expires_at", table_name="admin_sessions")
    op.drop_index("ix_admin_sessions_admin_id", table_name="admin_sessions")
    op.drop_table("admin_sessions")
    
    op.drop_index("ix_admin_audit_logs_action", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_resource", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_created_at", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_admin_id", table_name="admin_audit_logs")
    op.drop_table("admin_audit_logs")
    
    op.drop_index("ix_admin_settings_updated_by", table_name="admin_settings")
    op.drop_index("ix_admin_settings_category_key", table_name="admin_settings")
    op.drop_index("ix_admin_settings_category", table_name="admin_settings")
    op.drop_table("admin_settings")
    
    op.drop_index("ix_admin_roles_created_by", table_name="admin_roles")
    op.drop_index("ix_admin_roles_name", table_name="admin_roles")
    op.drop_table("admin_roles")
