"""
User and Role Models
====================

Models for users, roles, and role assignments with support for
scoped roles and time-limited grants.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import String, Text, Boolean, Integer, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from app.models.base import Base, UUIDMixin, TimestampMixin


class User(Base, UUIDMixin, TimestampMixin):
    """
    User model.
    
    Users belong to one or more organizations and can hold
    different roles in each organization.
    
    Attributes:
        email: Unique email address (login identifier)
        hashed_password: Bcrypt-hashed password
        full_name: User's display name
        is_active: Whether user can log in
        is_superuser: System-wide admin flag
    """
    
    __tablename__ = "users"
    
    # Identity
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="User email address (unique)",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Bcrypt-hashed password",
    )
    
    # Profile
    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="User's full display name",
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        doc="URL to user's avatar image",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether user account is active",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="System-wide admin (bypass org checks)",
    )

    # Admin UI flag (legacy toggle used by admin tests)
    is_admin: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Legacy admin flag for Admin UI access",
    )
    
    # Role (simple capability-based role)
    role: Mapped[str] = mapped_column(
        String(50),
        default="user",
        nullable=False,
        doc="User role for RBAC (admin, operator, viewer, user)",
    )
    
    # Security
    clearance_level: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        doc="Security clearance level (0-4)",
    )
    
    # Preferences
    preferences: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="User preferences (JSON)",
    )

    # Admin UI role assignment (optional)
    admin_role_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("admin_roles.id"),
        nullable=True,
        index=True,
        doc="Assigned admin role (admin UI permissions)",
    )
    
    # Last login tracking
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="Last successful login timestamp",
    )
    
    # Relationships
    roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="user",
        cascade="all, delete-orphan",
        foreign_keys="[UserRole.user_id]",
    )

    # Admin role relationship (optional)
    admin_role: Mapped[Optional["AdminRole"]] = relationship(
        "AdminRole",
        back_populates="users",
        foreign_keys="[User.admin_role_id]",
    )
    
    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Role(Base, UUIDMixin, TimestampMixin):
    """
    Role definition model.
    
    Roles can be system-wide or organization-specific.
    Permissions are stored as an array of permission strings.
    
    Permission format: "resource:action:scope"
    Examples:
    - "memory:read:own" - Read own memories
    - "memory:write:team" - Write to team memories
    - "admin:manage:org" - Manage organization
    
    Attributes:
        name: Role identifier (e.g., "org_admin")
        display_name: Human-readable name
        permissions: Array of permission strings
        is_system: Whether this is a system-defined role
    """
    
    __tablename__ = "roles"
    
    # Identity
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        doc="Role identifier (e.g., 'org_admin')",
    )
    display_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Human-readable role name",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Role description",
    )
    
    # Permissions
    permissions: Mapped[List[str]] = mapped_column(
        ARRAY(String),
        default=list,
        nullable=False,
        doc="Array of permission strings",
    )
    
    # Organization (null for system roles)
    organization_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        doc="Organization for org-specific roles (null for system roles)",
    )
    
    # Flags
    is_system: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether this is a system-defined role",
    )
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether this role is assigned by default to new users",
    )
    
    # Relationships
    user_roles: Mapped[List["UserRole"]] = relationship(
        "UserRole",
        back_populates="role",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        # Unique name per organization (or globally for system roles)
        Index("ix_roles_name_org", "name", "organization_id", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<Role {self.name}>"


class UserRole(Base, UUIDMixin, TimestampMixin):
    """
    User-Role assignment with scope and expiration.
    
    Allows assigning roles to users with:
    - Organization context
    - Optional scope (team, department, division)
    - Optional expiration date
    
    Attributes:
        user_id: User receiving the role
        role_id: Role being assigned
        organization_id: Organization context
        scope_type: Scope level (team/department/division/organization)
        scope_id: Specific scope entity ID
        expires_at: When the assignment expires (optional)
    """
    
    __tablename__ = "user_roles"
    
    # Assignment
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="User receiving the role",
    )
    role_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("roles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Role being assigned",
    )
    
    # Organization context (tenant)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization context for this assignment",
    )
    
    # Scope (for scoped roles)
    scope_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        doc="Scope level: team, department, division, or organization",
    )
    scope_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="Specific scope entity ID (e.g., team_id)",
    )
    
    # Expiration
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When this role assignment expires",
    )
    
    # Audit
    granted_by: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=True,
        doc="User who granted this role",
    )
    grant_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Reason for granting the role",
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="roles",
        foreign_keys=[user_id],
    )
    role: Mapped["Role"] = relationship(
        "Role",
        back_populates="user_roles",
    )
    
    __table_args__ = (
        # Composite index for permission lookups
        Index("ix_user_roles_lookup", "user_id", "organization_id", "role_id"),
        # Index for expiration checks
        Index("ix_user_roles_expires", "expires_at"),
    )
    
    def __repr__(self) -> str:
        return f"<UserRole user={self.user_id} role={self.role_id}>"
    
    @property
    def is_expired(self) -> bool:
        """Check if role assignment has expired."""
        if self.expires_at is None:
            return False
        return self.expires_at < datetime.now()
