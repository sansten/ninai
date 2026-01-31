"""Admin UI database models"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, INET, JSONB
from sqlalchemy.orm import relationship

from app.models.base import Base
from sqlalchemy.dialects.postgresql import UUID


class AdminRole(Base):
    """Admin role model with granular permissions"""
    __tablename__ = "admin_roles"

    id = Column(UUID(as_uuid=False), primary_key=True)
    name = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    permissions = Column(ARRAY(String), nullable=False, default=[])
    is_system = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    created_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True, index=True)

    # Relationships
    creator = relationship("User", foreign_keys=[created_by], primaryjoin="AdminRole.created_by==User.id")
    users = relationship("User", back_populates="admin_role", foreign_keys="User.admin_role_id")

    def __repr__(self) -> str:
        return f"<AdminRole(id={self.id}, name={self.name})>"


class AdminSetting(Base):
    """System settings model for admin configuration"""
    __tablename__ = "admin_settings"

    id = Column(UUID(as_uuid=False), primary_key=True)
    category = Column(String(50), nullable=False, index=True)
    key = Column(String(255), nullable=False)
    value = Column(JSONB, nullable=False)
    type = Column(String(50), nullable=True)  # string, number, boolean, json
    description = Column(Text, nullable=True)
    is_secret = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    updated_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True, index=True)

    # Unique constraint on category + key (matches migration expectation)
    __table_args__ = (
        UniqueConstraint("category", "key", name="uq_admin_settings_category_key"),
    )

    # Relationships
    updater = relationship("User", foreign_keys=[updated_by])

    def __repr__(self) -> str:
        return f"<AdminSetting(category={self.category}, key={self.key})>"


class AdminAuditLog(Base):
    """Admin action audit trail"""
    __tablename__ = "admin_audit_logs"

    id = Column(UUID(as_uuid=False), primary_key=True)
    admin_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String(50), nullable=False, index=True)  # create, update, delete, etc.
    resource_type = Column(String(50), nullable=False)  # user, role, setting, etc.
    resource_id = Column(String(255), nullable=True)
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    ip_address = Column(INET, nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)

    # Relationships
    admin = relationship("User", foreign_keys=[admin_id])

    def __repr__(self) -> str:
        return f"<AdminAuditLog(admin_id={self.admin_id}, action={self.action}, resource_type={self.resource_type})>"


class AdminSession(Base):
    """Admin session tracking for security"""
    __tablename__ = "admin_sessions"

    id = Column(UUID(as_uuid=False), primary_key=True)
    admin_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, unique=True)
    ip_address = Column(INET, nullable=False)
    user_agent = Column(String(500), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    last_activity = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    admin = relationship("User", foreign_keys=[admin_id])

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return self.expires_at <= datetime.utcnow()

    def __repr__(self) -> str:
        return f"<AdminSession(admin_id={self.admin_id}, expires_at={self.expires_at})>"


class AdminIPWhitelist(Base):
    """IP whitelist for admin access"""
    __tablename__ = "admin_ip_whitelist"

    id = Column(UUID(as_uuid=False), primary_key=True)
    ip_address = Column(INET, nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    created_by = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Relationships
    creator = relationship("User", foreign_keys=[created_by])

    def __repr__(self) -> str:
        return f"<AdminIPWhitelist(ip_address={self.ip_address})>"
