"""
Base Model Classes
==================

Shared base classes and mixins for all SQLAlchemy models.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import UserDefinedType


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class LtreeType(UserDefinedType):
    """
    Custom SQLAlchemy type for PostgreSQL LTREE.
    
    LTREE is a PostgreSQL extension for representing hierarchical
    tree-like structures as label paths (e.g., 'org.dept.team').
    """
    cache_ok = True
    
    def get_col_spec(self):
        return "LTREE"
    
    def bind_processor(self, dialect):
        return None
    
    def result_processor(self, dialect, coltype):
        return None


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base for all models.
    
    Provides:
    - Automatic table name generation
    - Common type annotations
    """
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }


class UUIDMixin:
    """
    Mixin that adds a UUID primary key.
    
    Uses PostgreSQL's UUID type for efficient storage and indexing.
    """
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
        doc="Unique identifier (UUID)",
    )


class TimestampMixin:
    """
    Mixin that adds created_at and updated_at timestamps.
    
    - created_at: Set automatically on insert
    - updated_at: Updated automatically on every change
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        server_default=func.now(),
        nullable=False,
        doc="Record creation timestamp",
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        server_default=func.now(),
        nullable=False,
        doc="Last update timestamp",
    )


class TenantMixin:
    """
    Mixin that adds organization_id for multi-tenant isolation.
    
    CRITICAL: All tenant-scoped tables must include this mixin
    to enable RLS-based isolation.
    """
    
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Organization this record belongs to (tenant isolation)",
    )
