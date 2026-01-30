"""
Organization Models
===================

Models for organizations and organizational hierarchy (divisions, departments, teams).
Uses PostgreSQL ltree for efficient hierarchical queries.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import String, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, UUIDMixin, TimestampMixin, LtreeType


class Organization(Base, UUIDMixin, TimestampMixin):
    """
    Organization (tenant) model.
    
    Organizations are the primary tenant boundary in the system.
    All data is isolated by organization_id.
    
    Attributes:
        name: Display name for the organization
        slug: URL-friendly identifier
        settings: JSON configuration for the org
        is_active: Whether org is active (soft delete)
        parent_org_id: For subsidiary organizations (optional)
    """
    
    __tablename__ = "organizations"
    
    # Basic info
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Organization display name",
    )
    slug: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        doc="URL-friendly identifier",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Organization description",
    )
    
    # Configuration
    settings: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Organization-specific settings (JSON)",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether organization is active",
    )
    
    # Hierarchy (for subsidiaries)
    parent_org_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id"),
        nullable=True,
        doc="Parent organization ID (for subsidiaries)",
    )
    
    # Relationships
    hierarchy_nodes: Mapped[List["OrganizationHierarchy"]] = relationship(
        "OrganizationHierarchy",
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Organization {self.slug}>"


class OrganizationHierarchy(Base, UUIDMixin, TimestampMixin):
    """
    Organizational hierarchy node.
    
    Represents divisions, departments, and teams within an organization.
    Uses ltree path for efficient ancestor/descendant queries.
    
    Hierarchy levels:
    - division: Top-level organizational unit
    - department: Within a division
    - team: Within a department
    
    Attributes:
        organization_id: Parent organization
        name: Node display name
        node_type: Type of node (division/department/team)
        path: Ltree path for hierarchical queries
        parent_id: Parent node in hierarchy
    """
    
    __tablename__ = "organization_hierarchy"
    
    # Organization (tenant)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this node belongs to",
    )
    
    # Node info
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Node display name",
    )
    node_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        doc="Type: division, department, or team",
    )
    
    # Hierarchy
    path: Mapped[str] = mapped_column(
        LtreeType(),
        nullable=False,
        doc="Ltree path for hierarchical queries (e.g., 'org1.div1.dept1.team1')",
    )
    parent_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organization_hierarchy.id"),
        nullable=True,
        doc="Parent node ID",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether node is active",
    )
    
    # Node settings/metadata
    settings: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Additional node settings (JSON)",
    )
    
    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization",
        back_populates="hierarchy_nodes",
    )
    
    __table_args__ = (
        # Composite index for org + path queries
        Index("ix_org_hierarchy_org_path", "organization_id", "path"),
        # Index for parent lookups
        Index("ix_org_hierarchy_parent", "parent_id"),
    )
    
    def __repr__(self) -> str:
        return f"<OrganizationHierarchy {self.path}>"
