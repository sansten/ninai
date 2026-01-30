"""
Team Models
===========

Models for teams and team membership within organizations.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import String, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, UUIDMixin, TimestampMixin


class Team(Base, UUIDMixin, TimestampMixin):
    """
    Team model.
    
    Teams are the primary unit for collaboration and memory sharing.
    Teams belong to an organization and optionally to a hierarchy node.
    
    Attributes:
        organization_id: Parent organization
        name: Team display name
        hierarchy_node_id: Link to organizational hierarchy
        settings: Team-specific configuration
    """
    
    __tablename__ = "teams"
    
    # Organization (tenant)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization this team belongs to",
    )
    
    # Basic info
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Team display name",
    )
    slug: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        doc="URL-friendly identifier",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Team description",
    )
    
    # Hierarchy link
    hierarchy_node_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organization_hierarchy.id"),
        nullable=True,
        doc="Link to organizational hierarchy node",
    )
    
    # Configuration
    settings: Mapped[dict] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
        doc="Team-specific settings (JSON)",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether team is active",
    )
    
    # Relationships
    members: Mapped[List["TeamMember"]] = relationship(
        "TeamMember",
        back_populates="team",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        # Unique slug per organization
        Index("ix_teams_slug_org", "organization_id", "slug", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<Team {self.name}>"


class TeamMember(Base, UUIDMixin, TimestampMixin):
    """
    Team membership model.
    
    Links users to teams with role information.
    
    Attributes:
        team_id: Team the user belongs to
        user_id: User who is a member
        role: Role within the team (member, lead, admin)
        is_active: Whether membership is active
    """
    
    __tablename__ = "team_members"
    
    # Team reference
    team_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Team ID",
    )
    
    # User reference
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="User ID",
    )
    
    # Organization (for RLS)
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="Organization ID (for RLS policies)",
    )
    
    # Team role
    role: Mapped[str] = mapped_column(
        String(50),
        default="member",
        nullable=False,
        doc="Role in team: member, lead, admin",
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether membership is active",
    )
    
    # Join/leave tracking
    joined_at: Mapped[datetime] = mapped_column(
        nullable=False,
        doc="When user joined the team",
    )
    left_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When user left the team (if inactive)",
    )
    
    # Relationships
    team: Mapped["Team"] = relationship(
        "Team",
        back_populates="members",
    )
    
    __table_args__ = (
        # Prevent duplicate memberships
        Index("ix_team_members_unique", "team_id", "user_id", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<TeamMember team={self.team_id} user={self.user_id}>"
