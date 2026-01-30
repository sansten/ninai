"""
Team Schemas
============

Request and response schemas for team operations.
"""

from typing import Optional, List
from datetime import datetime

from pydantic import Field

from app.schemas.base import BaseSchema


class TeamCreate(BaseSchema):
    """Request schema for creating a team."""
    
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    description: Optional[str] = None
    hierarchy_node_id: Optional[str] = None
    settings: Optional[dict] = None


class TeamUpdate(BaseSchema):
    """Request schema for updating a team."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class TeamResponse(BaseSchema):
    """Response schema for team data."""
    
    id: str
    organization_id: str
    name: str
    slug: str
    description: Optional[str]
    hierarchy_node_id: Optional[str]
    settings: dict
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    # Optional member count
    member_count: Optional[int] = None


class TeamSummary(BaseSchema):
    """Minimal team info for lists."""
    
    id: str
    name: str
    slug: str
    is_active: bool


# =============================================================================
# Team Member Schemas
# =============================================================================

class TeamMemberAdd(BaseSchema):
    """Request schema for adding a team member."""
    
    user_id: str
    role: str = "member"  # member, lead, admin


class TeamMemberUpdate(BaseSchema):
    """Request schema for updating a team member."""
    
    role: Optional[str] = None
    is_active: Optional[bool] = None


class TeamMemberResponse(BaseSchema):
    """Response schema for team member data."""
    
    id: str
    team_id: str
    user_id: str
    role: str
    is_active: bool
    joined_at: datetime
    left_at: Optional[datetime]
    
    # Optional user details
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class TeamMembersResponse(BaseSchema):
    """Response with team and members list."""
    
    team: TeamResponse
    members: List[TeamMemberResponse]


class UserTeamsResponse(BaseSchema):
    """Response with user's team memberships."""
    
    teams: List[TeamResponse]
