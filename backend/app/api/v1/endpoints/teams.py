"""
Team Endpoints
==============

Team management and membership operations.
"""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    require_roles,
)
from app.models.team import Team, TeamMember
from app.schemas.team import (
    TeamCreate,
    TeamUpdate,
    TeamResponse,
    TeamMemberAdd,
    TeamMemberResponse,
    TeamMembersResponse,
)


router = APIRouter()


@router.get("", response_model=List[TeamResponse])
async def list_teams(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    List all teams in the organization.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string
    )
    
    result = await db.execute(
        select(Team).where(
            Team.organization_id == tenant.org_id,
            Team.is_active == True,
        )
    )
    teams = result.scalars().all()
    
    return [TeamResponse.model_validate(t) for t in teams]


@router.post("", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    body: TeamCreate,
    tenant: TenantContext = Depends(require_roles("org_admin", "department_manager", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new team.
    
    **Required role:** org_admin, department_manager, or system_admin
    """
    # Check for duplicate slug
    result = await db.execute(
        select(Team).where(
            Team.organization_id == tenant.org_id,
            Team.slug == body.slug,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Team slug already exists in this organization",
        )
    
    team = Team(
        organization_id=tenant.org_id,
        name=body.name,
        slug=body.slug,
        description=body.description,
        hierarchy_node_id=body.hierarchy_node_id,
        settings=body.settings or {},
    )
    
    db.add(team)
    await db.commit()
    await db.refresh(team)
    
    return TeamResponse.model_validate(team)


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get team by ID.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string
    )
    
    result = await db.execute(
        select(Team).where(
            Team.id == team_id,
            Team.organization_id == tenant.org_id,
        )
    )
    team = result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )
    
    return TeamResponse.model_validate(team)


@router.patch("/{team_id}", response_model=TeamResponse)
async def update_team(
    team_id: str,
    body: TeamUpdate,
    tenant: TenantContext = Depends(require_roles("org_admin", "team_lead", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a team.
    
    **Required role:** org_admin, team_lead (of this team), or system_admin
    """
    result = await db.execute(
        select(Team).where(
            Team.id == team_id,
            Team.organization_id == tenant.org_id,
        )
    )
    team = result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )
    
    # Update fields
    if body.name is not None:
        team.name = body.name
    if body.description is not None:
        team.description = body.description
    if body.settings is not None:
        team.settings = body.settings
    if body.is_active is not None:
        team.is_active = body.is_active
    
    await db.commit()
    await db.refresh(team)
    
    return TeamResponse.model_validate(team)


@router.delete("/{team_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_team(
    team_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a team (soft delete).
    
    **Required role:** org_admin or system_admin
    """
    result = await db.execute(
        select(Team).where(
            Team.id == team_id,
            Team.organization_id == tenant.org_id,
        )
    )
    team = result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )
    
    team.is_active = False
    await db.commit()


# =============================================================================
# Team Members
# =============================================================================

@router.get("/{team_id}/members", response_model=TeamMembersResponse)
async def get_team_members(
    team_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get team with members list.
    """
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string
    )
    
    # Get team
    team_result = await db.execute(
        select(Team).where(
            Team.id == team_id,
            Team.organization_id == tenant.org_id,
        )
    )
    team = team_result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )
    
    # Get members
    members_result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.is_active == True,
        )
    )
    members = members_result.scalars().all()
    
    return TeamMembersResponse(
        team=TeamResponse.model_validate(team),
        members=[TeamMemberResponse.model_validate(m) for m in members],
    )


@router.post("/{team_id}/members", response_model=TeamMemberResponse, status_code=status.HTTP_201_CREATED)
async def add_team_member(
    team_id: str,
    body: TeamMemberAdd,
    tenant: TenantContext = Depends(require_roles("org_admin", "team_lead", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a member to a team.
    
    **Required role:** org_admin, team_lead (of this team), or system_admin
    """
    # Check team exists
    team_result = await db.execute(
        select(Team).where(
            Team.id == team_id,
            Team.organization_id == tenant.org_id,
        )
    )
    team = team_result.scalar_one_or_none()
    
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )
    
    # Check for existing membership
    existing = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == body.user_id,
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a member of this team",
        )
    
    member = TeamMember(
        team_id=team_id,
        user_id=body.user_id,
        organization_id=tenant.org_id,
        role=body.role,
        joined_at=datetime.now(timezone.utc),
    )
    
    db.add(member)
    await db.commit()
    await db.refresh(member)
    
    return TeamMemberResponse.model_validate(member)


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_team_member(
    team_id: str,
    user_id: str,
    tenant: TenantContext = Depends(require_roles("org_admin", "team_lead", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a member from a team.
    
    **Required role:** org_admin, team_lead (of this team), or system_admin
    """
    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
        )
    )
    member = result.scalar_one_or_none()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team member not found",
        )
    
    member.is_active = False
    member.left_at = datetime.now(timezone.utc)
    
    await db.commit()
