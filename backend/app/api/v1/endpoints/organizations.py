"""
Organization Endpoints
======================

Organization management and hierarchy operations.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import (
    TenantContext,
    get_tenant_context,
    require_roles,
)
from app.models.organization import Organization, OrganizationHierarchy
from app.schemas.organization import (
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse,
    HierarchyNodeCreate,
    HierarchyNodeResponse,
    HierarchyTreeResponse,
)


router = APIRouter()


@router.get("/me", response_model=OrganizationResponse)
async def get_current_organization(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the current organization context.
    """
    if not tenant.org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No organization context",
        )
    
    result = await db.execute(
        select(Organization).where(Organization.id == tenant.org_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )
    
    return OrganizationResponse.model_validate(org)


@router.get("", response_model=List[OrganizationResponse])
async def list_organizations(
    tenant: TenantContext = Depends(require_roles("system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    List all organizations.
    
    **Required role:** system_admin
    """
    result = await db.execute(select(Organization))
    orgs = result.scalars().all()
    
    return [OrganizationResponse.model_validate(org) for org in orgs]


@router.post("", response_model=OrganizationResponse, status_code=status.HTTP_201_CREATED)
async def create_organization(
    body: OrganizationCreate,
    tenant: TenantContext = Depends(require_roles("system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new organization.
    
    **Required role:** system_admin
    """
    # Check for duplicate slug
    result = await db.execute(
        select(Organization).where(Organization.slug == body.slug)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization slug already exists",
        )
    
    org = Organization(
        name=body.name,
        slug=body.slug,
        description=body.description,
        settings=body.settings or {},
        parent_org_id=body.parent_org_id,
    )
    
    db.add(org)
    await db.commit()
    await db.refresh(org)
    
    return OrganizationResponse.model_validate(org)


@router.get("/{org_id}", response_model=OrganizationResponse)
async def get_organization(
    org_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get organization by ID.
    """
    # Users can only access their own org unless system admin
    if org_id != tenant.org_id and not tenant.is_system_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    result = await db.execute(
        select(Organization).where(Organization.id == org_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )
    
    return OrganizationResponse.model_validate(org)


@router.patch("/{org_id}", response_model=OrganizationResponse)
async def update_organization(
    org_id: str,
    body: OrganizationUpdate,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Update organization.
    
    **Required role:** org_admin or system_admin
    """
    # Org admins can only update their own org
    if org_id != tenant.org_id and not tenant.is_system_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    result = await db.execute(
        select(Organization).where(Organization.id == org_id)
    )
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )
    
    # Update fields
    if body.name is not None:
        org.name = body.name
    if body.description is not None:
        org.description = body.description
    if body.settings is not None:
        org.settings = body.settings
    if body.is_active is not None and tenant.is_system_admin:
        org.is_active = body.is_active
    
    await db.commit()
    await db.refresh(org)
    
    return OrganizationResponse.model_validate(org)


# =============================================================================
# Hierarchy Endpoints
# =============================================================================

@router.get("/structure/tree", response_model=HierarchyTreeResponse)
async def get_hierarchy_tree(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the full organizational hierarchy tree.
    """
    # Get organization
    org_result = await db.execute(
        select(Organization).where(Organization.id == tenant.org_id)
    )
    org = org_result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )
    
    # Get all hierarchy nodes for the org
    nodes_result = await db.execute(
        select(OrganizationHierarchy)
        .where(OrganizationHierarchy.organization_id == tenant.org_id)
        .order_by(OrganizationHierarchy.path)
    )
    nodes = nodes_result.scalars().all()
    
    return HierarchyTreeResponse(
        organization_id=org.id,
        organization_name=org.name,
        nodes=[HierarchyNodeResponse.model_validate(n) for n in nodes],
    )


@router.post("/structure/nodes", response_model=HierarchyNodeResponse, status_code=status.HTTP_201_CREATED)
async def create_hierarchy_node(
    body: HierarchyNodeCreate,
    tenant: TenantContext = Depends(require_roles("org_admin", "system_admin")),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new hierarchy node (division, department, or team).
    
    **Required role:** org_admin or system_admin
    """
    # Build path
    if body.parent_id:
        parent_result = await db.execute(
            select(OrganizationHierarchy).where(
                OrganizationHierarchy.id == body.parent_id
            )
        )
        parent = parent_result.scalar_one_or_none()
        
        if not parent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parent node not found",
            )
        
        # Build path from parent
        path = f"{parent.path}.{body.name.lower().replace(' ', '_')}"
    else:
        # Root level node
        path = body.name.lower().replace(' ', '_')
    
    node = OrganizationHierarchy(
        organization_id=tenant.org_id,
        name=body.name,
        node_type=body.node_type,
        path=path,
        parent_id=body.parent_id,
        metadata=body.metadata or {},
    )
    
    db.add(node)
    await db.commit()
    await db.refresh(node)
    
    return HierarchyNodeResponse.model_validate(node)


@router.get("/structure/nodes/{node_id}", response_model=HierarchyNodeResponse)
async def get_hierarchy_node(
    node_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a hierarchy node by ID.
    """
    result = await db.execute(
        select(OrganizationHierarchy).where(
            OrganizationHierarchy.id == node_id,
            OrganizationHierarchy.organization_id == tenant.org_id,
        )
    )
    node = result.scalar_one_or_none()
    
    if not node:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found",
        )
    
    return HierarchyNodeResponse.model_validate(node)
