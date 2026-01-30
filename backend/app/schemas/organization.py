"""
Organization Schemas
====================

Request and response schemas for organization and hierarchy operations.
"""

from typing import Optional, List
from datetime import datetime
from enum import Enum

from pydantic import Field

from app.schemas.base import BaseSchema


class OrganizationCreate(BaseSchema):
    """Request schema for creating an organization."""
    
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    description: Optional[str] = None
    settings: Optional[dict] = None
    parent_org_id: Optional[str] = None


class OrganizationUpdate(BaseSchema):
    """Request schema for updating an organization."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class OrganizationResponse(BaseSchema):
    """Response schema for organization data."""
    
    id: str
    name: str
    slug: str
    description: Optional[str]
    settings: dict
    is_active: bool
    parent_org_id: Optional[str]
    created_at: datetime
    updated_at: datetime


class OrganizationSummary(BaseSchema):
    """Minimal organization info for lists."""
    
    id: str
    name: str
    slug: str
    is_active: bool


# =============================================================================
# Hierarchy Schemas
# =============================================================================

class HierarchyNodeType(str, Enum):
    """Valid hierarchy node types."""
    DIVISION = "division"
    DEPARTMENT = "department"
    TEAM = "team"


class HierarchyNodeCreate(BaseSchema):
    """Request schema for creating a hierarchy node."""
    
    name: str = Field(..., min_length=1, max_length=255)
    node_type: HierarchyNodeType
    parent_id: Optional[str] = None
    settings: Optional[dict] = None


class HierarchyNodeUpdate(BaseSchema):
    """Request schema for updating a hierarchy node."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    settings: Optional[dict] = None
    is_active: Optional[bool] = None


class HierarchyNodeResponse(BaseSchema):
    """Response schema for hierarchy node data."""
    
    id: str
    organization_id: str
    name: str
    node_type: str
    path: str
    parent_id: Optional[str]
    is_active: bool
    settings: dict
    created_at: datetime
    updated_at: datetime
    
    # Optional children for tree responses
    children: Optional[List["HierarchyNodeResponse"]] = None


class HierarchyTreeResponse(BaseSchema):
    """Response schema for full hierarchy tree."""
    
    organization_id: str
    organization_name: str
    nodes: List[HierarchyNodeResponse]


# Allow recursive reference
HierarchyNodeResponse.model_rebuild()
