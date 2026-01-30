"""
FalkorDB Graph API Endpoints
Provides graph-based relationship queries and analysis using Redis-based FalkorDB
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.graph_service import get_graph_service
from pydantic import BaseModel, Field


router = APIRouter()


class GraphNodeResponse(BaseModel):
    """Graph node information."""
    id: str
    labels: list[str] = []
    properties: dict = Field(default_factory=dict)


class GraphRelationshipResponse(BaseModel):
    """Graph relationship information."""
    type: str
    properties: dict = Field(default_factory=dict)


class GraphPathResponse(BaseModel):
    """Graph path between nodes."""
    length: int
    nodes: list[dict]
    relationships: list[dict]


class GraphDegreeResponse(BaseModel):
    """Node degree (connection count)."""
    node_id: str
    in_degree: int = Field(alias="in")
    out_degree: int = Field(alias="out")
    total_degree: int = Field(alias="total")


class GraphStatsResponse(BaseModel):
    """Graph statistics."""
    enabled: bool
    total_nodes: int
    total_relationships: int
    label_types: int = 0


class RelatedMemoriesResponse(BaseModel):
    """Related memories response."""
    memory_id: str
    related: list[dict]
    total: int


@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_statistics(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get graph statistics for the organization.
    
    Returns node counts, relationship counts, and graph metrics.
    """
    graph_svc = get_graph_service()
    
    if not graph_svc.redis:
        return GraphStatsResponse(
            enabled=False,
            total_nodes=0,
            total_relationships=0
        )
    
    stats = await graph_svc.get_graph_statistics(tenant.org_id)
    return GraphStatsResponse(**stats)


@router.get("/memories/{memory_id}/related", response_model=RelatedMemoriesResponse)
async def get_related_memories(
    memory_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
    relationship_types: list[str] = Query(None, description="Filter by relationship types"),
    max_depth: int = Query(2, ge=1, le=5, description="Maximum traversal depth"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
):
    """
    Find memories related to a specific memory via graph traversal.
    
    Uses FalkorDB (Redis-based) graph database to traverse relationships and find connected memories.
    """
    graph_svc = get_graph_service()
    
    if not graph_svc.redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph database not available"
        )
    
    related = await graph_svc.find_related_memories(
        memory_id=memory_id,
        org_id=tenant.org_id,
        relationship_types=relationship_types,
        max_depth=max_depth,
        limit=limit
    )
    
    return RelatedMemoriesResponse(
        memory_id=memory_id,
        related=related,
        total=len(related)
    )


@router.get("/path", response_model=GraphPathResponse)
async def find_shortest_path(
    from_id: str = Query(..., description="Source node ID"),
    to_id: str = Query(..., description="Target node ID"),
    max_depth: int = Query(5, ge=1, le=10, description="Maximum path length"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Find shortest path between two nodes in the graph.
    
    Returns the shortest path with nodes and relationships.
    """
    graph_svc = get_graph_service()
    
    if not graph_svc.redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph database not available"
        )
    
    path = await graph_svc.find_shortest_path(
        from_id=from_id,
        to_id=to_id,
        org_id=tenant.org_id,
        max_depth=max_depth
    )
    
    if not path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No path found between {from_id} and {to_id}"
        )
    
    return GraphPathResponse(**path)


@router.get("/nodes/{node_id}/degree", response_model=GraphDegreeResponse)
async def get_node_degree(
    node_id: str,
    direction: str = Query("both", regex="^(in|out|both)$"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Get the degree (number of connections) for a node.
    
    Returns in-degree, out-degree, and total degree.
    """
    graph_svc = get_graph_service()
    
    if not graph_svc.redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph database not available"
        )
    
    degree = await graph_svc.get_node_degree(
        node_id=node_id,
        org_id=tenant.org_id,
        direction=direction
    )
    
    return GraphDegreeResponse(
        node_id=node_id,
        **degree
    )


@router.post("/memories/{memory_id}/relationships")
async def create_memory_relationship(
    memory_id: str,
    target_id: str = Query(..., description="Target memory ID"),
    relationship_type: str = Query("RELATES_TO", description="Relationship type"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a relationship between two memories in the graph.
    
    Relationship types:
    - RELATES_TO: General relationship
    - DEPENDS_ON: Dependency relationship
    - REFERENCES: Reference relationship
    - FOLLOWS: Sequential relationship
    """
    graph_svc = get_graph_service()
    
    if not graph_svc.redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph database not available"
        )
    
    # Validate relationship type
    valid_types = ["RELATES_TO", "DEPENDS_ON", "REFERENCES", "FOLLOWS", "BLOCKS"]
    if relationship_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid relationship type. Must be one of: {valid_types}"
        )
    
    relationship = await graph_svc.create_relationship(
        from_id=memory_id,
        to_id=target_id,
        relationship_type=relationship_type,
        properties=properties or {}
    )
    
    return {
        "success": True,
        "from_id": memory_id,
        "to_id": target_id,
        "relationship_type": relationship_type,
        "relationship": relationship
    }
