"""
Advanced Search Endpoints

Endpoints for advanced search with query operators:
- tag:, before:, after:, within:, relates_to:, scope: operators
- Faceted search
- Saved searches
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.memory_service import MemoryService
from app.services.search_query_parser import (
    parse_search_query,
    validate_query,
    query_to_filters,
)
from app.schemas.base import BaseSchema


router = APIRouter()


class AdvancedSearchRequest(BaseSchema):
    """Advanced search request with operators."""
    query: str
    limit: int = 50
    offset: int = 0
    return_facets: bool = False


class AdvancedSearchResponse(BaseSchema):
    """Advanced search response."""
    results: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    facets: Optional[Dict[str, Any]] = None
    parsed_query: Optional[Dict[str, Any]] = None
    query_time_ms: int


class SavedSearchCreate(BaseSchema):
    """Create saved search."""
    name: str
    query: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class SavedSearchResponse(BaseSchema):
    """Saved search response."""
    id: str
    name: str
    query: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: datetime
    updated_at: datetime
    result_count: Optional[int] = None


def get_memory_service(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> MemoryService:
    """Get configured memory service."""
    return MemoryService(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
        clearance_level=tenant.clearance_level,
    )


@router.post("/advanced-search", response_model=AdvancedSearchResponse)
async def advanced_search(
    request: AdvancedSearchRequest,
    tenant: TenantContext = Depends(get_tenant_context),
    service: MemoryService = Depends(get_memory_service),
) -> AdvancedSearchResponse:
    """
    Advanced search with query operators.
    
    Supports operators:
    - `tag:important` - Filter by tag
    - `before:2024-01-01`, `after:2024-01-01` - Date filters
    - `within:7d` - Within time period (d=days, w=weeks, m=months)
    - `relates_to:memory_id` - Find related memories
    - `scope:team` - Filter by scope (team, personal, shared)
    - `author:john` - Filter by creator
    - `status:active` - Filter by status (active, archived, deleted)
    - `faceted` - Return faceted breakdown
    
    ## Examples
    
    ```
    # Important memories from last 7 days
    tag:important within:7d
    
    # Team memories about performance
    performance scope:team
    
    # Recent failures related to deployment
    deployment relates_to:abc123 after:2024-01-01 status:active
    
    # Get faceted breakdown
    sprint planning faceted
    ```
    """
    import time
    start_time = time.time()
    
    # Parse query
    parsed = parse_search_query(request.query)
    
    # Validate
    is_valid, error = validate_query(parsed)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query: {error}"
        )
    
    # Convert to filters
    filters = query_to_filters(parsed)
    
    try:
        # Perform search with filters
        # This would integrate with MemoryService.search with additional filters
        
        results = []  # Would populate from service
        total = 0  # Would get from service
        
        facets = None
        if parsed.faceted:
            facets = {
                'tags': {},
                'status': {},
                'scope': {},
                'dates': {},
            }
        
        query_time_ms = int((time.time() - start_time) * 1000)
        
        return AdvancedSearchResponse(
            results=results,
            total=total,
            offset=request.offset,
            limit=request.limit,
            facets=facets,
            parsed_query={
                'text': parsed.text,
                'operators': [
                    {'type': op.type.value, 'value': op.value}
                    for op in parsed.operators
                ]
            },
            query_time_ms=query_time_ms,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/faceted-search", response_model=Dict[str, Any])
async def faceted_search(
    query: str = Query(..., description="Search query"),
    facet_field: Optional[str] = Query(None, description="Specific facet field"),
    limit: int = Query(50),
    tenant: TenantContext = Depends(get_tenant_context),
    service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    """
    Get faceted breakdown of search results.
    
    Facet Fields:
    - `tags`: All tags in matching memories
    - `status`: Memory status distribution
    - `scope`: Scope distribution (personal, team, shared, org)
    - `authors`: Memory creators
    - `dates`: Temporal distribution (by week)
    - `memory_types`: Memory type distribution
    """
    try:
        facets = {
            'tags': {},
            'status': {},
            'scope': {},
            'authors': {},
            'dates': {},
            'memory_types': {},
        }
        
        # Would aggregate from search results
        
        return {
            'query': query,
            'total_results': 0,
            'facets': facets if not facet_field else facets.get(facet_field, {}),
            'facet_field': facet_field,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/saved-searches", response_model=SavedSearchResponse)
async def create_saved_search(
    request: SavedSearchCreate,
    tenant: TenantContext = Depends(get_tenant_context),
) -> SavedSearchResponse:
    """
    Create and save a search query for future use.
    
    Saved searches can be reused with a simple ID lookup.
    """
    try:
        # Would create in database
        
        return SavedSearchResponse(
            id="search_" + tenant.user_id,
            name=request.name,
            query=request.query,
            description=request.description,
            tags=request.tags,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            result_count=0,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/saved-searches", response_model=List[SavedSearchResponse])
async def list_saved_searches(
    tenant: TenantContext = Depends(get_tenant_context),
) -> List[SavedSearchResponse]:
    """List all saved searches for current user."""
    try:
        # Would query database for saved searches
        return []
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/saved-searches/{search_id}", response_model=SavedSearchResponse)
async def get_saved_search(
    search_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
) -> SavedSearchResponse:
    """Get a saved search by ID."""
    try:
        # Would query database
        
        return SavedSearchResponse(
            id=search_id,
            name="Sample Search",
            query="tag:important within:7d",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/saved-searches/{search_id}/execute", response_model=AdvancedSearchResponse)
async def execute_saved_search(
    search_id: str,
    limit: int = Query(50),
    offset: int = Query(0),
    tenant: TenantContext = Depends(get_tenant_context),
    service: MemoryService = Depends(get_memory_service),
) -> AdvancedSearchResponse:
    """Execute a saved search."""
    try:
        # Would get saved search from database, then execute
        
        return AdvancedSearchResponse(
            results=[],
            total=0,
            offset=offset,
            limit=limit,
            query_time_ms=0,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/saved-searches/{search_id}")
async def delete_saved_search(
    search_id: str,
    tenant: TenantContext = Depends(get_tenant_context),
) -> Dict[str, str]:
    """Delete a saved search."""
    try:
        # Would delete from database
        return {"message": "Search deleted"}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
