"""Health and system status endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.bootstrap import bootstrap_service
from app.core.circuit_breaker import circuit_breaker_registry
from app.core.resource_profiler import resource_profiler
from app.core.llm_integration import get_llm_status

router = APIRouter()


@router.get("")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok",
        "bootstrap_complete": bootstrap_service.bootstrap_complete,
    }


@router.get("/bootstrap")
async def bootstrap_status():
    """Get bootstrap initialization status."""
    return bootstrap_service.get_status()


@router.get("/circuit-breakers")
async def circuit_breaker_status():
    """Get circuit breaker status for all services."""
    all_status = circuit_breaker_registry.get_all_status()
    
    # Separate by type
    llm_breakers = {
        name: data for name, data in all_status.items()
        if name.startswith("llm_")
    }
    other_breakers = {
        name: data for name, data in all_status.items()
        if not name.startswith("llm_")
    }
    
    # Count open breakers
    open_count = sum(
        1 for data in all_status.values()
        if data["state"] == "open"
    )
    
    return {
        "total_breakers": len(all_status),
        "open_breakers": open_count,
        "llm_providers": llm_breakers,
        "other_services": other_breakers,
    }


@router.get("/resources")
async def resource_status(
    organization_id: str = None,
    db: AsyncSession = Depends(get_db),
):
    """Get resource consumption status."""
    memory = resource_profiler.get_memory_info()
    
    result = {
        "process_memory": memory,
        "profiled_tasks": len(resource_profiler.metrics),
    }
    
    if organization_id:
        result["organization_summary"] = resource_profiler.get_org_summary(organization_id)
    
    return result


@router.get("/full")
async def full_status(
    db: AsyncSession = Depends(get_db),
):
    """Get complete system health status."""
    return {
        "bootstrap": bootstrap_service.get_status(),
        "circuit_breakers": circuit_breaker_registry.get_all_status(),
        "resources": {
            "memory": resource_profiler.get_memory_info(),
            "profiled_tasks": len(resource_profiler.metrics),
        },
    }
