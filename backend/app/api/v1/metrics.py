"""Metrics endpoints including cognitive observability surfaces (Feature 21)."""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.database import get_db, set_tenant_context
from app.middleware.prometheus import metrics_registry
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.services.cognitive_metrics import CognitiveMetricsService


router = APIRouter(tags=["metrics"])
_cognitive_metrics_service = CognitiveMetricsService()


@router.get("/metrics", include_in_schema=False)
async def metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus exposition format
    Scraped by Prometheus server
    """
    return Response(
        content=generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/metrics/cognitive", include_in_schema=False)
async def cognitive_metrics():
    """Prometheus-compatible cognitive metrics surface."""
    return Response(
        content=generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST,
    )


def _ensure_metrics_capability(tenant: TenantContext) -> None:
    if not tenant.has_capability("canViewMetrics"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


@router.get("/metrics/cognitive/summary")
async def cognitive_metrics_summary(
    tenant: TenantContext = Depends(get_tenant_context),
    db=Depends(get_db),
):
    _ensure_metrics_capability(tenant)
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    return await _cognitive_metrics_service.get_summary(db=db, org_id=tenant.org_id)


@router.get("/metrics/agents")
async def cognitive_agent_metrics(
    tenant: TenantContext = Depends(get_tenant_context),
    db=Depends(get_db),
):
    _ensure_metrics_capability(tenant)
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    return await _cognitive_metrics_service.get_agent_metrics(db=db, org_id=tenant.org_id)


@router.get("/metrics/memory")
async def cognitive_memory_metrics(
    tenant: TenantContext = Depends(get_tenant_context),
    db=Depends(get_db),
):
    _ensure_metrics_capability(tenant)
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    return await _cognitive_metrics_service.get_memory_metrics(db=db, org_id=tenant.org_id)


@router.get("/metrics/events")
async def cognitive_event_metrics(
    tenant: TenantContext = Depends(get_tenant_context),
    db=Depends(get_db),
):
    _ensure_metrics_capability(tenant)
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    return await _cognitive_metrics_service.get_event_metrics(db=db, org_id=tenant.org_id)

