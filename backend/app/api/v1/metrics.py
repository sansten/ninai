"""
Metrics API Route
Exposes Prometheus metrics for scraping
"""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.middleware.prometheus import metrics_registry

router = APIRouter(tags=["metrics"])


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
