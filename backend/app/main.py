"""
Ninai - Main Application Entry Point
====================================

This module initializes the FastAPI application with all routes, middleware,
and event handlers configured for the enterprise memory operating system.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.api.v1.router import api_router
from app.api.v1.metrics import router as metrics_router
from app.core.config import settings
from app.core.database import engine, create_db_and_tables, get_db
from app.core.bootstrap import bootstrap_service, create_default_bootstrap_checks
from app.core.feature_gate import CommunityFeatureGate, set_feature_gate
from app.core.enterprise_loader import try_register_enterprise
from app.middleware.audit_logger import AuditLoggerMiddleware
from app.middleware.rate_limiter import init_rate_limiter
from app.middleware.request_id import RequestIdMiddleware
from app.middleware.prometheus import PrometheusMiddleware, StructuredLoggingMiddleware
import logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize database connections, caches, bootstrap system checks
    - Shutdown: Close connections gracefully
    """
    # Startup
    if settings.APP_ENV != "test":
        try:
            await create_db_and_tables()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not create database tables: {e} - continuing without database")
        
        # Initialize rate limiter with Redis (optional)
        try:
            from app.core.redis import get_redis_client
            redis_client = await get_redis_client()
            init_rate_limiter(redis_client)
        except Exception:
            pass  # Rate limiting is optional
    
    yield
    
    # Shutdown
    if settings.APP_ENV != "test":
        await engine.dispose()


def create_application() -> FastAPI:
    """
    Application factory function.
    
    Creates and configures the FastAPI application with all middleware,
    routes, and settings applied.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.APP_NAME,
        description="Enterprise Agentic AI Memory Operating System",
        version="1.0.0",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Feature gate (Community defaults). Enterprise can replace this at runtime.
    set_feature_gate(app, CommunityFeatureGate())

    # ---------------------------------------------------------------------------
    # Middleware (order matters - first added = last executed)
    # ---------------------------------------------------------------------------
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request ID middleware (adds unique ID to each request)
    app.add_middleware(RequestIdMiddleware)
    
    # Audit logging middleware (logs request/response metadata)
    app.add_middleware(AuditLoggerMiddleware)
    
    # Structured logging middleware (JSON logs with correlation IDs)
    logger = logging.getLogger(__name__)
    app.add_middleware(StructuredLoggingMiddleware, logger=logger)
    
    # Prometheus metrics middleware (collects request/response metrics)
    app.add_middleware(PrometheusMiddleware)

    # ---------------------------------------------------------------------------
    # Routes
    # ---------------------------------------------------------------------------
    
    # Health check endpoint (no auth required)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Health check endpoint for container orchestration.
        
        Returns:
            dict: Health status with application name
        """
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/docs")

    # Metrics endpoint (Prometheus metrics)
    app.include_router(metrics_router, prefix="")
    
    # API v1 routes
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # Optional: register Enterprise add-ons (routes/tasks/gates) if installed.
    try_register_enterprise(app)

    return app


# Create application instance
app = create_application()
