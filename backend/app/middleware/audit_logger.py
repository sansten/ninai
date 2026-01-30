"""
Audit Logger Middleware
=======================

Structured logging middleware that captures request metadata
for security auditing and operational monitoring.
"""

import time
import logging
from typing import Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings


# Configure structured logger
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("audit")


class AuditLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured audit logging of all requests.
    
    Captures:
    - Request method, path, and query parameters
    - Client IP and User-Agent
    - Request ID for correlation
    - Response status and latency
    - User and org context (when available)
    """
    
    # Paths to exclude from logging (e.g., health checks)
    EXCLUDED_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Log request and response metadata.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
        
        Returns:
            Response from handler
        """
        # Skip logging for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # Start timing
        start_time = time.perf_counter()
        
        # Extract request metadata
        request_id = getattr(request.state, "request_id", "unknown")
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        
        # Extract auth context if available
        user_id = getattr(request.state, "user_id", None)
        org_id = getattr(request.state, "org_id", None)
        
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Build log context
        log_context = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "client_ip": client_ip,
            "user_agent": user_agent,
        }
        
        # Add auth context if available
        if user_id:
            log_context["user_id"] = user_id
        if org_id:
            log_context["org_id"] = org_id
        
        # Log based on status code
        if response.status_code >= 500:
            logger.error("request_completed", **log_context)
        elif response.status_code >= 400:
            logger.warning("request_completed", **log_context)
        else:
            logger.info("request_completed", **log_context)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.
        
        Handles X-Forwarded-For header for proxied requests.
        
        Args:
            request: HTTP request
        
        Returns:
            str: Client IP address
        """
        # Check X-Forwarded-For header (set by proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
