"""
Request ID Middleware
=====================

Adds a unique request ID to each incoming request for tracing
and correlation across logs and services.
"""

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware that assigns a unique ID to each request.
    
    The request ID is:
    - Generated if not provided in X-Request-ID header
    - Stored in request.state for access in handlers
    - Added to response headers for client correlation
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request and add request ID.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
        
        Returns:
            Response with X-Request-ID header
        """
        # Get or generate request/trace ID
        # Prefer X-Trace-ID if provided, otherwise fall back to X-Request-ID.
        request_id = request.headers.get("X-Trace-ID") or request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for handlers to access
        request.state.request_id = request_id
        # Alias for clarity in agent/pipeline code.
        request.state.trace_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request/trace ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = request_id
        
        return response
