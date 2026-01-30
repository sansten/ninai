"""
Prometheus Metrics Middleware
Collects metrics on HTTP requests, responses, and custom application metrics
"""

import time
import json
from typing import Callable
from datetime import datetime
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry

# Create a global registry for metrics
metrics_registry = CollectorRegistry()

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=metrics_registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
    registry=metrics_registry
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint'],
    registry=metrics_registry
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status'],
    registry=metrics_registry
)

# Authentication Metrics
auth_attempts_total = Counter(
    'auth_attempts_total',
    'Total authentication attempts',
    ['method', 'status'],
    registry=metrics_registry
)

# Database Metrics
db_connections_active = Gauge(
    'db_connections_active',
    'Active database connections',
    registry=metrics_registry
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    registry=metrics_registry
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_name'],
    registry=metrics_registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_name'],
    registry=metrics_registry
)

# Error Metrics
errors_total = Counter(
    'errors_total',
    'Total errors by type',
    ['error_type', 'endpoint'],
    registry=metrics_registry
)

# Business Metrics
memories_created_total = Counter(
    'memories_created_total',
    'Total memories created',
    registry=metrics_registry
)

users_created_total = Counter(
    'users_created_total',
    'Total users created',
    registry=metrics_registry
)

# API Rate Limiting Metrics
rate_limit_exceeded_total = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit exceeded responses',
    ['endpoint'],
    registry=metrics_registry
)

# WebSocket Metrics
websocket_connections_active = Gauge(
    'websocket_connections_active',
    'Active WebSocket connections',
    registry=metrics_registry
)

# Job Metrics
celery_tasks_total = Counter(
    'celery_tasks_total',
    'Total Celery tasks',
    ['task_name', 'status'],
    registry=metrics_registry
)

celery_task_duration_seconds = Histogram(
    'celery_task_duration_seconds',
    'Celery task duration in seconds',
    ['task_name'],
    registry=metrics_registry
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics for HTTP requests
    """
    
    # Endpoints to skip (health checks, metrics endpoint, etc)
    SKIP_ENDPOINTS = ['/health', '/metrics', '/docs', '/openapi.json', '/redoc']
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics
        """
        # Skip metrics collection for certain endpoints
        if any(request.url.path.startswith(skip) for skip in self.SKIP_ENDPOINTS):
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Get request info
        method = request.method
        endpoint = request.url.path
        
        # Try to get request body size
        request_body_size = 0
        try:
            if hasattr(request, '_body'):
                request_body_size = len(request._body)
        except Exception:
            pass
        
        try:
            # Call the next middleware/endpoint
            response = await call_next(request)
            
            # Record metrics
            status_code = response.status_code
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get response body size (approximation)
            response_body_size = 0
            try:
                if hasattr(response, 'body'):
                    response_body_size = len(response.body)
            except Exception:
                pass
            
            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_body_size)
            
            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
                status=status_code
            ).observe(response_body_size)
            
            # Add custom headers for tracing
            response.headers["X-Response-Time"] = str(duration)
            
            return response
            
        except Exception as exc:
            # Record error
            duration = time.time() - start_time
            
            # Record in error metrics
            errors_total.labels(
                error_type=type(exc).__name__,
                endpoint=endpoint
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            raise


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured JSON logging with correlation IDs
    """
    
    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request/response in structured JSON format
        """
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get('X-Correlation-ID', self._generate_correlation_id())
        
        # Store correlation ID in request state
        request.state.correlation_id = correlation_id
        
        # Record start time
        start_time = time.time()
        
        # Log request
        request_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "event": "http_request_start",
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params) if request.query_params else {},
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get('user-agent'),
        }
        
        self.logger.info(json.dumps(request_log))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record duration
            duration = time.time() - start_time
            
            # Log response
            response_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id,
                "event": "http_request_complete",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_seconds": duration,
            }
            
            self.logger.info(json.dumps(response_log))
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            
            return response
            
        except Exception as exc:
            # Log error
            error_log = {
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": correlation_id,
                "event": "http_request_error",
                "method": request.method,
                "path": request.url.path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            
            self.logger.error(json.dumps(error_log))
            raise
    
    @staticmethod
    def _generate_correlation_id() -> str:
        """Generate a unique correlation ID"""
        import uuid
        return str(uuid.uuid4())


# Metric update functions for application events

def record_auth_attempt(method: str, success: bool) -> None:
    """Record an authentication attempt"""
    status = "success" if success else "failure"
    auth_attempts_total.labels(method=method, status=status).inc()


def record_memory_created() -> None:
    """Record a memory creation event"""
    memories_created_total.inc()


def record_user_created() -> None:
    """Record a user creation event"""
    users_created_total.inc()


def record_cache_hit(cache_name: str) -> None:
    """Record a cache hit"""
    cache_hits_total.labels(cache_name=cache_name).inc()


def record_cache_miss(cache_name: str) -> None:
    """Record a cache miss"""
    cache_misses_total.labels(cache_name=cache_name).inc()


def record_celery_task(task_name: str, success: bool, duration: float) -> None:
    """Record a Celery task execution"""
    status = "success" if success else "failure"
    celery_tasks_total.labels(task_name=task_name, status=status).inc()
    celery_task_duration_seconds.labels(task_name=task_name).observe(duration)


def record_error(error_type: str, endpoint: str) -> None:
    """Record an application error"""
    errors_total.labels(error_type=error_type, endpoint=endpoint).inc()


def record_rate_limit_exceeded(endpoint: str) -> None:
    """Record a rate limit exceeded event"""
    rate_limit_exceeded_total.labels(endpoint=endpoint).inc()


def update_db_connections(count: int) -> None:
    """Update active database connection count"""
    db_connections_active.set(count)


def record_db_query(query_type: str, duration: float) -> None:
    """Record a database query"""
    db_query_duration_seconds.labels(query_type=query_type).observe(duration)
