"""
Metrics Enhancements - Public /metrics endpoint and Prometheus integration.

Features:
- Public /metrics endpoint for Prometheus scraping
- Detailed metrics dashboard configuration
- Alert rules for critical metrics
- Custom performance metrics
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
from prometheus_client.exposition import generate_latest
from prometheus_client.core import REGISTRY


class MetricScope(str, Enum):
    """Scope of metrics."""
    GLOBAL = "global"
    ENDPOINT = "endpoint"
    SERVICE = "service"
    DATABASE = "database"


class MetricsEnhancementService:
    """
    Enhanced metrics collection and export service.
    
    Provides:
    - Prometheus metrics collection
    - Custom application metrics
    - Performance analytics
    - Alert-worthy metric tracking
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics service.
        
        Args:
            registry: Prometheus registry (uses default if not provided)
        """
        self.registry = registry or REGISTRY
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup all custom metrics."""
        # Memory metrics
        self.memory_total = Gauge(
            'memory_total_count',
            'Total number of memories',
            registry=self.registry
        )
        self.memory_by_status = Gauge(
            'memory_by_status',
            'Memories by status',
            ['status'],
            registry=self.registry
        )
        self.memory_by_scope = Gauge(
            'memory_by_scope',
            'Memories by scope',
            ['scope'],
            registry=self.registry
        )
        
        # Relationship metrics
        self.relationships_total = Gauge(
            'relationships_total_count',
            'Total number of relationships',
            registry=self.registry
        )
        self.relationships_strength = Gauge(
            'relationships_avg_strength',
            'Average relationship strength',
            registry=self.registry
        )
        
        # Search metrics
        self.searches_total = Counter(
            'searches_total',
            'Total number of searches',
            registry=self.registry
        )
        self.search_latency = Histogram(
            'search_latency_seconds',
            'Search query latency in seconds',
            registry=self.registry
        )
        self.search_results = Histogram(
            'search_results_count',
            'Number of search results returned',
            registry=self.registry
        )
        
        # Vector DB metrics
        self.embeddings_created = Counter(
            'embeddings_created_total',
            'Total embeddings created',
            registry=self.registry
        )
        self.embeddings_cached = Counter(
            'embeddings_cached_total',
            'Total cached embeddings used',
            registry=self.registry
        )
        self.embedding_latency = Histogram(
            'embedding_latency_seconds',
            'Embedding generation latency',
            registry=self.registry
        )
        
        # Task metrics
        self.tasks_queued = Gauge(
            'tasks_queued',
            'Currently queued tasks',
            registry=self.registry
        )
        self.tasks_processing = Gauge(
            'tasks_processing',
            'Currently processing tasks',
            registry=self.registry
        )
        self.tasks_completed = Counter(
            'tasks_completed_total',
            'Total completed tasks',
            registry=self.registry
        )
        self.tasks_failed = Counter(
            'tasks_failed_total',
            'Total failed tasks',
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        self.api_latency = Histogram(
            'api_latency_seconds',
            'API request latency',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections = Gauge(
            'db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        self.db_query_latency = Histogram(
            'db_query_latency_seconds',
            'Database query latency',
            ['query_type'],
            registry=self.registry
        )
        self.db_slow_queries = Counter(
            'db_slow_queries_total',
            'Slow database queries (>1s)',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['cache_type'],
            registry=self.registry
        )
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
    
    def record_memory_stats(self, total: int, by_status: Dict[str, int], by_scope: Dict[str, int]):
        """Record memory statistics."""
        self.memory_total.set(total)
        for status, count in by_status.items():
            self.memory_by_status.labels(status=status).set(count)
        for scope, count in by_scope.items():
            self.memory_by_scope.labels(scope=scope).set(count)
    
    def record_relationships_stats(self, total: int, avg_strength: float):
        """Record relationship statistics."""
        self.relationships_total.set(total)
        self.relationships_strength.set(avg_strength)
    
    def record_search(self, latency_seconds: float, results_count: int):
        """Record search operation."""
        self.searches_total.inc()
        self.search_latency.observe(latency_seconds)
        self.search_results.observe(results_count)
    
    def record_embedding_creation(self, latency_seconds: float):
        """Record embedding creation."""
        self.embeddings_created.inc()
        self.embedding_latency.observe(latency_seconds)
    
    def record_embedding_cache_hit(self):
        """Record embedding cache hit."""
        self.embeddings_cached.inc()
    
    def record_task_queued(self, count: int = 1):
        """Record task queued."""
        self.tasks_queued.set(count)
    
    def record_task_processing(self, count: int = 1):
        """Record task processing."""
        self.tasks_processing.set(count)
    
    def record_task_completed(self):
        """Record task completed."""
        self.tasks_completed.inc()
    
    def record_task_failed(self):
        """Record task failed."""
        self.tasks_failed.inc()
    
    def record_api_request(self, method: str, endpoint: str, status: int, latency_seconds: float):
        """Record API request."""
        self.api_requests.labels(method=method, endpoint=endpoint, status=status).inc()
        self.api_latency.labels(method=method, endpoint=endpoint).observe(latency_seconds)
    
    def record_db_connection(self, count: int):
        """Record active database connections."""
        self.db_connections.set(count)
    
    def record_db_query(self, query_type: str, latency_seconds: float):
        """Record database query."""
        self.db_query_latency.labels(query_type=query_type).observe(latency_seconds)
        
        if latency_seconds > 1.0:
            self.db_slow_queries.inc()
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_cache_size(self, cache_type: str, size_bytes: int):
        """Record cache size."""
        self.cache_size.labels(cache_type=cache_type).set(size_bytes)
    
    def get_metrics(self) -> bytes:
        """
        Get all metrics in Prometheus format.
        
        Returns:
            Prometheus text format metrics
        """
        return generate_latest(self.registry)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from current metrics.
        
        Returns:
            Dictionary of key metrics
        """
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {
                'searches_total': self.searches_total._value.get() if hasattr(self.searches_total, '_value') else 0,
                'embeddings_created': self.embeddings_created._value.get() if hasattr(self.embeddings_created, '_value') else 0,
                'tasks_completed': self.tasks_completed._value.get() if hasattr(self.tasks_completed, '_value') else 0,
                'tasks_failed': self.tasks_failed._value.get() if hasattr(self.tasks_failed, '_value') else 0,
            }
        }


# Global instance
_metrics_service: Optional[MetricsEnhancementService] = None


def get_metrics_service() -> MetricsEnhancementService:
    """Get or create global metrics service."""
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsEnhancementService()
    return _metrics_service
