"""Observability service for metrics collection and monitoring.

Provides:
- Prometheus-style metrics
- System resource monitoring
- Application-level metrics
- Custom metric tracking
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Any


class MetricType:
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class ObservabilityService:
    """Service for collecting and exposing metrics."""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = defaultdict(list)

    def increment_counter(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels/tags
        """
        key = self._make_key(name, labels)
        self.counters[key] += value

        if name not in self.metrics:
            self.metrics[name] = {
                "type": MetricType.COUNTER,
                "help": f"Counter for {name}",
                "values": {},
            }

        label_str = self._format_labels(labels)
        self.metrics[name]["values"][label_str] = self.counters[key]

    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        """Set a gauge metric to a specific value.

        Args:
            name: Metric name
            value: Value to set
            labels: Optional labels/tags
        """
        key = self._make_key(name, labels)
        self.gauges[key] = value

        if name not in self.metrics:
            self.metrics[name] = {
                "type": MetricType.GAUGE,
                "help": f"Gauge for {name}",
                "values": {},
            }

        label_str = self._format_labels(labels)
        self.metrics[name]["values"][label_str] = value

    def observe_histogram(self, name: str, value: float, labels: dict | None = None) -> None:
        """Record an observation in a histogram.

        Args:
            name: Metric name
            value: Value to observe
            labels: Optional labels/tags
        """
        key = self._make_key(name, labels)
        self.histograms[key].append(value)

        if name not in self.metrics:
            self.metrics[name] = {
                "type": MetricType.HISTOGRAM,
                "help": f"Histogram for {name}",
                "values": {},
            }

        label_str = self._format_labels(labels)

        # Calculate histogram statistics
        values = self.histograms[key]
        if values:
            self.metrics[name]["values"][label_str] = {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

    def get_metrics(self) -> dict:
        """Get all metrics.

        Returns:
            Dict of all metrics
        """
        return self.metrics.copy()

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Metrics as Prometheus text format
        """
        lines = []

        for name, metric in self.metrics.items():
            # Add HELP line
            lines.append(f"# HELP {name} {metric['help']}")

            # Add TYPE line
            lines.append(f"# TYPE {name} {metric['type']}")

            # Add values
            for labels, value in metric["values"].items():
                if metric["type"] in (MetricType.COUNTER, MetricType.GAUGE):
                    lines.append(f"{name}{labels} {value}")
                elif metric["type"] == MetricType.HISTOGRAM:
                    # Export histogram as summary
                    lines.append(f"{name}_count{labels} {value['count']}")
                    lines.append(f"{name}_sum{labels} {value['sum']}")

            lines.append("")  # Blank line between metrics

        return "\n".join(lines)

    def _make_key(self, name: str, labels: dict | None) -> str:
        """Create a unique key for a metric with labels.

        Args:
            name: Metric name
            labels: Labels dict

        Returns:
            Unique key string
        """
        if not labels:
            return name

        label_parts = [f"{k}={v}" for k, v in sorted(labels.items())]
        return f"{name}:{':'.join(label_parts)}"

    def _format_labels(self, labels: dict | None) -> str:
        """Format labels for Prometheus output.

        Args:
            labels: Labels dict

        Returns:
            Formatted label string
        """
        if not labels:
            return ""

        label_parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ", ".join(label_parts) + "}"

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()


# Standard application metrics
class AppMetrics:
    """Standard application metrics."""

    def __init__(self, observability: ObservabilityService):
        self.obs = observability

    def track_request(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        """Track HTTP request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
        """
        labels = {
            "method": method,
            "path": path,
            "status": str(status_code),
        }

        self.obs.increment_counter("http_requests_total", labels=labels)
        self.obs.observe_histogram("http_request_duration_ms", duration_ms, labels=labels)

    def track_agent_execution(self, agent_type: str, success: bool, duration_ms: float, tokens: int) -> None:
        """Track agent execution.

        Args:
            agent_type: Type of agent
            success: Whether execution succeeded
            duration_ms: Execution duration
            tokens: Tokens consumed
        """
        labels = {
            "agent_type": agent_type,
            "success": str(success),
        }

        self.obs.increment_counter("agent_executions_total", labels=labels)
        self.obs.observe_histogram("agent_execution_duration_ms", duration_ms, labels=labels)
        self.obs.observe_histogram("agent_tokens_consumed", tokens, labels=labels)

    def track_memory_operation(self, operation: str, success: bool, duration_ms: float) -> None:
        """Track memory syscall operation.

        Args:
            operation: Operation type (read, write, search, etc.)
            success: Whether operation succeeded
            duration_ms: Operation duration
        """
        labels = {
            "operation": operation,
            "success": str(success),
        }

        self.obs.increment_counter("memory_operations_total", labels=labels)
        self.obs.observe_histogram("memory_operation_duration_ms", duration_ms, labels=labels)

    def track_queue_depth(self, queue_name: str, depth: int) -> None:
        """Track queue depth.

        Args:
            queue_name: Name of the queue
            depth: Number of items in queue
        """
        self.obs.set_gauge("queue_depth", depth, labels={"queue": queue_name})

    def track_resource_utilization(self, organization_id: str, resource: str, utilization: float) -> None:
        """Track resource utilization percentage.

        Args:
            organization_id: Organization ID
            resource: Resource type (tokens, storage, etc.)
            utilization: Utilization percentage (0.0-1.0)
        """
        labels = {
            "organization_id": organization_id,
            "resource": resource,
        }

        self.obs.set_gauge("resource_utilization", utilization, labels=labels)

    def track_cache_hit_rate(self, cache_name: str, hits: int, misses: int) -> None:
        """Track cache hit rate.

        Args:
            cache_name: Cache name
            hits: Number of hits
            misses: Number of misses
        """
        labels = {"cache": cache_name}

        self.obs.increment_counter("cache_hits_total", hits, labels=labels)
        self.obs.increment_counter("cache_misses_total", misses, labels=labels)

        total = hits + misses
        if total > 0:
            hit_rate = hits / total
            self.obs.set_gauge("cache_hit_rate", hit_rate, labels=labels)


# Global observability instances
observability_service = ObservabilityService()
app_metrics = AppMetrics(observability_service)
