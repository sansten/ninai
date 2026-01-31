"""Tests for observability service."""

import pytest

from app.services.observability_service import ObservabilityService, AppMetrics, MetricType


@pytest.fixture
def observability():
    """Create a fresh observability service for each test."""
    return ObservabilityService()


@pytest.fixture
def app_metrics(observability):
    """Create app metrics with observability service."""
    return AppMetrics(observability)


def test_increment_counter(observability):
    """Test incrementing a counter."""
    observability.increment_counter("test_counter", value=1.0)
    observability.increment_counter("test_counter", value=2.0)

    metrics = observability.get_metrics()

    assert "test_counter" in metrics
    assert metrics["test_counter"]["type"] == MetricType.COUNTER
    assert metrics["test_counter"]["values"][""] == 3.0


def test_increment_counter_with_labels(observability):
    """Test incrementing a counter with labels."""
    observability.increment_counter("requests", labels={"method": "GET", "status": "200"})
    observability.increment_counter("requests", labels={"method": "GET", "status": "200"})
    observability.increment_counter("requests", labels={"method": "POST", "status": "201"})

    metrics = observability.get_metrics()

    assert "requests" in metrics
    assert len(metrics["requests"]["values"]) == 2


def test_set_gauge(observability):
    """Test setting a gauge value."""
    observability.set_gauge("cpu_usage", 45.5)
    observability.set_gauge("cpu_usage", 67.3)  # Overwrite

    metrics = observability.get_metrics()

    assert "cpu_usage" in metrics
    assert metrics["cpu_usage"]["type"] == MetricType.GAUGE
    assert metrics["cpu_usage"]["values"][""] == 67.3


def test_set_gauge_with_labels(observability):
    """Test setting gauge with labels."""
    observability.set_gauge("queue_depth", 10, labels={"queue": "high_priority"})
    observability.set_gauge("queue_depth", 25, labels={"queue": "low_priority"})

    metrics = observability.get_metrics()

    assert "queue_depth" in metrics
    assert len(metrics["queue_depth"]["values"]) == 2


def test_observe_histogram(observability):
    """Test observing histogram values."""
    observability.observe_histogram("latency", 10.0)
    observability.observe_histogram("latency", 20.0)
    observability.observe_histogram("latency", 15.0)

    metrics = observability.get_metrics()

    assert "latency" in metrics
    assert metrics["latency"]["type"] == MetricType.HISTOGRAM

    values = metrics["latency"]["values"][""]
    assert values["count"] == 3
    assert values["sum"] == 45.0
    assert values["min"] == 10.0
    assert values["max"] == 20.0
    assert values["avg"] == 15.0


def test_observe_histogram_with_labels(observability):
    """Test observing histogram with labels."""
    observability.observe_histogram("response_time", 100, labels={"endpoint": "/api/users"})
    observability.observe_histogram("response_time", 150, labels={"endpoint": "/api/users"})
    observability.observe_histogram("response_time", 50, labels={"endpoint": "/api/posts"})

    metrics = observability.get_metrics()

    assert "response_time" in metrics
    assert len(metrics["response_time"]["values"]) == 2


def test_prometheus_format(observability):
    """Test Prometheus text format export."""
    observability.increment_counter("http_requests", labels={"method": "GET"})
    observability.set_gauge("active_connections", 42)

    prometheus_text = observability.get_prometheus_format()

    assert "# HELP http_requests" in prometheus_text
    assert "# TYPE http_requests counter" in prometheus_text
    assert "# HELP active_connections" in prometheus_text
    assert "# TYPE active_connections gauge" in prometheus_text
    assert "active_connections 42" in prometheus_text


def test_prometheus_format_histogram(observability):
    """Test Prometheus format for histograms."""
    observability.observe_histogram("request_duration", 10.0)
    observability.observe_histogram("request_duration", 20.0)

    prometheus_text = observability.get_prometheus_format()

    assert "# HELP request_duration" in prometheus_text
    assert "# TYPE request_duration histogram" in prometheus_text
    assert "request_duration_count" in prometheus_text
    assert "request_duration_sum" in prometheus_text


def test_reset_metrics(observability):
    """Test resetting all metrics."""
    observability.increment_counter("test_counter")
    observability.set_gauge("test_gauge", 10)

    assert len(observability.get_metrics()) == 2

    observability.reset()

    assert len(observability.get_metrics()) == 0


def test_track_request(app_metrics, observability):
    """Test tracking HTTP request."""
    app_metrics.track_request("GET", "/api/users", 200, 45.5)

    metrics = observability.get_metrics()

    assert "http_requests_total" in metrics
    assert "http_request_duration_ms" in metrics


def test_track_agent_execution(app_metrics, observability):
    """Test tracking agent execution."""
    app_metrics.track_agent_execution("search_agent", True, 1500.0, 250)

    metrics = observability.get_metrics()

    assert "agent_executions_total" in metrics
    assert "agent_execution_duration_ms" in metrics
    assert "agent_tokens_consumed" in metrics


def test_track_memory_operation(app_metrics, observability):
    """Test tracking memory operation."""
    app_metrics.track_memory_operation("read", True, 12.5)

    metrics = observability.get_metrics()

    assert "memory_operations_total" in metrics
    assert "memory_operation_duration_ms" in metrics


def test_track_queue_depth(app_metrics, observability):
    """Test tracking queue depth."""
    app_metrics.track_queue_depth("q.memory_ingest", 15)

    metrics = observability.get_metrics()

    assert "queue_depth" in metrics
    assert metrics["queue_depth"]["values"]['{queue="q.memory_ingest"}'] == 15


def test_track_resource_utilization(app_metrics, observability):
    """Test tracking resource utilization."""
    app_metrics.track_resource_utilization("org123", "tokens", 0.75)

    metrics = observability.get_metrics()

    assert "resource_utilization" in metrics


def test_track_cache_hit_rate(app_metrics, observability):
    """Test tracking cache hit rate."""
    app_metrics.track_cache_hit_rate("agent_cache", 80, 20)

    metrics = observability.get_metrics()

    assert "cache_hits_total" in metrics
    assert "cache_misses_total" in metrics
    assert "cache_hit_rate" in metrics

    # Hit rate should be 80/100 = 0.8
    hit_rate = metrics["cache_hit_rate"]["values"]['{cache="agent_cache"}']
    assert hit_rate == 0.8


def test_multiple_request_tracking(app_metrics, observability):
    """Test tracking multiple requests with different status codes."""
    app_metrics.track_request("GET", "/api/users", 200, 45.0)
    app_metrics.track_request("GET", "/api/users", 200, 55.0)
    app_metrics.track_request("GET", "/api/users", 404, 10.0)
    app_metrics.track_request("POST", "/api/users", 201, 120.0)

    metrics = observability.get_metrics()

    # Check that different combinations are tracked separately
    request_values = metrics["http_requests_total"]["values"]
    assert len(request_values) >= 3  # At least 3 different combinations


def test_histogram_statistics_accuracy(observability):
    """Test histogram statistics are calculated correctly."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0]

    for val in values:
        observability.observe_histogram("test_metric", val)

    metrics = observability.get_metrics()
    stats = metrics["test_metric"]["values"][""]

    assert stats["count"] == 5
    assert stats["sum"] == 150.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["avg"] == 30.0


def test_label_formatting(observability):
    """Test label formatting for Prometheus."""
    labels = {"method": "GET", "status": "200", "endpoint": "/api/users"}

    formatted = observability._format_labels(labels)

    assert "method=\"GET\"" in formatted
    assert "status=\"200\"" in formatted
    assert "endpoint=\"/api/users\"" in formatted
    assert formatted.startswith("{")
    assert formatted.endswith("}")


def test_empty_labels(observability):
    """Test handling of empty labels."""
    formatted = observability._format_labels(None)
    assert formatted == ""

    formatted = observability._format_labels({})
    assert formatted == ""
