"""
Example 7: Observability & Monitoring

Demonstrates production observability setup:
- Prometheus metrics
- Structured logging (Loki)
- Distributed tracing (Jaeger)
- Custom metrics
- Health checks
"""

import asyncio
import aiohttp
from datetime import datetime
import time
from typing import Optional


class ObservabilityManager:
    """Configure and manage observability in Ninai2."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}
    
    async def health_check(self) -> dict:
        """Get service health status."""
        async with self.session.get(
            f"{self.base_url}/health",
            headers=self._headers()
        ) as resp:
            return await resp.json()
    
    async def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        async with self.session.get(f"{self.base_url}/metrics") as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Metrics retrieval failed: {resp.status}")
            return await resp.text()
    
    async def get_database_metrics(self) -> dict:
        """Get database performance metrics."""
        async with self.session.get(
            f"{self.base_url}/api/v1/admin/metrics/database",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Database metrics failed: {resp.status}")
            return await resp.json()
    
    async def get_cache_metrics(self) -> dict:
        """Get Redis cache metrics."""
        async with self.session.get(
            f"{self.base_url}/api/v1/admin/metrics/cache",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Cache metrics failed: {resp.status}")
            return await resp.json()
    
    async def record_event(self, event_type: str, labels: dict, value: float = 1.0) -> None:
        """Record a custom metric event."""
        payload = {
            "event_type": event_type,
            "labels": labels,
            "value": value,
            "timestamp": datetime.utcnow().isoformat()
        }
        async with self.session.post(
            f"{self.base_url}/api/v1/metrics/events",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Event recording failed: {resp.status}")


async def main():
    """Run observability and monitoring example."""
    
    base_url = "http://localhost:8000"
    api_key = "test-api-key"
    
    print("Ninai2 Observability & Monitoring Example")
    print("=" * 50)
    print()
    
    try:
        async with ObservabilityManager(base_url, api_key) as obs:
            
            # 1. Health checks
            print("1. System Health Checks...")
            health = await obs.health_check()
            
            # Parse health response
            status = health.get("status", "unknown")
            icon = "✓" if status == "healthy" else "⚠"
            print(f"{icon} Overall Status: {status}")
            
            services = health.get("services", {})
            for service, service_status in services.items():
                service_icon = "✓" if service_status.get("healthy") else "✗"
                print(f"  {service_icon} {service}: {service_status.get('status', 'unknown')}")
            print()
            
            # 2. Database metrics
            print("2. Database Performance Metrics...")
            try:
                db_metrics = await obs.get_database_metrics()
                
                print(f"✓ Database Health:")
                print(f"  Pool size: {db_metrics.get('pool_size', 'N/A')}")
                print(f"  Active connections: {db_metrics.get('active_connections', 'N/A')}")
                print(f"  Slow queries (>1s): {db_metrics.get('slow_query_count', 0)}")
                print(f"  Average query time: {db_metrics.get('avg_query_time_ms', 'N/A')}ms")
                print()
            except RuntimeError:
                print("  ⚠ Database metrics unavailable (configure /metrics/database endpoint)")
                print()
            
            # 3. Cache metrics
            print("3. Cache Performance Metrics...")
            try:
                cache_metrics = await obs.get_cache_metrics()
                
                hit_rate = cache_metrics.get('hit_rate', 0)
                print(f"✓ Redis Cache:")
                print(f"  Hit rate: {hit_rate * 100:.1f}%")
                print(f"  Keys in cache: {cache_metrics.get('keys_count', 'N/A')}")
                print(f"  Memory usage: {cache_metrics.get('memory_usage_bytes', 'N/A')} bytes")
                print(f"  Evictions: {cache_metrics.get('eviction_count', 0)}")
                print()
            except RuntimeError:
                print("  ⚠ Cache metrics unavailable (configure /metrics/cache endpoint)")
                print()
            
            # 4. Prometheus metrics format (sample)
            print("4. Prometheus Metrics Sample...")
            try:
                metrics_text = await obs.get_metrics()
                lines = metrics_text.split('\n')
                
                # Show first few metrics
                metric_count = 0
                for line in lines:
                    if not line.startswith('#') and line.strip():
                        metric_count += 1
                        if metric_count <= 5:
                            print(f"  {line}")
                
                print(f"  ... and {len(lines) - metric_count - 10} more metrics")
                print(f"\n  Total metric lines: {len(lines)}")
            except RuntimeError:
                print("  ⚠ Prometheus endpoint not available")
            print()
            
            # 5. Custom event recording
            print("5. Recording Custom Metrics...")
            try:
                # Record different event types
                await obs.record_event(
                    "agent_invocation",
                    {"agent_id": "research_bot", "status": "success"},
                    value=1.0
                )
                print(f"✓ Recorded: agent_invocation (success)")
                
                await obs.record_event(
                    "memory_search_latency",
                    {"org_id": "org_123"},
                    value=145.3  # milliseconds
                )
                print(f"✓ Recorded: memory_search_latency (145.3ms)")
                
                await obs.record_event(
                    "api_rate_limit_hit",
                    {"endpoint": "/memories/search", "client_id": "mobile_app"},
                    value=1.0
                )
                print(f"✓ Recorded: api_rate_limit_hit")
            except RuntimeError as e:
                print(f"⚠ Custom metrics unavailable: {e}")
            print()
            
            # 6. Observability setup guide
            print("6. Production Observability Stack Setup:")
            setup_steps = [
                ("Prometheus", "Scrape /metrics endpoint every 15s", "monitoring/prometheus.yml"),
                ("Loki", "Consume docker logs via Promtail", "monitoring/loki-config.yml"),
                ("Jaeger", "Enable distributed tracing exports", "monitoring/jaeger-config.yml"),
                ("Grafana", "Dashboard templates in monitoring/grafana/dashboards", "grafana_dashboards/"),
                ("AlertManager", "Configure alert rules in monitoring/rules.yml", "monitoring/alertmanager.yml"),
            ]
            
            for tool, description, config_path in setup_steps:
                print(f"  ✓ {tool}")
                print(f"    → {description}")
                print(f"    → Config: {config_path}")
            print()
            
            # 7. Dashboard recommendations
            print("7. Recommended Dashboards:")
            dashboards = [
                ("System Health", ["CPU", "Memory", "Disk I/O", "Network"]),
                ("API Performance", ["Request rate", "Latency P50/P95/P99", "Error rate"]),
                ("Database", ["Connections", "Query time", "Slow queries", "Lock contention"]),
                ("Cache (Redis)", ["Hit rate", "Evictions", "Memory usage"]),
                ("LLM Routing", ["Provider usage", "Fallback rate", "Token cost"]),
                ("Business Metrics", ["Users active", "Memories created", "Agents invoked"])
            ]
            
            for dashboard_name, metrics in dashboards:
                metric_str = ", ".join(metrics)
                print(f"  • {dashboard_name}: {metric_str}")
            print()
            
            print("✓ Observability & monitoring example completed!")
            print("\nKey Reminders:")
            print("  - Set up Prometheus scraping (15-30s intervals)")
            print("  - Configure alerting for critical metrics (>95% CPU, >10% error rate)")
            print("  - Retain metrics for at least 30 days")
            print("  - Create runbooks for common alerts")
            print("  - Review dashboards weekly for anomalies")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check health endpoint: curl http://localhost:8000/health")
        print("  2. Verify Prometheus endpoints enabled: /metrics, /metrics/db, /metrics/cache")
        print("  3. Check Loki/Jaeger dependencies are running")
        print("  4. Ensure org_admin role for metrics access")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
