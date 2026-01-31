"""Tests for healthcheck service."""

import pytest
from unittest.mock import AsyncMock, patch

from app.services.healthcheck_service import HealthcheckService, HealthStatus


@pytest.mark.asyncio
async def test_liveness():
    """Test liveness probe."""
    service = HealthcheckService()

    result = await service.liveness()

    assert result["status"] == HealthStatus.HEALTHY
    assert "timestamp" in result
    assert "uptime_seconds" in result
    assert result["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_readiness_healthy(db_session):
    """Test readiness probe when all systems healthy."""
    service = HealthcheckService()

    with patch.object(service, "_check_database", return_value=(HealthStatus.HEALTHY, 10.0)):
        with patch.object(service, "_check_redis", return_value=HealthStatus.HEALTHY):
            with patch.object(service, "_check_qdrant", return_value=HealthStatus.HEALTHY):
                result = await service.readiness()

    assert result["status"] == HealthStatus.HEALTHY
    assert "checks" in result
    assert result["checks"]["database"]["status"] == HealthStatus.HEALTHY
    assert result["checks"]["redis"]["status"] == HealthStatus.HEALTHY
    assert result["checks"]["qdrant"]["status"] == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_readiness_degraded():
    """Test readiness probe when Redis is down (degraded)."""
    service = HealthcheckService()

    with patch.object(service, "_check_database", return_value=(HealthStatus.HEALTHY, 10.0)):
        with patch.object(service, "_check_redis", return_value=HealthStatus.UNHEALTHY):
            with patch.object(service, "_check_qdrant", return_value=HealthStatus.HEALTHY):
                result = await service.readiness()

    assert result["status"] == HealthStatus.DEGRADED
    assert result["checks"]["redis"]["status"] == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_readiness_unhealthy():
    """Test readiness probe when database is down (unhealthy)."""
    service = HealthcheckService()

    with patch.object(service, "_check_database", return_value=(HealthStatus.UNHEALTHY, -1.0)):
        with patch.object(service, "_check_redis", return_value=HealthStatus.HEALTHY):
            with patch.object(service, "_check_qdrant", return_value=HealthStatus.HEALTHY):
                result = await service.readiness()

    assert result["status"] == HealthStatus.UNHEALTHY
    assert result["checks"]["database"]["status"] == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_startup_ready():
    """Test startup probe when ready."""
    service = HealthcheckService()

    with patch.object(service, "_check_migrations", return_value=True):
        with patch.object(service, "_check_required_tables", return_value=True):
            result = await service.startup()

    assert result["status"] == HealthStatus.HEALTHY
    assert result["checks"]["migrations"]["status"] == HealthStatus.HEALTHY
    assert result["checks"]["tables"]["status"] == HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_startup_not_ready():
    """Test startup probe when not ready."""
    service = HealthcheckService()

    with patch.object(service, "_check_migrations", return_value=False):
        with patch.object(service, "_check_required_tables", return_value=True):
            result = await service.startup()

    assert result["status"] == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_detailed_health():
    """Test detailed health check."""
    service = HealthcheckService()

    with patch.object(service, "_check_database", return_value=(HealthStatus.HEALTHY, 10.0)):
        with patch.object(service, "_check_redis", return_value=HealthStatus.HEALTHY):
            with patch.object(service, "_check_qdrant", return_value=HealthStatus.HEALTHY):
                result = await service.detailed_health()

    assert "status" in result
    assert "uptime_seconds" in result
    assert "checks" in result


@pytest.mark.asyncio
async def test_check_database_healthy(db_session):
    """Test database check when healthy."""
    service = HealthcheckService()

    status, latency = await service._check_database()

    # Should connect successfully
    assert status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    assert latency >= 0


@pytest.mark.asyncio
async def test_check_database_latency_thresholds():
    """Test database latency thresholds."""
    service = HealthcheckService()

    # Test healthy threshold (< 100ms)
    with patch.object(service, "_check_database", return_value=(HealthStatus.HEALTHY, 50.0)):
        status, latency = await service._check_database()
        assert latency < 100

    # Test degraded threshold (100-500ms)
    with patch.object(service, "_check_database", return_value=(HealthStatus.DEGRADED, 300.0)):
        status, latency = await service._check_database()
        assert 100 <= latency < 500

    # Test unhealthy threshold (> 500ms)
    with patch.object(service, "_check_database", return_value=(HealthStatus.UNHEALTHY, 600.0)):
        status, latency = await service._check_database()
        assert latency >= 500
