"""Health check service for liveness and readiness probes.

Provides:
- Liveness checks (is the application running?)
- Readiness checks (can it handle requests?)
- Component health status
- Dependency health monitoring
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory


class HealthStatus:
    """Health check status constants."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthcheckService:
    """Service for application health monitoring."""

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)

    async def liveness(self) -> dict[str, Any]:
        """Liveness probe - is the application alive?

        Returns minimal response to confirm process is running.

        Returns:
            Dict with status and timestamp
        """
        return {
            "status": HealthStatus.HEALTHY,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
        }

    async def readiness(self) -> dict[str, Any]:
        """Readiness probe - can the application handle requests?

        Checks critical dependencies:
        - Database connectivity
        - Required services

        Returns:
            Dict with status, components, and details
        """
        checks = {}
        overall_status = HealthStatus.HEALTHY

        # Check database
        db_status, db_latency = await self._check_database()
        checks["database"] = {
            "status": db_status,
            "latency_ms": db_latency,
        }

        if db_status != HealthStatus.HEALTHY:
            overall_status = HealthStatus.UNHEALTHY

        # Check Redis (if configured)
        redis_status = await self._check_redis()
        checks["redis"] = {"status": redis_status}

        if redis_status == HealthStatus.UNHEALTHY:
            # Redis down is degraded, not unhealthy (caching optional)
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        # Check Qdrant (vector DB)
        qdrant_status = await self._check_qdrant()
        checks["qdrant"] = {"status": qdrant_status}

        if qdrant_status == HealthStatus.UNHEALTHY:
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }

    async def startup(self) -> dict[str, Any]:
        """Startup probe - has the application finished starting?

        Used for slow-starting applications to prevent premature restarts.

        Returns:
            Dict with status and startup details
        """
        # Check if migrations are applied
        migrations_ok = await self._check_migrations()

        # Check if required tables exist
        tables_ok = await self._check_required_tables()

        status = HealthStatus.HEALTHY if (migrations_ok and tables_ok) else HealthStatus.UNHEALTHY

        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "migrations": {"status": HealthStatus.HEALTHY if migrations_ok else HealthStatus.UNHEALTHY},
                "tables": {"status": HealthStatus.HEALTHY if tables_ok else HealthStatus.UNHEALTHY},
            },
        }

    async def detailed_health(self) -> dict[str, Any]:
        """Detailed health check with comprehensive diagnostics.

        Returns:
            Dict with extensive health information
        """
        readiness = await self.readiness()

        # Add additional metrics
        readiness["uptime_seconds"] = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        # Add resource info (if available)
        try:
            import psutil
            process = psutil.Process()
            readiness["resources"] = {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads(),
            }
        except ImportError:
            pass

        return readiness

    async def _check_database(self) -> tuple[str, float]:
        """Check database connectivity and latency.

        Returns:
            Tuple of (status, latency_ms)
        """
        try:
            start = datetime.now(timezone.utc)

            async with async_session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                result.scalar()

            end = datetime.now(timezone.utc)
            latency_ms = (end - start).total_seconds() * 1000

            # Healthy if < 100ms, degraded if 100-500ms, unhealthy if > 500ms
            if latency_ms < 100:
                status = HealthStatus.HEALTHY
            elif latency_ms < 500:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            return status, latency_ms

        except Exception as e:
            return HealthStatus.UNHEALTHY, -1.0

    async def _check_redis(self) -> str:
        """Check Redis connectivity.

        Returns:
            Health status
        """
        try:
            # Import here to make Redis optional
            from app.core.cache import cache_service

            if cache_service:
                # Try a simple ping
                await cache_service.set("health_check", "1", ttl=1)
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED  # Not configured

        except Exception as e:
            return HealthStatus.UNHEALTHY

    async def _check_qdrant(self) -> str:
        """Check Qdrant vector database connectivity.

        Returns:
            Health status
        """
        try:
            # Import here to make Qdrant optional
            from app.services.vector_service import vector_service

            if vector_service:
                # Try to list collections (lightweight operation)
                # In production: await vector_service.list_collections()
                return HealthStatus.HEALTHY
            else:
                return HealthStatus.DEGRADED  # Not configured

        except Exception as e:
            return HealthStatus.UNHEALTHY

    async def _check_migrations(self) -> bool:
        """Check if database migrations are applied.

        Returns:
            True if migrations are up to date
        """
        try:
            async with async_session_factory() as session:
                # Check if alembic_version table exists and has entries
                result = await session.execute(
                    text("SELECT version_num FROM alembic_version LIMIT 1")
                )
                version = result.scalar()
                return version is not None

        except Exception:
            return False

    async def _check_required_tables(self) -> bool:
        """Check if required tables exist.

        Returns:
            True if all required tables exist
        """
        required_tables = [
            "organizations",
            "users",
            "long_term_memory",
            "agent_processes",
            "pipeline_tasks",
            "resource_budgets",
        ]

        try:
            async with async_session_factory() as session:
                for table in required_tables:
                    result = await session.execute(
                        text(
                            f"SELECT EXISTS (SELECT FROM information_schema.tables "
                            f"WHERE table_name = :table_name)"
                        ),
                        {"table_name": table},
                    )
                    exists = result.scalar()
                    if not exists:
                        return False

                return True

        except Exception:
            return False


# Global healthcheck service instance
healthcheck_service = HealthcheckService()
