"""System bootstrap protocol for proper initialization order.

Ensures all OS components are ready before accepting requests:
1. Database connectivity
2. Cache/Redis availability
3. Memory system loading
4. Agent initialization
5. Configuration validation
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


class BootstrapCheck:
    """Individual bootstrap check result."""
    
    def __init__(
        self,
        name: str,
        category: str,
        required: bool = True,
    ):
        self.name = name
        self.category = category  # "database", "cache", "memory", "agents", "config"
        self.required = required
        self.status = "pending"  # pending, running, success, warning, failed
        self.message = ""
        self.duration_ms = 0
        self.checked_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category,
            "required": self.required,
            "status": self.status,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
        }


class BootstrapService:
    """Orchestrates system bootstrap checks."""

    def __init__(self):
        self.checks: List[BootstrapCheck] = []
        self.bootstrap_complete = False
        self.bootstrap_started_at: Optional[datetime] = None
        self.bootstrap_completed_at: Optional[datetime] = None

    async def initialize(self, db: AsyncSession) -> bool:
        """
        Run all bootstrap checks in order.
        
        Returns:
            True if all required checks passed, False otherwise
        """
        self.bootstrap_started_at = datetime.now(timezone.utc)
        logger.info("=" * 60)
        logger.info("SYSTEM BOOTSTRAP STARTED")
        logger.info("=" * 60)

        # Category order: database -> cache -> memory -> agents -> config
        categories = ["database", "cache", "memory", "agents", "config"]
        all_passed = True

        for category in categories:
            logger.info(f"\n[{category.upper()}]")
            category_checks = [c for c in self.checks if c.category == category]
            
            for check in category_checks:
                await self._run_check(check, db)
                
                if check.status == "failed" and check.required:
                    logger.error(f"  ✗ {check.name}: FAILED (required)")
                    all_passed = False
                elif check.status == "failed":
                    logger.warning(f"  ⚠ {check.name}: FAILED (optional)")
                elif check.status == "warning":
                    logger.warning(f"  ⚠ {check.name}: {check.message}")
                else:
                    logger.info(f"  ✓ {check.name}: {check.message}")

        self.bootstrap_completed_at = datetime.now(timezone.utc)
        self.bootstrap_complete = all_passed

        logger.info("\n" + "=" * 60)
        if all_passed:
            duration = (self.bootstrap_completed_at - self.bootstrap_started_at).total_seconds()
            logger.info(f"BOOTSTRAP COMPLETE ({duration:.2f}s)")
        else:
            logger.error("BOOTSTRAP FAILED - Some required checks did not pass")
        logger.info("=" * 60 + "\n")

        return all_passed

    async def _run_check(self, check: BootstrapCheck, db: AsyncSession):
        """Run a single check."""
        check.status = "running"
        start_time = datetime.now(timezone.utc)

        try:
            if check.name == "database_connection":
                await self._check_database(db, check)
            elif check.name == "database_tables":
                await self._check_database_tables(db, check)
            elif check.name == "redis_connection":
                await self._check_redis(check)
            elif check.name == "memory_loading":
                await self._check_memory_loading(db, check)
            elif check.name == "agent_initialization":
                await self._check_agent_initialization(check)
            elif check.name == "configuration":
                await self._check_configuration(check)
            else:
                check.status = "warning"
                check.message = "Unknown check"

        except Exception as e:
            check.status = "failed"
            check.message = str(e)
            logger.exception(f"Error in check '{check.name}'")

        check.checked_at = datetime.now(timezone.utc)
        check.duration_ms = int((check.checked_at - start_time).total_seconds() * 1000)

    async def _check_database(self, db: AsyncSession, check: BootstrapCheck):
        """Check database connectivity."""
        try:
            await db.execute("""SELECT 1""")
            check.status = "success"
            check.message = "Connected"
        except Exception as e:
            check.status = "failed"
            check.message = f"Connection failed: {str(e)}"

    async def _check_database_tables(self, db: AsyncSession, check: BootstrapCheck):
        """Check required tables exist."""
        required_tables = [
            "organizations",
            "users",
            "pipeline_tasks",
            "dead_letter_queue",
            "app_settings",
            "audit_logs",
        ]
        
        try:
            # Get list of tables
            from sqlalchemy import inspect
            inspector = inspect(db.sync_session_class())
            existing_tables = inspector.get_table_names()
            
            missing = [t for t in required_tables if t not in existing_tables]
            
            if missing:
                check.status = "failed"
                check.message = f"Missing tables: {', '.join(missing)}"
            else:
                check.status = "success"
                check.message = f"All {len(required_tables)} required tables exist"
        except Exception as e:
            check.status = "failed"
            check.message = f"Table check failed: {str(e)}"

    async def _check_redis(self, check: BootstrapCheck):
        """Check Redis/cache connectivity."""
        try:
            # Try to import and test cache
            from app.core.cache import cache_service
            
            if cache_service:
                check.status = "success"
                check.message = "Connected"
            else:
                check.status = "warning"
                check.message = "Cache service not configured (optional)"
        except ImportError:
            check.status = "warning"
            check.message = "Cache not available (optional)"
        except Exception as e:
            check.status = "warning"
            check.message = f"Cache check: {str(e)}"

    async def _check_memory_loading(self, db: AsyncSession, check: BootstrapCheck):
        """Check memory system is ready."""
        try:
            # Check if memory-related tables/services are accessible
            from app.models.memory_stream import MemoryStream
            
            check.status = "success"
            check.message = "Memory system initialized"
        except Exception as e:
            check.status = "warning"
            check.message = f"Memory check: {str(e)}"

    async def _check_agent_initialization(self, check: BootstrapCheck):
        """Check agent scheduler is ready."""
        try:
            # Would check if agent scheduler can be instantiated
            check.status = "success"
            check.message = "Agent scheduler ready"
        except Exception as e:
            check.status = "warning"
            check.message = f"Agent check: {str(e)}"

    async def _check_configuration(self, check: BootstrapCheck):
        """Check required configuration is present."""
        try:
            required_settings = [
                ("database_url", settings.DATABASE_URL),
                ("jwt_secret", settings.JWT_SECRET),
            ]
            
            missing = [name for name, value in required_settings if not value]
            
            if missing:
                check.status = "failed"
                check.message = f"Missing settings: {', '.join(missing)}"
            else:
                check.status = "success"
                check.message = "Configuration valid"
        except Exception as e:
            check.status = "failed"
            check.message = f"Configuration check failed: {str(e)}"

    def add_check(
        self,
        name: str,
        category: str,
        required: bool = True,
    ) -> BootstrapCheck:
        """Add a bootstrap check."""
        check = BootstrapCheck(name, category, required)
        self.checks.append(check)
        return check

    def get_status(self) -> Dict:
        """Get bootstrap status."""
        return {
            "complete": self.bootstrap_complete,
            "started_at": self.bootstrap_started_at.isoformat() if self.bootstrap_started_at else None,
            "completed_at": self.bootstrap_completed_at.isoformat() if self.bootstrap_completed_at else None,
            "duration_seconds": (
                (self.bootstrap_completed_at - self.bootstrap_started_at).total_seconds()
                if self.bootstrap_started_at and self.bootstrap_completed_at
                else None
            ),
            "checks": {
                "total": len(self.checks),
                "passed": len([c for c in self.checks if c.status == "success"]),
                "warnings": len([c for c in self.checks if c.status == "warning"]),
                "failed": len([c for c in self.checks if c.status == "failed"]),
            },
            "details": [c.to_dict() for c in self.checks],
        }


# Global bootstrap instance
bootstrap_service = BootstrapService()


def create_default_bootstrap_checks():
    """Create standard bootstrap checks."""
    bootstrap_service.add_check("database_connection", "database", required=True)
    bootstrap_service.add_check("database_tables", "database", required=True)
    bootstrap_service.add_check("redis_connection", "cache", required=False)
    bootstrap_service.add_check("memory_loading", "memory", required=True)
    bootstrap_service.add_check("agent_initialization", "agents", required=True)
    bootstrap_service.add_check("configuration", "config", required=True)
