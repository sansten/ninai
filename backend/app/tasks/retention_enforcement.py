"""Retention Policy Enforcement Task — Gap D.

Nightly Celery task that enforces data retention policies set by org admins.

For each org with an active DataRetentionPolicy:
  1. Delete memory_metadata rows older than retention_days_memory.
  2. Archive (soft-delete via classification="archived") rows older than
     auto_archive_after_days but within the retention window.
  3. Return a summary of how many memories were deleted and archived.

Runs at 01:30 UTC daily (before the sleep cycle at 02:00) so the sleep
cycle never operates on memories that should have been purged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetentionEnforcementResult:
    org_id: str
    deleted_count: int = 0
    archived_count: int = 0
    errors: list[str] = field(default_factory=list)
    ran_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


async def enforce_retention_for_org(
    db: Any,
    *,
    org_id: str,
    dry_run: bool = False,
) -> RetentionEnforcementResult:
    """Enforce retention policy for a single org.

    Parameters
    ----------
    db:       Async SQLAlchemy session (already within a transaction or will
              manage its own via begin_nested).
    org_id:   Organisation to process.
    dry_run:  If True, compute counts but do not commit any deletes/updates.
    """
    from sqlalchemy import select, and_, update, delete as sa_delete, func
    from app.models.compliance import DataRetentionPolicy
    from app.models.memory import MemoryMetadata

    result = RetentionEnforcementResult(org_id=org_id)
    now = datetime.now(timezone.utc)

    # --- Load policy ---
    stmt = select(DataRetentionPolicy).where(
        and_(
            DataRetentionPolicy.organization_id == org_id,
            DataRetentionPolicy.is_active.is_(True),
        )
    )
    res = await db.execute(stmt)
    policy = res.scalar_one_or_none()
    if policy is None:
        return result  # no active policy, nothing to do

    retention_cutoff = now - timedelta(days=int(policy.retention_days_memory))
    archive_cutoff = now - timedelta(days=int(policy.auto_archive_after_days))

    # --- Count / delete expired memories ---
    try:
        expired_stmt = select(func.count()).select_from(MemoryMetadata).where(
            and_(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.created_at < retention_cutoff,
            )
        )
        expired_count = (await db.execute(expired_stmt)).scalar_one()
        result.deleted_count = int(expired_count)

        if not dry_run and expired_count > 0:
            del_stmt = sa_delete(MemoryMetadata).where(
                and_(
                    MemoryMetadata.organization_id == org_id,
                    MemoryMetadata.created_at < retention_cutoff,
                )
            )
            await db.execute(del_stmt)
    except Exception as exc:
        result.errors.append(f"delete_expired: {type(exc).__name__}: {exc}")

    # --- Count / archive stale memories (between archive_cutoff and retention_cutoff) ---
    try:
        stale_stmt = select(func.count()).select_from(MemoryMetadata).where(
            and_(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.created_at < archive_cutoff,
                MemoryMetadata.created_at >= retention_cutoff,
                MemoryMetadata.classification != "archived",
            )
        )
        stale_count = (await db.execute(stale_stmt)).scalar_one()
        result.archived_count = int(stale_count)

        if not dry_run and stale_count > 0:
            archive_stmt = (
                update(MemoryMetadata)
                .where(
                    and_(
                        MemoryMetadata.organization_id == org_id,
                        MemoryMetadata.created_at < archive_cutoff,
                        MemoryMetadata.created_at >= retention_cutoff,
                        MemoryMetadata.classification != "archived",
                    )
                )
                .values(classification="archived")
            )
            await db.execute(archive_stmt)
    except Exception as exc:
        result.errors.append(f"archive_stale: {type(exc).__name__}: {exc}")

    return result


try:
    from app.core.celery_app import celery_app
    from app.core.database import AsyncSessionLocal, set_tenant_context
    from sqlalchemy import select
    from app.models.compliance import DataRetentionPolicy

    @celery_app.task(
        bind=True,
        name="app.tasks.retention_enforcement.retention_enforcement_task",
        max_retries=2,
        default_retry_delay=300,
    )
    def retention_enforcement_task(self: Any) -> dict[str, Any]:  # type: ignore[misc]
        """Nightly task: enforce retention policies for all active orgs."""
        import asyncio

        async def _run() -> dict[str, Any]:
            summary: dict[str, Any] = {
                "orgs_processed": 0,
                "total_deleted": 0,
                "total_archived": 0,
                "errors": [],
            }
            async with AsyncSessionLocal() as db:
                # Collect all orgs with an active policy
                stmt = select(DataRetentionPolicy.organization_id).where(
                    DataRetentionPolicy.is_active.is_(True)
                )
                res = await db.execute(stmt)
                org_ids = [row[0] for row in res.all()]

            for org_id in org_ids:
                try:
                    async with AsyncSessionLocal() as db:
                        async with db.begin():
                            await set_tenant_context(
                                db, None, org_id,
                                roles="system",
                                clearance_level=0,
                                justification="retention_enforcement",
                            )
                            result = await enforce_retention_for_org(db, org_id=org_id)

                        summary["orgs_processed"] += 1
                        summary["total_deleted"] += result.deleted_count
                        summary["total_archived"] += result.archived_count
                        if result.errors:
                            summary["errors"].extend(
                                [f"{org_id}: {e}" for e in result.errors]
                            )
                except Exception as exc:
                    summary["errors"].append(f"{org_id}: {type(exc).__name__}: {exc}")
                    logger.exception("retention_enforcement_task failed for org %s", org_id)

            logger.info(
                "retention_enforcement_task complete: %d orgs, %d deleted, %d archived",
                summary["orgs_processed"],
                summary["total_deleted"],
                summary["total_archived"],
            )
            return summary

        return asyncio.get_event_loop().run_until_complete(_run())

except Exception:
    pass  # Celery not available in test environment

# Beat schedule entry — imported by celery_app._load_task_module
CELERY_BEAT_SCHEDULE: dict[str, Any] = {
    "nightly-retention-enforcement": {
        "task": "app.tasks.retention_enforcement.retention_enforcement_task",
        "schedule": __import__("celery.schedules", fromlist=["crontab"]).crontab(
            minute=30, hour=1
        ),
        "args": (),
    },
}
