"""Environment sync reconciliation tasks - Phase 52 Slice 2."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import traceback
from uuid import uuid4

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.middleware.prometheus import (
    record_environment_reconcile_run,
    update_environment_sync_connector_metrics,
)
from app.models.pipeline_task import PipelineTask, PipelineTaskStatus, PipelineTaskType
from app.services.dead_letter_queue_service import DeadLetterQueueService
from app.services.environment_reconciliation_service import EnvironmentReconciliationService
from app.services.environment_sync_service import (
    CanonicalConnectorState,
    get_environment_sync_service,
)
from app.services.external_connector_service import ExternalConnectorService


logger = get_task_logger(__name__)


async def _handoff_reconcile_failure_to_dlq(
    *,
    org_id: str,
    state: CanonicalConnectorState,
    error: str,
) -> None:
    """Persist a reconciliation failure into DLQ via DeadLetterQueueService."""
    async with async_session_factory() as session:
        async with session.begin():
            now = datetime.now(timezone.utc)
            pipeline_task = PipelineTask(
                organization_id=org_id,
                task_type=PipelineTaskType.FEEDBACK_LOOP.value,
                status=PipelineTaskStatus.FAILED.value,
                priority=10,
                sla_deadline=now,
                attempts=1,
                max_attempts=1,
                started_at=now,
                finished_at=now,
                input_session_id="env_sync_reconcile",
                target_resource_id=f"{state.connector_type}:{state.external_object_id}",
                task_metadata={
                    "category": "environment_sync_reconciliation",
                    "connector_type": state.connector_type,
                    "external_object_id": state.external_object_id,
                    "source_event_id": state.source_event_id,
                    "state_hash": state.state_hash,
                },
                last_error=error,
            )
            session.add(pipeline_task)
            await session.flush()

            dlq = DeadLetterQueueService(session)
            await dlq.check_and_quarantine(
                task=pipeline_task,
                reason="sync_reconcile_failed",
            )


@celery_app.task(name="app.tasks.environment_sync.reconcile_environment_sync_org_task")
def reconcile_environment_sync_org_task(
    *,
    org_id: str,
    max_lag_seconds: int = 30,
    limit: int = 100,
) -> dict:
    """Run outbound reconciliation for one org."""

    async def _run() -> dict:
        sync_service = get_environment_sync_service()
        worker = EnvironmentReconciliationService(
            sync_service=sync_service,
            connector=ExternalConnectorService(),
            failure_handler=lambda state, error: _handoff_reconcile_failure_to_dlq(
                org_id=org_id,
                state=state,
                error=error,
            ),
        )

        result = await worker.reconcile_org(
            org_id=org_id,
            max_lag_seconds=max_lag_seconds,
            limit=limit,
        )

        connector_summaries = sync_service.connector_summaries(org_id)
        update_environment_sync_connector_metrics(org_id, connector_summaries)

        run_status = "success" if result.failed == 0 else "partial_failure"
        record_environment_reconcile_run(
            run_status=run_status,
            succeeded=result.succeeded,
            failed=result.failed,
            dlq_handoffs=result.dlq_handoffs,
        )

        logger.info(
            "environment sync reconcile run org=%s attempted=%s succeeded=%s failed=%s dlq=%s",
            org_id,
            result.attempted,
            result.succeeded,
            result.failed,
            result.dlq_handoffs,
        )
        return result.to_dict()

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run())
    except Exception as exc:
        logger.exception("environment sync reconcile org task failed: %s", exc)
        record_environment_reconcile_run(
            run_status="failure",
            succeeded=0,
            failed=0,
            dlq_handoffs=0,
        )
        return {
            "org_id": org_id,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


@celery_app.task(name="app.tasks.environment_sync.reconcile_environment_sync_all_task")
def reconcile_environment_sync_all_task(
    *,
    max_lag_seconds: int = 30,
    per_org_limit: int = 100,
) -> dict:
    """Run outbound reconciliation for every org known to sync state."""

    async def _run() -> dict:
        sync_service = get_environment_sync_service()
        org_ids = sync_service.org_ids()
        results: dict[str, dict] = {}
        for org_id in org_ids:
            worker = EnvironmentReconciliationService(
                sync_service=sync_service,
                connector=ExternalConnectorService(),
                failure_handler=lambda state, error, _org_id=org_id: _handoff_reconcile_failure_to_dlq(
                    org_id=_org_id,
                    state=state,
                    error=error,
                ),
            )
            run = await worker.reconcile_org(
                org_id=org_id,
                max_lag_seconds=max_lag_seconds,
                limit=per_org_limit,
            )
            update_environment_sync_connector_metrics(org_id, sync_service.connector_summaries(org_id))
            run_status = "success" if run.failed == 0 else "partial_failure"
            record_environment_reconcile_run(
                run_status=run_status,
                succeeded=run.succeeded,
                failed=run.failed,
                dlq_handoffs=run.dlq_handoffs,
            )
            results[org_id] = run.to_dict()

        return {
            "org_count": len(org_ids),
            "results": results,
        }

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_run())
