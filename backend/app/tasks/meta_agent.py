"""Meta Agent Supervision & Calibration Celery tasks.

Celery tasks are synchronous entrypoints; we bridge into async.
Tenant context (RLS) MUST be set for every DB session.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.services.meta_agent.meta_supervisor import MetaSupervisor
from app.services.meta_agent.calibration_service import CalibrationService
from app.services.audit_service import AuditService
from app.models.meta_agent import MetaAgentRun
from app.models.cognitive_session import CognitiveSession
from app.models.tool_call_log import ToolCallLog
from app.models.memory_feedback import MemoryFeedback
from app.models.memory_promotion_history import MemoryPromotionHistory
from app.models.memory import MemoryMetadata
from sqlalchemy import func, select
from sqlalchemy.sql import case


logger = get_task_logger(__name__)


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


async def _calibration_update_impl(
    *,
    org_id: str,
    initiator_user_id: str,
    signal_weights: dict[str, float] | None = None,
    learning_rate: float | None = None,
    promotion_threshold: float | None = None,
    conflict_escalation_threshold: float | None = None,
    drift_threshold: float | None = None,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "calibration_update",
) -> str:
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(
                db,
                initiator_user_id,
                org_id,
                roles=roles,
                clearance_level=int(clearance_level or 0),
                justification=justification,
            )
            svc = CalibrationService()
            profile = await svc.update_profile(
                db,
                org_id=org_id,
                signal_weights=signal_weights,
                learning_rate=learning_rate,
                promotion_threshold=promotion_threshold,
                conflict_escalation_threshold=conflict_escalation_threshold,
                drift_threshold=drift_threshold,
            )

            # Spec: calibration learning may be triggered without explicit weights.
            if signal_weights is None:
                profile = await svc.learn_from_feedback(db, org_id=org_id)

            await AuditService(db).log_event(
                event_type="policy.calibration_update",
                actor_id=initiator_user_id,
                organization_id=org_id,
                resource_type="calibration_profile",
                resource_id=org_id,
                success=True,
                details={"signal_weights": profile.signal_weights, "learning_triggered": signal_weights is None},
            )
            return profile.organization_id


@celery_app.task(name="app.tasks.meta_agent.meta_review_memory_task")
def meta_review_memory_task(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "meta_review_memory",
    trace_id: str | None = None,
) -> str:
    async def _run() -> str:
        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )
                run = await MetaSupervisor().review_memory(db, org_id=org_id, memory_id=memory_id, trace_id=trace_id)
                return run.id

    return _run_async(_run())


@celery_app.task(name="app.tasks.meta_agent.meta_review_cognitive_session_task")
def meta_review_cognitive_session_task(
    *,
    org_id: str,
    session_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "meta_review_cognitive_session",
    trace_id: str | None = None,
) -> str:
    async def _run() -> str:
        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )
                run = await MetaSupervisor().review_cognitive_session(
                    db, org_id=org_id, session_id=session_id, trace_id=trace_id
                )
                return run.id

    return _run_async(_run())


@celery_app.task(name="app.tasks.meta_agent.calibration_update_task")
def calibration_update_task(
    *,
    org_id: str,
    initiator_user_id: str,
    signal_weights: dict[str, float] | None = None,
    learning_rate: float | None = None,
    promotion_threshold: float | None = None,
    conflict_escalation_threshold: float | None = None,
    drift_threshold: float | None = None,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "calibration_update",
) -> str:
    return _run_async(
        _calibration_update_impl(
            org_id=org_id,
            initiator_user_id=initiator_user_id,
            signal_weights=signal_weights,
            learning_rate=learning_rate,
            promotion_threshold=promotion_threshold,
            conflict_escalation_threshold=conflict_escalation_threshold,
            drift_threshold=drift_threshold,
            roles=roles,
            clearance_level=clearance_level,
            justification=justification,
        )
    )


@celery_app.task(name="app.tasks.meta_agent.meta_conflict_sweep_task")
def meta_conflict_sweep_task(
    *,
    org_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "meta_conflict_sweep",
) -> str:
    async def _run() -> str:
        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(
                    db,
                    initiator_user_id,
                    org_id,
                    roles=roles,
                    clearance_level=int(clearance_level or 0),
                    justification=justification,
                )

                # No-op sweep for now. Kept to satisfy spec task surface.
                return "ok"

    return _run_async(_run())
