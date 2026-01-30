"""Agent process dequeue and scheduling worker task.

This task pulls the next runnable process from the queue and manages its lifecycle.
Optional: can trigger actual work execution if integrated with appropriate handlers.
"""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.services.agent_scheduler_service import AgentSchedulerService

logger = get_task_logger(__name__)


@celery_app.task(name="app.tasks.agent_processes.dequeue_next_process_task")
def dequeue_next_process_task(
    *,
    org_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "process_dequeue",
) -> dict[str, str | None]:
    """Dequeue and inspect the next runnable process for an organization.
    
    Returns the process_id if one is available, else None.
    Callers can use this to trigger appropriate handling (e.g., cognitive_loop_task).
    """

    async def _run() -> dict[str, str | None]:
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

                scheduler = AgentSchedulerService(db, auto_commit=False)
                proc = await scheduler.dequeue_next(org_id=org_id, scopes={"scheduler.dequeue"})

                if proc is None:
                    logger.debug(f"No runnable processes for org {org_id}")
                    return {"process_id": None, "agent_name": None}

                logger.info(
                    f"Dequeued process {proc.id}: agent={proc.agent_name}, attempts={proc.attempts}/{proc.max_attempts}"
                )
                return {
                    "process_id": str(proc.id),
                    "agent_name": proc.agent_name,
                    "session_id": str(proc.session_id) if proc.session_id else None,
                }

    try:
        return asyncio.run(_run())
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise

        import threading

        result: dict[str, dict[str, str | None]] = {}
        error: dict[str, BaseException] = {}

        def _thread_target() -> None:
            try:
                result["value"] = asyncio.run(_run())
            except BaseException as e:  # noqa: BLE001
                error["error"] = e

        t = threading.Thread(target=_thread_target, name="dequeue_task_asyncio")
        t.start()
        t.join()

        if "error" in error:
            raise error["error"]
        return result["value"]


@celery_app.task(name="app.tasks.agent_processes.schedule_dequeue_and_run_task")
def schedule_dequeue_and_run_task(
    *,
    org_id: str,
    initiator_user_id: str,
    roles: str = "",
    clearance_level: int = 0,
    justification: str = "schedule_dequeue",
) -> dict[str, str | None]:
    """Dequeue next process and immediately enqueue it for execution.
    
    This is a convenience task for continuous scheduling loops.
    """
    from app.tasks.cognitive_loop import cognitive_loop_task

    result = dequeue_next_process_task(
        org_id=org_id,
        initiator_user_id=initiator_user_id,
        roles=roles,
        clearance_level=clearance_level,
        justification=justification,
    )

    process_id = result.get("process_id")
    session_id = result.get("session_id")

    if process_id and session_id:
        logger.info(f"Scheduling cognitive loop for process {process_id}, session {session_id}")
        cognitive_loop_task.delay(
            org_id=org_id,
            session_id=session_id,
            initiator_user_id=initiator_user_id,
            process_id=process_id,
            roles=roles,
            clearance_level=clearance_level,
            justification=justification,
        )

    return result
