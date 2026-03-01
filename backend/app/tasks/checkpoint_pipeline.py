"""Checkpoint persistence task (PR5: Replayability)."""

from __future__ import annotations

import asyncio
from datetime import datetime

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.services.checkpoint_service import CheckpointService


logger = get_task_logger(__name__)


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    return asyncio.run(coro)


async def persist_checkpoint(
    *,
    org_id: str,
    agent_run_id: str,
    step_index: int,
    input_snapshot: dict,
    retrieval_snapshot: dict,
    model_snapshot: dict,
    output_snapshot: dict,
    actor_user_id: str,
) -> str:
    """Persist a checkpoint snapshot for an agent run step.
    
    This captures the complete deterministic state at each step for replay/audit.
    
    Args:
        org_id: Organization ID
        agent_run_id: Agent run ID
        step_index: Sequential step number (0-based)
        input_snapshot: Input state (query, params, etc.)
        retrieval_snapshot: Retrieved memories with ids/scores
        model_snapshot: Model config/state
        output_snapshot: Step output/response
        actor_user_id: System user creating checkpoint
        
    Returns:
        Checkpoint ID
    """
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(
                db,
                actor_user_id,
                org_id,
                roles="system,org_admin",
                clearance_level=4,
            )

            svc = CheckpointService(db, org_id)
            checkpoint_id = await svc.create_checkpoint(
                agent_run_id=agent_run_id,
                step_index=step_index,
                input_snapshot=input_snapshot,
                retrieval_snapshot=retrieval_snapshot,
                model_snapshot=model_snapshot,
                output_snapshot=output_snapshot,
            )

            logger.info(
                f"Checkpoint {checkpoint_id} persisted for run {agent_run_id} step {step_index}"
            )
            return checkpoint_id


@celery_app.task(name="checkpoint_persister_task", queue="q.agent_enrich")
def checkpoint_persister_task(
    org_id: str,
    agent_run_id: str,
    step_index: int,
    input_snapshot: dict,
    retrieval_snapshot: dict,
    model_snapshot: dict,
    output_snapshot: dict,
    actor_user_id: str,
) -> str:
    """Celery task wrapper for async checkpoint persistence.
    
    Routes to q.agent_enrich for async processing.
    """
    return _run_async(
        persist_checkpoint(
            org_id=org_id,
            agent_run_id=agent_run_id,
            step_index=step_index,
            input_snapshot=input_snapshot,
            retrieval_snapshot=retrieval_snapshot,
            model_snapshot=model_snapshot,
            output_snapshot=output_snapshot,
            actor_user_id=actor_user_id,
        )
    )


def enqueue_checkpoint_persistence(
    *,
    org_id: str,
    agent_run_id: str,
    step_index: int,
    input_snapshot: dict,
    retrieval_snapshot: dict,
    model_snapshot: dict,
    output_snapshot: dict,
    initiator_user_id: str,
) -> None:
    """Enqueue checkpoint persistence for async processing.
    
    Helper that respects memory broker disabled state for tests.
    """
    try:
        checkpoint_persister_task.apply_async(
            kwargs={
                "org_id": org_id,
                "agent_run_id": agent_run_id,
                "step_index": step_index,
                "input_snapshot": input_snapshot,
                "retrieval_snapshot": retrieval_snapshot,
                "model_snapshot": model_snapshot,
                "output_snapshot": output_snapshot,
                "actor_user_id": initiator_user_id,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to enqueue checkpoint persistence: {e}")
