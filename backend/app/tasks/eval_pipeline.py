"""Evaluation pipeline Celery tasks (PR6: Eval Harness + Drift Detection)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.models.eval_suite import EvalSuite
from app.models.eval_run import EvalRun
from app.services.eval_run_service import EvalRunService
from app.services.drift_detection_service import DriftDetectionService
from sqlalchemy import select

logger = logging.getLogger(__name__)


async def run_eval_suite_async(
    *,
    organization_id: str,
    suite_id: str,
    user_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Run an evaluation suite and compute metrics (async implementation).
    
    Args:
        organization_id: Organization ID
        suite_id: Eval suite ID
        user_id: User ID for tenant context
        config: Optional configuration overrides
        
    Returns:
        ID of the created eval run
    """
    async with async_session_factory() as session:
        try:
            # Set tenant context
            await set_tenant_context(
                session,
                user_id or "system",
                organization_id,
                roles="system",
                clearance_level=4,
            )

            # Get the suite
            result = await session.execute(
                select(EvalSuite).where(EvalSuite.id == suite_id)
            )
            suite = result.scalar_one_or_none()
            
            if not suite:
                logger.error(f"Eval suite {suite_id} not found")
                return ""

            # Create eval run
            service = EvalRunService(session, organization_id)
            eval_run_id = await service.create_eval_run(suite_id=suite_id, config=config)
            await session.commit()

            # Execute queries and collect results
            # NOTE: In production, this would call the actual memory read() API
            # For now, we'll simulate with the structure expected by compute_metrics
            query_results = []
            queries = suite.queries if isinstance(suite.queries, list) else []
            
            for idx, query_spec in enumerate(queries):
                # Simulate query execution
                # In production: actual_ids, memories = await memory_service.read(query_spec["query"])
                
                query_id = f"query_{idx}"
                expected_results = suite.expected.get(query_id, {})
                
                query_result = {
                    "query": query_spec.get("query", ""),
                    "actual_ids": expected_results.get("ids", []),  # Simulated
                    "expected_ids": expected_results.get("ids", []),
                    "actual_memories": [],  # Would be populated by real query
                    "leaked_orgs": [],  # RLS check would populate this
                    "policy_violations": [],  # Access control check
                    "latency_ms": 50.0,  # Simulated latency
                }
                query_results.append(query_result)

            # Compute metrics
            metrics = await service.compute_metrics(eval_run_id, query_results)
            
            # Finalize eval run
            await service.finalize_eval_run(eval_run_id, metrics, status="success")
            await session.commit()

            logger.info(f"Eval run {eval_run_id} completed successfully")
            return eval_run_id

        except Exception as e:
            logger.error(f"Error running eval suite {suite_id}: {e}")
            
            # Mark as failed if we have an eval_run_id
            if "eval_run_id" in locals():
                await service.finalize_eval_run(
                    eval_run_id, {}, status="failure", error=str(e)
                )
                await session.commit()
            
            raise


async def compute_drift_async(
    *,
    organization_id: str,
    baseline_run_id: str,
    current_run_id: str,
    user_id: str | None = None,
) -> str:
    """Compute drift between two eval runs (async implementation).
    
    Args:
        organization_id: Organization ID
        baseline_run_id: Baseline eval run ID
        current_run_id: Current eval run ID
        user_id: User ID for tenant context
        
    Returns:
        ID of the created drift report
    """
    async with async_session_factory() as session:
        try:
            # Set tenant context
            await set_tenant_context(
                session,
                user_id or "system",
                organization_id,
                roles="system",
                clearance_level=4,
            )

            # Compute drift
            service = DriftDetectionService(session, organization_id)
            drift_report_id = await service.compute_drift(baseline_run_id, current_run_id)
            await session.commit()

            logger.info(f"Drift report {drift_report_id} created successfully")
            return drift_report_id

        except Exception as e:
            logger.error(f"Error computing drift: {e}")
            raise


# Celery task wrappers

@celery_app.task(name="run_eval_suite_task", bind=True)
def run_eval_suite_task(
    self,
    organization_id: str,
    suite_id: str,
    user_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Celery task: Run an evaluation suite.
    
    Args:
        organization_id: Organization ID
        suite_id: Eval suite ID
        user_id: Optional user ID for tenant context
        config: Optional configuration overrides
        
    Returns:
        ID of the created eval run
    """
    return asyncio.run(
        run_eval_suite_async(
            organization_id=organization_id,
            suite_id=suite_id,
            user_id=user_id,
            config=config,
        )
    )


@celery_app.task(name="compute_drift_task", bind=True)
def compute_drift_task(
    self,
    organization_id: str,
    baseline_run_id: str,
    current_run_id: str,
    user_id: str | None = None,
) -> str:
    """Celery task: Compute drift between eval runs.
    
    Args:
        organization_id: Organization ID
        baseline_run_id: Baseline eval run ID
        current_run_id: Current eval run ID
        user_id: Optional user ID for tenant context
        
    Returns:
        ID of the created drift report
    """
    return asyncio.run(
        compute_drift_async(
            organization_id=organization_id,
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            user_id=user_id,
        )
    )


# Helper functions

def enqueue_eval_suite(
    organization_id: str,
    suite_id: str,
    user_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Enqueue an eval suite execution.
    
    Args:
        organization_id: Organization ID
        suite_id: Eval suite ID
        user_id: Optional user ID for tenant context
        config: Optional configuration overrides
    """
    try:
        run_eval_suite_task.apply_async(
            kwargs={
                "organization_id": organization_id,
                "suite_id": suite_id,
                "user_id": user_id,
                "config": config,
            },
        )
    except Exception as e:
        # If broker is not available (e.g., in tests), run synchronously
        logger.warning(f"Celery broker unavailable, running eval suite synchronously: {e}")
        asyncio.run(
            run_eval_suite_async(
                organization_id=organization_id,
                suite_id=suite_id,
                user_id=user_id,
                config=config,
            )
        )


def enqueue_drift_computation(
    organization_id: str,
    baseline_run_id: str,
    current_run_id: str,
    user_id: str | None = None,
) -> None:
    """Enqueue a drift computation.
    
    Args:
        organization_id: Organization ID
        baseline_run_id: Baseline eval run ID
        current_run_id: Current eval run ID
        user_id: Optional user ID for tenant context
    """
    try:
        compute_drift_task.apply_async(
            kwargs={
                "organization_id": organization_id,
                "baseline_run_id": baseline_run_id,
                "current_run_id": current_run_id,
                "user_id": user_id,
            },
        )
    except Exception as e:
        # If broker is not available (e.g., in tests), run synchronously
        logger.warning(f"Celery broker unavailable, running drift computation synchronously: {e}")
        asyncio.run(
            compute_drift_async(
                organization_id=organization_id,
                baseline_run_id=baseline_run_id,
                current_run_id=current_run_id,
                user_id=user_id,
            )
        )
