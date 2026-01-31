"""Task execution wrapper with resource profiling.

Wraps task execution to automatically track resource metrics (tokens, latency, cost).
"""

from __future__ import annotations

import time
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from app.core.resource_profiler import resource_profiler

logger = logging.getLogger(__name__)


class TaskExecutionContext:
    """Context for tracking task execution metrics."""

    def __init__(
        self,
        task_id: str,
        organization_id: str,
        task_type: str,
        estimated_tokens: int = 0,
    ):
        self.task_id = task_id
        self.organization_id = organization_id
        self.task_type = task_type
        self.estimated_tokens = estimated_tokens

        # Timing
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.queued_start: float | None = None

        # Resource tracking
        self.actual_tokens: int = 0
        self.model_latency_ms: float = 0.0
        self.preprocessing_ms: float = 0.0
        self.postprocessing_ms: float = 0.0
        self.peak_memory_mb: float = 0.0
        self.avg_memory_mb: float = 0.0

        # Status
        self.succeeded: bool = False
        self.error_message: Optional[str] = None

    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

    def queued_duration_ms(self) -> float:
        """Get queued duration in milliseconds."""
        if self.queued_start is None or self.start_time is None:
            return 0.0
        return (self.start_time - self.queued_start) * 1000.0

    def record_metrics(self) -> None:
        """Record metrics to profiler."""
        try:
            # Calculate estimated cost based on tokens
            # Ollama local = basically free, use $0.0001 per 1K tokens as baseline
            cost_per_1k = 0.0001  # Conservative estimate for local models
            estimated_cost_usd = (self.actual_tokens / 1000.0) * cost_per_1k

            resource_profiler.record_metrics(
                task_id=self.task_id,
                organization_id=self.organization_id,
                task_type=self.task_type,
                queued_duration_ms=self.queued_duration_ms(),
                execution_duration_ms=self.duration_ms(),
                estimated_tokens=self.estimated_tokens,
                actual_tokens=self.actual_tokens,
                model_latency_ms=self.model_latency_ms,
                preprocessing_ms=self.preprocessing_ms,
                postprocessing_ms=self.postprocessing_ms,
                peak_memory_mb=self.peak_memory_mb,
                avg_memory_mb=self.avg_memory_mb,
                estimated_cost_usd=estimated_cost_usd,
                succeeded=self.succeeded,
            )
            logger.info(
                f"Task {self.task_type}:{self.task_id} recorded: "
                f"duration={self.duration_ms():.1f}ms, "
                f"tokens={self.actual_tokens}, cost=${estimated_cost_usd:.6f}"
            )
        except Exception as e:
            logger.exception(f"Failed to record metrics for task {self.task_id}: {e}")
            # Don't fail task if profiling fails


@asynccontextmanager
async def execute_with_profiling(
    task_id: str,
    organization_id: str,
    task_type: str,
    estimated_tokens: int = 0,
):
    """
    Context manager for task execution with automatic profiling.

    Usage:
        async with execute_with_profiling(
            task_id="task-123",
            organization_id="org-456",
            task_type="CONSOLIDATION",
            estimated_tokens=500,
        ) as ctx:
            # Do work here
            ctx.actual_tokens = 520
            ctx.model_latency_ms = 1500.0
            ctx.preprocessing_ms = 100.0
            ctx.postprocessing_ms = 50.0
            result = await do_work()
            ctx.succeeded = True
            return result

    Args:
        task_id: Unique task identifier
        organization_id: Organization executing the task
        task_type: Type of task (CONSOLIDATION, CRITIQUE, etc.)
        estimated_tokens: Estimated token usage

    Yields:
        TaskExecutionContext: Context object for tracking metrics
    """
    ctx = TaskExecutionContext(
        task_id=task_id,
        organization_id=organization_id,
        task_type=task_type,
        estimated_tokens=estimated_tokens,
    )

    # Mark queued time (when context was created)
    ctx.queued_start = time.perf_counter()

    try:
        # Mark execution start
        ctx.start_time = time.perf_counter()

        yield ctx

        # Mark execution end
        ctx.end_time = time.perf_counter()

        # If not explicitly set to failed, mark as succeeded
        if not ctx.succeeded:
            ctx.succeeded = True

    except Exception as e:
        # Mark execution end on error
        ctx.end_time = time.perf_counter()
        ctx.succeeded = False
        ctx.error_message = str(e)
        logger.error(
            f"Task {task_type}:{task_id} failed: {e}",
            exc_info=True,
        )
        # Don't re-raise; let caller decide
    finally:
        # Always record metrics, even on failure
        ctx.record_metrics()


def profile_execution(
    *,
    organization_id: str,
    task_type: str,
    estimated_tokens: int = 0,
):
    """
    Decorator for task execution with automatic profiling.

    Usage:
        @profile_execution(
            organization_id="org-123",
            task_type="CONSOLIDATION",
            estimated_tokens=500,
        )
        async def consolidate_memories(task_id: str, memories: list[str]):
            # Function body
            # Can access context via get_current_task_context()
            return result

    Args:
        organization_id: Organization executing task
        task_type: Type of task
        estimated_tokens: Estimated token usage
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(task_id: str, *args, **kwargs) -> Any:
            async with execute_with_profiling(
                task_id=task_id,
                organization_id=organization_id,
                task_type=task_type,
                estimated_tokens=estimated_tokens,
            ) as ctx:
                # Call the function
                result = await func(task_id, *args, **kwargs, _profile_ctx=ctx)
                return result

        return wrapper

    return decorator
