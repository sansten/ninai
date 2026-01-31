"""Resource profiling for tracking task resource consumption.

Measures per-task:
- Token usage (estimated and actual)
- Latency (queue wait, execution time)
- Memory consumption (if available)
- Cost (based on LLM pricing)
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from uuid import UUID
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource consumption metrics for a task."""
    
    task_id: str
    organization_id: str
    task_type: str
    
    # Timing (milliseconds)
    queued_duration_ms: int = 0
    execution_duration_ms: int = 0
    total_duration_ms: int = 0
    
    # Tokens
    estimated_tokens: int = 0
    actual_tokens: int = 0
    
    # Latency breakdown
    model_latency_ms: int = 0  # Time waiting for LLM response
    preprocessing_ms: int = 0  # Data prep time
    postprocessing_ms: int = 0  # Result processing time
    
    # Memory (approximate, in MB)
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Cost (USD, estimated)
    estimated_cost_usd: float = 0.0
    
    # Metadata
    succeeded: bool = True
    error: Optional[str] = None
    recorded_at: datetime = None
    
    def __post_init__(self):
        if self.recorded_at is None:
            self.recorded_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling datetime serialization."""
        d = asdict(self)
        d['recorded_at'] = self.recorded_at.isoformat()
        return d


class ResourceProfiler:
    """Profiles resource consumption of tasks."""

    def __init__(self):
        self.metrics: Dict[str, ResourceMetrics] = {}

    def start_profile(self, task_id: str) -> "TaskProfiler":
        """
        Start profiling a task.
        
        Returns:
            TaskProfiler context manager
        """
        return TaskProfiler(self, task_id)

    def record_metrics(
        self,
        task_id: str,
        organization_id: str,
        task_type: str,
        queued_duration_ms: int = 0,
        execution_duration_ms: int = 0,
        estimated_tokens: int = 0,
        actual_tokens: int = 0,
        model_latency_ms: int = 0,
        preprocessing_ms: int = 0,
        postprocessing_ms: int = 0,
        peak_memory_mb: float = 0.0,
        avg_memory_mb: float = 0.0,
        estimated_cost_usd: float = 0.0,
        succeeded: bool = True,
        error: Optional[str] = None,
    ) -> ResourceMetrics:
        """Record resource metrics for a task."""
        total_duration_ms = queued_duration_ms + execution_duration_ms
        
        metrics = ResourceMetrics(
            task_id=str(task_id),
            organization_id=str(organization_id),
            task_type=task_type,
            queued_duration_ms=queued_duration_ms,
            execution_duration_ms=execution_duration_ms,
            total_duration_ms=total_duration_ms,
            estimated_tokens=estimated_tokens,
            actual_tokens=actual_tokens,
            model_latency_ms=model_latency_ms,
            preprocessing_ms=preprocessing_ms,
            postprocessing_ms=postprocessing_ms,
            peak_memory_mb=peak_memory_mb,
            avg_memory_mb=avg_memory_mb,
            estimated_cost_usd=estimated_cost_usd,
            succeeded=succeeded,
            error=error,
        )
        
        self.metrics[str(task_id)] = metrics
        
        # Log summary
        logger.info(
            f"Task profiling: {task_type} | "
            f"Duration: {total_duration_ms}ms | "
            f"Tokens: {actual_tokens}/{estimated_tokens} | "
            f"Cost: ${estimated_cost_usd:.4f}"
        )
        
        return metrics

    def get_metrics(self, task_id: str) -> Optional[ResourceMetrics]:
        """Get metrics for a specific task."""
        return self.metrics.get(str(task_id))

    def get_org_summary(
        self,
        organization_id: str,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Get resource summary for an organization."""
        org_metrics = [
            m for m in self.metrics.values()
            if m.organization_id == str(organization_id)
        ]
        
        if not org_metrics:
            return {
                "organization_id": str(organization_id),
                "total_tasks": 0,
                "summary": {},
            }
        
        total_tasks = len(org_metrics)
        successful = len([m for m in org_metrics if m.succeeded])
        failed = total_tasks - successful
        
        total_tokens = sum(m.actual_tokens for m in org_metrics)
        total_cost = sum(m.estimated_cost_usd for m in org_metrics)
        avg_duration_ms = sum(m.total_duration_ms for m in org_metrics) / max(total_tasks, 1)
        
        # Group by task type
        by_type = {}
        for metric in org_metrics:
            if metric.task_type not in by_type:
                by_type[metric.task_type] = {
                    "count": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "avg_duration_ms": 0,
                }
            by_type[metric.task_type]["count"] += 1
            by_type[metric.task_type]["total_tokens"] += metric.actual_tokens
            by_type[metric.task_type]["total_cost"] += metric.estimated_cost_usd
            by_type[metric.task_type]["avg_duration_ms"] += metric.total_duration_ms
        
        # Calculate averages
        for task_type_data in by_type.values():
            task_type_data["avg_duration_ms"] = int(
                task_type_data["avg_duration_ms"] / max(task_type_data["count"], 1)
            )
        
        return {
            "organization_id": str(organization_id),
            "total_tasks": total_tasks,
            "successful": successful,
            "failed": failed,
            "summary": {
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 2),
                "avg_duration_ms": int(avg_duration_ms),
                "avg_tokens_per_task": int(total_tokens / max(total_tasks, 1)),
                "avg_cost_per_task": round(total_cost / max(total_tasks, 1), 4),
            },
            "by_task_type": by_type,
        }

    def get_memory_info(self) -> Dict[str, float]:
        """Get current process memory usage."""
        try:
            memory_info = self.process.memory_info()
            mem_percent = self.process.memory_percent()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),  # Resident set size
                "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual memory size
                "percent": mem_percent,
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}

    def cleanup_old_metrics(self, keep_count: int = 10000):
        """Keep only the most recent metrics to prevent memory bloat."""
        if len(self.metrics) > keep_count:
            # Keep newest metrics
            sorted_items = sorted(
                self.metrics.items(),
                key=lambda x: x[1].recorded_at,
                reverse=True
            )
            self.metrics = dict(sorted_items[:keep_count])
            logger.info(
                f"Cleaned up metrics: kept {keep_count} most recent, "
                f"removed {len(sorted_items) - keep_count}"
            )


class TaskProfiler:
    """Context manager for profiling a single task."""

    def __init__(self, profiler: ResourceProfiler, task_id: str):
        self.profiler = profiler
        self.task_id = task_id
        self.start_time = time.time()
        self.start_memory = self.profiler.get_memory_info()
        
        # Will be set by context
        self.queued_duration_ms = 0
        self.model_latency_ms = 0
        self.preprocessing_ms = 0
        self.postprocessing_ms = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record metrics on exit."""
        execution_time = (time.time() - self.start_time) * 1000  # ms
        
        # Memory delta
        end_memory = self.profiler.get_memory_info()
        peak_memory = max(self.start_memory["rss_mb"], end_memory["rss_mb"])
        avg_memory = (self.start_memory["rss_mb"] + end_memory["rss_mb"]) / 2
        
        # Error tracking
        error = None
        succeeded = exc_type is None
        if exc_val:
            error = str(exc_val)

    def set_latencies(
        self,
        queued_ms: int = 0,
        model_ms: int = 0,
        preprocessing_ms: int = 0,
        postprocessing_ms: int = 0,
    ):
        """Set latency breakdown."""
        self.queued_duration_ms = queued_ms
        self.model_latency_ms = model_ms
        self.preprocessing_ms = preprocessing_ms
        self.postprocessing_ms = postprocessing_ms


# Global profiler instance
resource_profiler = ResourceProfiler()


def get_resource_profiler() -> ResourceProfiler:
    """Get global resource profiler instance."""
    return resource_profiler
