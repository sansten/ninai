"""Memory enrichment pipeline tasks.

Implements the DAG from ASYNC_PIPELINE_CELERY.md.

Tasks accept explicit context:
- org_id
- memory_id
- initiator_user_id (optional)
- trace_id
- storage (short_term|long_term)

Missing agents are recorded as skipped via AgentRunner.
"""

from __future__ import annotations

import threading

from celery import chain, group
from celery.utils.log import get_task_logger

from app.agents.registry import get_write_time_agent_spec, list_write_time_agent_specs
from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.agent_runner import AgentRunner, PipelineContext
from app.tasks.async_runtime import run_async


logger = get_task_logger(__name__)

DEFAULT_BULK_ENRICHMENT_AGENTS: tuple[str, ...] = (
    "entity_resolution",
    "world_model",
    "temporal_reasoning",
    "episodic_grouping",
    "causal_reasoning",
)


def _enabled_write_time_agent_tiers() -> set[int]:
    configured = getattr(settings, "WRITE_TIME_AGENT_ENABLED_TIERS", None) or [1, 2]
    return {int(value) for value in configured}


def _write_time_catalog(*, trigger_type: str) -> dict[str, object]:
    return {
        spec.key: spec
        for spec in list_write_time_agent_specs(trigger_type=trigger_type, enabled_tiers=None)
    }


def build_write_time_execution_plan(
    *,
    trigger_type: str,
    requested_agents: list[str] | tuple[str, ...] | None = None,
    enabled_tiers: set[int] | None = None,
):
    """Build a staged execution plan from registry metadata."""

    catalog = _write_time_catalog(trigger_type=trigger_type)
    if requested_agents:
        requested_keys: list[str] = []
        for agent_name in requested_agents:
            spec = get_write_time_agent_spec(agent_name)
            if spec is None or trigger_type not in spec.trigger_types:
                raise ValueError(f"Unknown agents: ['{agent_name}']")
            requested_keys.append(spec.key)

        selected_keys: set[str] = set()

        def _include_with_dependencies(key: str) -> None:
            if key in selected_keys:
                return
            spec = catalog.get(key)
            if spec is None:
                raise ValueError(f"Unsupported dependency for trigger {trigger_type}: {key}")
            for dep_key in spec.depends_on:
                _include_with_dependencies(dep_key)
            selected_keys.add(key)

        for key in requested_keys:
            _include_with_dependencies(key)
    else:
        selected_keys = set(
            spec.key
            for spec in list_write_time_agent_specs(
                trigger_type=trigger_type,
                enabled_tiers=enabled_tiers,
            )
        )

    if not selected_keys:
        return []

    order_index = {
        spec.key: idx
        for idx, spec in enumerate(list_write_time_agent_specs(trigger_type=trigger_type, enabled_tiers=None))
    }
    remaining = dict(sorted(
        ((key, catalog[key]) for key in selected_keys),
        key=lambda item: order_index.get(item[0], 10_000),
    ))

    stages: list[list[object]] = []
    while remaining:
        ready = [
            spec
            for key, spec in remaining.items()
            if all(dep_key not in remaining for dep_key in spec.depends_on)
        ]
        if not ready:
            unresolved = ", ".join(sorted(remaining))
            raise ValueError(f"Cyclic or unresolved write-time agent dependencies: {unresolved}")
        stages.append(ready)
        for spec in ready:
            remaining.pop(spec.key, None)

    return stages


def _build_registered_agent_signature(*, agent_name: str, queue: str, kwargs: dict):
    return run_registered_agent_task.si(agent_name=agent_name, **kwargs).set(queue=queue)


def _compose_stage_canvases(stages: list[list[object]], *, base_kwargs: dict):
    canvases = []
    for stage in stages:
        signatures = [
            _build_registered_agent_signature(
                agent_name=spec.agent_name,
                queue=spec.queue,
                kwargs=base_kwargs,
            )
            for spec in stage
        ]
        canvases.append(signatures[0] if len(signatures) == 1 else group(signatures))

    if not canvases:
        raise ValueError("No write-time agents selected for execution")
    if len(canvases) == 1:
        return canvases[0]
    return chain(*canvases)


def build_memory_enrichment_canvas(
    *,
    agent_names: list[str] | tuple[str, ...],
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    initiator_roles: str = "",
    initiator_clearance_level: int = 0,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    base_kwargs = {
        "org_id": org_id,
        "memory_id": memory_id,
        "initiator_user_id": initiator_user_id,
        "initiator_roles": initiator_roles,
        "initiator_clearance_level": initiator_clearance_level,
        "trace_id": trace_id,
        "storage": storage,
    }
    stages = build_write_time_execution_plan(
        trigger_type="memory_reenrich",
        requested_agents=list(agent_names),
        enabled_tiers=None,
    )
    return _compose_stage_canvases(stages, base_kwargs=base_kwargs)


def build_memory_dag(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    initiator_roles: str = "",
    initiator_clearance_level: int = 0,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    base_kwargs = {
        "org_id": org_id,
        "memory_id": memory_id,
        "initiator_user_id": initiator_user_id,
        "initiator_roles": initiator_roles,
        "initiator_clearance_level": initiator_clearance_level,
        "trace_id": trace_id,
        "storage": storage,
    }
    stages = build_write_time_execution_plan(
        trigger_type="memory_write",
        enabled_tiers=_enabled_write_time_agent_tiers(),
    )
    return _compose_stage_canvases(stages, base_kwargs=base_kwargs)


def _run_memory_pipeline_inline(**kwargs):
    """Best-effort in-process fallback for write-time enrichment.

    Used only when Celery broker is unavailable or enqueue fails.
    Executes the same registry-driven memory_write plan, stage by stage.
    """
    base_kwargs = {
        "org_id": kwargs["org_id"],
        "memory_id": kwargs["memory_id"],
        "initiator_user_id": kwargs.get("initiator_user_id"),
        "initiator_roles": kwargs.get("initiator_roles", ""),
        "initiator_clearance_level": int(kwargs.get("initiator_clearance_level", 0) or 0),
        "trace_id": kwargs.get("trace_id"),
        "storage": kwargs.get("storage", "long_term"),
    }

    try:
        stages = build_write_time_execution_plan(
            trigger_type="memory_write",
            enabled_tiers=_enabled_write_time_agent_tiers(),
        )
    except Exception as exc:
        logger.warning(
            "inline memory pipeline plan failed for memory_id=%s: %s",
            base_kwargs.get("memory_id"),
            exc,
        )
        return

    for stage in stages:
        for spec in stage:
            try:
                _run_registered_agent(
                    agent_name=spec.agent_name,
                    attempt=1,
                    org_id=base_kwargs["org_id"],
                    memory_id=base_kwargs["memory_id"],
                    initiator_user_id=base_kwargs["initiator_user_id"],
                    initiator_roles=base_kwargs["initiator_roles"],
                    initiator_clearance_level=base_kwargs["initiator_clearance_level"],
                    trace_id=base_kwargs["trace_id"],
                    storage=base_kwargs["storage"],
                )
            except Exception as exc:
                logger.warning(
                    "inline memory pipeline agent failed memory_id=%s agent=%s: %s",
                    base_kwargs.get("memory_id"),
                    getattr(spec, "agent_name", "unknown"),
                    exc,
                )


def _start_memory_pipeline_inline_fallback(**kwargs):
    memory_id = str(kwargs.get("memory_id") or "")
    t = threading.Thread(
        target=_run_memory_pipeline_inline,
        kwargs=kwargs,
        daemon=True,
        name=f"mem-inline-{memory_id[:8] or 'unknown'}",
    )
    t.start()
    return t


def enqueue_memory_pipeline(**kwargs):
    """Enqueue write-time enrichment pipeline.

    If Celery is unavailable, run a best-effort inline fallback in a background thread
    so foundational enrichment still materializes.
    """

    broker = celery_app.conf.broker_url
    # In unit tests we default to memory://. Keep write-time enrichment available
    # via inline fallback instead of silently skipping.
    if not broker or str(broker).startswith("memory://"):
        logger.warning(
            "memory pipeline broker unavailable for memory_id=%s (broker=%r); using inline fallback",
            kwargs.get("memory_id"),
            broker,
        )
        _start_memory_pipeline_inline_fallback(**kwargs)
        return None

    sig = build_memory_dag(**kwargs)
    try:
        return sig.apply_async(retry=False)
    except Exception as exc:
        logger.warning("memory pipeline enqueue failed for memory_id=%s: %s", kwargs.get("memory_id"), exc)
        _start_memory_pipeline_inline_fallback(**kwargs)
        return None


def enqueue_feedback_learning(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    initiator_roles: str = "",
    initiator_clearance_level: int = 0,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    """Enqueue only the feedback learning task if Celery is configured; otherwise no-op."""

    broker = celery_app.conf.broker_url
    if not broker or str(broker).startswith("memory://"):
        return None

    try:
        return feedback_learning_task.si(
            org_id=org_id,
            memory_id=memory_id,
            initiator_user_id=initiator_user_id,
            initiator_roles=initiator_roles,
            initiator_clearance_level=initiator_clearance_level,
            trace_id=trace_id,
            storage=storage,
        ).apply_async(retry=False)
    except Exception as exc:
        logger.warning("feedback pipeline enqueue failed for memory_id=%s: %s", memory_id, exc)
        return None


def _pipeline_ctx(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None,
    initiator_roles: str,
    initiator_clearance_level: int,
    trace_id: str | None,
    storage: str,
) -> PipelineContext:
    return PipelineContext(
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


def _run_registered_agent(
    *,
    agent_name: str,
    attempt: int,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None,
    initiator_roles: str,
    initiator_clearance_level: int,
    trace_id: str | None,
    storage: str,
):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = _pipeline_ctx(
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )
    res = run_async(runner.run_agent(ctx=ctx, agent_name=agent_name, attempt=attempt))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def run_registered_agent_task(self, agent_name: str, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name=agent_name,
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def classification_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="classification",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def metadata_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="metadata",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def topic_modeling_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="topics",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def pattern_detection_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="patterns",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def promotion_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="promotion",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def graph_linking_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="graph",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def logseq_export_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="logseq_export",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def semantic_normalization_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="semantic_normalization",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def feedback_learning_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="feedback",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def entity_resolution_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="entity_resolution",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def world_model_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="world_model",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def temporal_reasoning_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="temporal_reasoning",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def episodic_grouping_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="episodic_grouping",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def causal_reasoning_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, initiator_roles: str = "", initiator_clearance_level: int = 0, trace_id: str | None = None, storage: str = "long_term"):
    return _run_registered_agent(
        agent_name="causal_reasoning",
        attempt=self.request.retries + 1,
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        initiator_roles=initiator_roles,
        initiator_clearance_level=initiator_clearance_level,
        trace_id=trace_id,
        storage=storage,
    )
