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

import asyncio

from celery import chain, group
from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.agent_runner import AgentRunner, PipelineContext


logger = get_task_logger(__name__)


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
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


def build_memory_dag(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    # 1) classification
    # 2) metadata
    # 3) topic modeling (after 1+2)
    # 4) pattern detection
    # 5) promotion
    # 6) graph linking (after 2/3)
    # 7) logseq export (optional)
    # 8) feedback learning

    base_kwargs = {
        "org_id": org_id,
        "memory_id": memory_id,
        "initiator_user_id": initiator_user_id,
        "trace_id": trace_id,
        "storage": storage,
    }

    enrich = chain(
        classification_task.si(**base_kwargs),
        metadata_task.si(**base_kwargs),
    )

    topics_then_patterns = chain(
        topic_modeling_task.si(**base_kwargs),
        pattern_detection_task.si(**base_kwargs),
        promotion_task.si(**base_kwargs),
    )

    graph_and_export = group(
        graph_linking_task.si(**base_kwargs),
        logseq_export_task.si(**base_kwargs),
    )

    # classification+metadata -> (topics->patterns->promotion) -> (graph+logseq) -> feedback
    return chain(enrich, topics_then_patterns, graph_and_export, feedback_learning_task.si(**base_kwargs))


def enqueue_memory_pipeline(**kwargs):
    """Enqueue the pipeline if Celery is configured; otherwise no-op."""

    broker = celery_app.conf.broker_url
    # In unit tests we default to memory://. Treat that as disabled for enqueuing.
    if not broker or str(broker).startswith("memory://"):
        return None

    sig = build_memory_dag(**kwargs)
    return sig.apply_async()


def enqueue_feedback_learning(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    """Enqueue only the feedback learning task if Celery is configured; otherwise no-op."""

    broker = celery_app.conf.broker_url
    if not broker or str(broker).startswith("memory://"):
        return None

    return feedback_learning_task.si(
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        trace_id=trace_id,
        storage=storage,
    ).apply_async()


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def classification_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="classification", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def metadata_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="metadata", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def topic_modeling_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="topics", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def pattern_detection_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="patterns", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def promotion_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="promotion", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def graph_linking_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="graph", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def logseq_export_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="logseq_export", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def feedback_learning_task(self, org_id: str, memory_id: str, initiator_user_id: str | None = None, trace_id: str | None = None, storage: str = "long_term"):
    runner = AgentRunner(service_user_id=getattr(settings, "SYSTEM_TASK_USER_ID", None) or None)
    ctx = PipelineContext(org_id=org_id, memory_id=memory_id, initiator_user_id=initiator_user_id, trace_id=trace_id, storage=storage)
    res = _run_async(runner.run_agent(ctx=ctx, agent_name="feedback", attempt=self.request.retries + 1))
    return res.model_dump(mode="json") if hasattr(res, "model_dump") else res
