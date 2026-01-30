"""Celery application configuration.

Follows ASYNC_PIPELINE_CELERY.md:
- Multiple queues for light/heavy tasks
- Explicit routing per task type
- Import-safe defaults (memory broker) for unit tests
"""

from __future__ import annotations

import importlib

from celery import Celery
from celery.schedules import crontab
from kombu import Queue

from app.core.config import settings


def _load_enterprise_celery_config() -> tuple[list[str], dict, dict]:
    """Best-effort enterprise Celery extensions.

    Community builds must run without the enterprise package installed.
    """

    try:
        mod = importlib.import_module("ninai_enterprise.celery")
    except ImportError:
        return [], {}, {}
    except Exception:
        # Never block app/worker startup on enterprise import issues.
        return [], {}, {}

    includes = list(getattr(mod, "CELERY_INCLUDES", []) or [])
    task_routes = dict(getattr(mod, "CELERY_TASK_ROUTES", {}) or {})
    beat_schedule = dict(getattr(mod, "CELERY_BEAT_SCHEDULE", {}) or {})
    return includes, task_routes, beat_schedule


_enterprise_includes, _enterprise_routes, _enterprise_beat = _load_enterprise_celery_config()


def _default_broker() -> str:
    # Keep imports safe in dev/tests even without Redis.
    return settings.CELERY_BROKER_URL or "memory://"


def _default_backend() -> str:
    # Cache-like in-memory backend for tests.
    return settings.CELERY_RESULT_BACKEND or "cache+memory://"


celery_app = Celery(
    "ninai",
    broker=_default_broker(),
    backend=_default_backend(),
    include=[
        # Memory activation scoring tasks live under services/ (imported for registration)
        "app.services.memory_activation.tasks",
        "app.tasks.memory_pipeline",
        "app.tasks.maintenance",
        "app.tasks.webhooks",
        "app.tasks.export_jobs",
        "app.tasks.cognitive_loop",
        "app.tasks.meta_agent",
        "app.tasks.goals",
        "app.tasks.self_model",
        "app.tasks.agent_processes",
        *_enterprise_includes,
    ],
)


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="q.agent_enrich",
    task_queues=(
        Queue("q.memory_ingest"),
        Queue("q.agent_enrich"),
        Queue("q.agent_topics"),
        Queue("q.agent_patterns"),
        Queue("q.agent_graph"),
        Queue("q.agent_feedback"),
        Queue("q.cognitive_loop"),
        Queue("q.meta_agent"),
        Queue("q.maintenance"),
        Queue("q.webhooks"),
    ),
    task_routes={
        "app.tasks.memory_pipeline.classification_task": {"queue": "q.agent_enrich"},
        "app.tasks.memory_pipeline.metadata_task": {"queue": "q.agent_enrich"},
        "app.tasks.memory_pipeline.topic_modeling_task": {"queue": "q.agent_topics"},
        "app.tasks.memory_pipeline.pattern_detection_task": {"queue": "q.agent_patterns"},
        "app.tasks.memory_pipeline.promotion_task": {"queue": "q.agent_patterns"},
        "app.tasks.memory_pipeline.graph_linking_task": {"queue": "q.agent_graph"},
        "app.tasks.memory_pipeline.logseq_export_task": {"queue": "q.agent_graph"},
        "app.tasks.memory_pipeline.feedback_learning_task": {"queue": "q.agent_feedback"},
        "app.tasks.maintenance.nightly_logseq_export_task": {"queue": "q.maintenance"},
        "app.tasks.maintenance.cleanup_expired_snapshot_exports_task": {"queue": "q.maintenance"},
        "app.tasks.webhooks.dispatch_webhooks_task": {"queue": "q.webhooks"},
        "app.tasks.export_jobs.run_snapshot_export_job_task": {"queue": "q.maintenance"},
        "app.tasks.cognitive_loop.cognitive_loop_task": {"queue": "q.cognitive_loop"},
        "app.tasks.meta_agent.meta_review_memory_task": {"queue": "q.meta_agent"},
        "app.tasks.meta_agent.meta_review_cognitive_session_task": {"queue": "q.meta_agent"},
        "app.tasks.meta_agent.calibration_update_task": {"queue": "q.meta_agent"},
        "app.tasks.meta_agent.meta_conflict_sweep_task": {"queue": "q.meta_agent"},

        # GoalGraph
        "app.tasks.goals.goal_plan_from_session_task": {"queue": "q.agent_enrich"},
        "app.tasks.goals.goal_link_memory_task": {"queue": "q.agent_enrich"},
        "app.tasks.goals.goal_progress_recompute_task": {"queue": "q.agent_enrich"},
        "app.tasks.goals.goal_blocker_detection_task": {"queue": "q.agent_enrich"},

        # SelfModel
        "app.tasks.self_model.self_model_recompute_task": {"queue": "q.agent_enrich"},

        # Agent process scheduling
        "app.tasks.agent_processes.dequeue_next_process_task": {"queue": "q.maintenance"},
        "app.tasks.agent_processes.schedule_dequeue_and_run_task": {"queue": "q.maintenance"},

        # Memory activation scoring (lightweight async updates)
        "app.services.memory_activation.tasks.memory_access_update_task": {"queue": "q.agent_enrich"},
        "app.services.memory_activation.tasks.coactivation_update_task": {"queue": "q.agent_graph"},
        "app.services.memory_activation.tasks.nightly_decay_refresh_task": {"queue": "q.maintenance"},
        "app.services.memory_activation.tasks.causal_hypothesis_update_task": {"queue": "q.maintenance"},
        **_enterprise_routes,
    },
    beat_schedule={
        # Default schedule: 02:05 UTC daily
        "nightly-logseq-export": {
            "task": "app.tasks.maintenance.nightly_logseq_export_task",
            "schedule": crontab(minute=5, hour=2),
            "args": (),
        },
        "cleanup-expired-snapshot-exports": {
            "task": "app.tasks.maintenance.cleanup_expired_snapshot_exports_task",
            "schedule": crontab(minute=35, hour=2),
            "args": (),
        },
        "nightly-memory-decay-refresh": {
            "task": "app.services.memory_activation.tasks.nightly_decay_refresh_task",
            "schedule": crontab(minute=15, hour=2),
            "args": (),
        },
        "dispatch-webhooks": {
            "task": "app.tasks.webhooks.dispatch_webhooks_task",
            "schedule": 30.0,
            "args": (),
        },
        **_enterprise_beat,
    },
)
