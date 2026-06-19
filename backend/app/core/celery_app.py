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
        "app.tasks.episode_pipeline",
        "app.tasks.fact_pipeline",
        "app.tasks.playbook_pipeline",
        "app.tasks.checkpoint_pipeline",
        "app.tasks.eval_pipeline",
        "app.tasks.maintenance",
        "app.tasks.webhooks",
        "app.tasks.export_jobs",
        "app.tasks.cognitive_loop",
        "app.tasks.meta_agent",
        "app.tasks.goals",
        "app.tasks.self_model",
        "app.tasks.agent_processes",
        "app.tasks.feature_readiness_task",
        "app.tasks.memory_sleep_pipeline",
        "app.tasks.digest_pipeline",
        "app.tasks.gdpr_pipeline",
        "app.tasks.environment_sync",
        "app.tasks.strategy_evolution",
        "app.tasks.playbook_auto_synthesis_pipeline",
        "app.tasks.retention_enforcement",
        "app.tasks.cognitive_heartbeat",
        "app.tasks.proactive_push_beat",
        "app.tasks.v2_enrich_task",
        *_enterprise_includes,
    ],
)


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_publish_retry=False,
    broker_connection_timeout=2,
    broker_transport_options={"socket_timeout": 2, "socket_connect_timeout": 2},
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="q.agent_enrich",
    task_queues=(
        Queue("q.memory_ingest"),
        Queue("q.agent_enrich"),
        Queue("q.agent_topics"),
        Queue("q.agent_patterns"),
        Queue("q.agent_graph"),
        Queue("q.agent_reasoning"),
        Queue("q.agent_feedback"),
        Queue("q.cognitive_loop"),
        Queue("q.meta_agent"),
        Queue("q.maintenance"),
        Queue("q.webhooks"),
    ),
    task_routes={
        "app.tasks.memory_pipeline.run_registered_agent_task": {"queue": "q.agent_enrich"},
        "app.tasks.memory_pipeline.classification_task": {"queue": "q.agent_enrich"},
        "app.tasks.memory_pipeline.metadata_task": {"queue": "q.agent_enrich"},
        "app.tasks.memory_pipeline.semantic_normalization_task": {"queue": "q.agent_enrich"},
        "app.tasks.memory_pipeline.topic_modeling_task": {"queue": "q.agent_topics"},
        "app.tasks.memory_pipeline.pattern_detection_task": {"queue": "q.agent_patterns"},
        "app.tasks.memory_pipeline.promotion_task": {"queue": "q.agent_patterns"},
        "app.tasks.memory_pipeline.entity_resolution_task": {"queue": "q.agent_graph"},
        "app.tasks.memory_pipeline.graph_linking_task": {"queue": "q.agent_graph"},
        "app.tasks.memory_pipeline.logseq_export_task": {"queue": "q.agent_graph"},
        "app.tasks.memory_pipeline.world_model_task": {"queue": "q.agent_reasoning"},
        "app.tasks.memory_pipeline.temporal_reasoning_task": {"queue": "q.agent_reasoning"},
        "app.tasks.memory_pipeline.episodic_grouping_task": {"queue": "q.agent_reasoning"},
        "app.tasks.memory_pipeline.causal_reasoning_task": {"queue": "q.agent_reasoning"},
        "app.tasks.memory_pipeline.feedback_learning_task": {"queue": "q.agent_feedback"},
        "app.tasks.episode_pipeline.episode_router_task": {"queue": "q.agent_enrich"},
        "app.tasks.episode_pipeline.episode_summarizer_task": {"queue": "q.agent_feedback"},
        "app.tasks.fact_pipeline.fact_extractor_task": {"queue": "q.agent_enrich"},
        "app.tasks.playbook_pipeline.playbook_extractor_task": {"queue": "q.agent_feedback"},
        "app.tasks.checkpoint_pipeline.checkpoint_persister_task": {"queue": "q.agent_enrich"},
        "app.tasks.eval_pipeline.run_eval_suite_task": {"queue": "q.agent_feedback"},
        "app.tasks.eval_pipeline.compute_drift_task": {"queue": "q.agent_feedback"},
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

        # Feature readiness (Phase 28)
        "app.tasks.feature_readiness_task.evaluate_feature_readiness_task": {"queue": "q.maintenance"},

        # Memory sleep pipeline (Phase 40)
        "app.tasks.memory_sleep_pipeline.memory_sleep_task": {"queue": "q.maintenance"},

        # Intelligence digest pipeline (P46)
        "app.tasks.digest_pipeline.digest_task": {"queue": "q.maintenance"},

        # GDPR deletion pipeline (P47)
        "app.tasks.gdpr_pipeline.process_deletion_task": {"queue": "q.maintenance"},

        # Environment sync reconciliation (Phase 52)
        "app.tasks.environment_sync.reconcile_environment_sync_org_task": {"queue": "q.maintenance"},
        "app.tasks.environment_sync.reconcile_environment_sync_all_task": {"queue": "q.maintenance"},

        # Continuous learning — strategy evolution (Phase 50)
        "app.tasks.strategy_evolution.strategy_evolution_pipeline_task": {"queue": "q.maintenance"},
        "app.tasks.playbook_auto_synthesis_pipeline.playbook_auto_synthesis_task": {"queue": "q.maintenance"},

        # Cognitive OS heartbeat
        "app.tasks.cognitive_heartbeat.cognitive_heartbeat_task": {"queue": "q.cognitive_loop"},

        # Proactive intelligence push (Feature 24.10)
        "app.tasks.proactive_push_beat.proactive_push_beat_task": {"queue": "q.cognitive_loop"},

        # Retention policy enforcement — Gap D
        "app.tasks.retention_enforcement.retention_enforcement_task": {"queue": "q.maintenance"},

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
        "nightly-feature-readiness-evaluation": {
            "task": "app.tasks.feature_readiness_task.evaluate_feature_readiness_task",
            "schedule": crontab(minute=0, hour=3),
            "args": (),
        },
        "nightly-memory-sleep-cycle": {
            "task": "app.tasks.memory_sleep_pipeline.memory_sleep_task",
            "schedule": crontab(minute=0, hour=2),
            "args": (),
        },
        "daily-intelligence-digest": {
            "task": "app.tasks.digest_pipeline.digest_task",
            "schedule": crontab(minute=0, hour=6),
            "args": (),
        },
        "reconcile-environment-sync": {
            "task": "app.tasks.environment_sync.reconcile_environment_sync_all_task",
            "schedule": 60.0,
            "kwargs": {"max_lag_seconds": 30, "per_org_limit": 100},
        },
        "nightly-strategy-evolution": {
            "task": "app.tasks.strategy_evolution.strategy_evolution_pipeline_task",
            "schedule": crontab(minute=30, hour=3),
            "kwargs": {"window_days": 30},
        },
        "nightly-retention-enforcement": {
            "task": "app.tasks.retention_enforcement.retention_enforcement_task",
            "schedule": crontab(minute=30, hour=1),
            "args": (),
        },
        "nightly-playbook-auto-synthesis": {
            "task": "app.tasks.playbook_auto_synthesis_pipeline.playbook_auto_synthesis_task",
            "schedule": crontab(minute=0, hour=4),
            "kwargs": {"window_days": 30},
        },
        "cognitive-os-heartbeat": {
            "task": "app.tasks.cognitive_heartbeat.cognitive_heartbeat_task",
            "schedule": 300.0,  # every 5 minutes
            "args": (),
        },
        "proactive-intelligence-push": {
            "task": "app.tasks.proactive_push_beat.proactive_push_beat_task",
            "schedule": 900.0,  # every 15 minutes
            "kwargs": {
                "session_lookback_minutes": 180,
                "event_lookback_minutes": 30,
                "max_pushes_per_org": 25,
            },
        },
        **_enterprise_beat,
    },
)
