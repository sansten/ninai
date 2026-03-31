"""Tests for environment sync Prometheus metric helpers."""

from __future__ import annotations

from types import SimpleNamespace

from app.middleware.prometheus import (
    env_sync_diverged_objects,
    env_sync_lag_seconds,
    env_sync_reconcile_candidates_total,
    env_sync_reconcile_dlq_handoffs_total,
    env_sync_reconcile_runs_total,
    record_environment_reconcile_run,
    update_environment_sync_connector_metrics,
)


def _counter_value(counter, **labels) -> float:
    return float(counter.labels(**labels)._value.get())


def _gauge_value(gauge, **labels) -> float:
    return float(gauge.labels(**labels)._value.get())


def test_update_environment_sync_connector_metrics_sets_gauges():
    summary = SimpleNamespace(
        connector_type="webhook",
        max_lag_seconds=12.5,
        diverged_objects=3,
    )

    update_environment_sync_connector_metrics("org-1", [summary])

    assert _gauge_value(env_sync_lag_seconds, org_id="org-1", connector_type="webhook") == 12.5
    assert _gauge_value(env_sync_diverged_objects, org_id="org-1", connector_type="webhook") == 3.0


def test_record_environment_reconcile_run_increments_counters():
    run_before = _counter_value(env_sync_reconcile_runs_total, status="partial_failure")
    ok_before = _counter_value(env_sync_reconcile_candidates_total, status="succeeded")
    fail_before = _counter_value(env_sync_reconcile_candidates_total, status="failed")
    dlq_before = _counter_value(env_sync_reconcile_dlq_handoffs_total, reason="dispatch_failure")

    record_environment_reconcile_run(
        run_status="partial_failure",
        succeeded=2,
        failed=1,
        dlq_handoffs=1,
    )

    assert _counter_value(env_sync_reconcile_runs_total, status="partial_failure") == run_before + 1
    assert _counter_value(env_sync_reconcile_candidates_total, status="succeeded") == ok_before + 2
    assert _counter_value(env_sync_reconcile_candidates_total, status="failed") == fail_before + 1
    assert _counter_value(env_sync_reconcile_dlq_handoffs_total, reason="dispatch_failure") == dlq_before + 1
