from __future__ import annotations

import pytest

from app.core.celery_app import celery_app
from app.tasks.proactive_push_beat import proactive_push_beat_task


def test_celery_task_route_registered():
    route = celery_app.conf.task_routes.get("app.tasks.proactive_push_beat.proactive_push_beat_task")
    assert route is not None
    assert route.get("queue") == "q.cognitive_loop"


def test_celery_beat_schedule_registered():
    beat_cfg = celery_app.conf.beat_schedule.get("proactive-intelligence-push")
    assert beat_cfg is not None
    assert beat_cfg.get("task") == "app.tasks.proactive_push_beat.proactive_push_beat_task"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"session_lookback_minutes": 120, "event_lookback_minutes": 20, "max_pushes_per_org": 10},
    ],
)
def test_task_entrypoint_invokes_runner(monkeypatch, kwargs):
    expected = {"ok": True, "pushed": 3}

    def _fake_run_async(_coro):
        _coro.close()
        return expected

    monkeypatch.setattr("app.tasks.proactive_push_beat._run_async", _fake_run_async)
    result = proactive_push_beat_task(**kwargs)
    assert result == expected
