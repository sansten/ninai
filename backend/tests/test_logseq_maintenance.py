from __future__ import annotations

from app.tasks.maintenance import nightly_logseq_export_task


def test_nightly_logseq_export_task_noops_with_memory_broker():
    # In unit tests Celery defaults to memory:// broker; task should be a safe no-op.
    res = nightly_logseq_export_task.run(batch_size=10, lookback_hours=1)
    assert isinstance(res, dict)
    assert res.get("ok") is True
    assert res.get("skipped") is True
    assert res.get("reason") == "broker_disabled"
