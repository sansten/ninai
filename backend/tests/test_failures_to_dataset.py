from __future__ import annotations

import json

from app.utils.failures_to_dataset import failures_to_focused_dataset, load_failures_jsonl


def test_failures_to_focused_dataset_aggregates(tmp_path):
    p = tmp_path / "failures.jsonl"
    lines = [
        {
            "source": "dataset",
            "mode": "balanced",
            "phase": "default",
            "query": "q1",
            "expected_ids": ["a"],
            "retrieved_ids": ["x"],
        },
        {
            "source": "dataset",
            "mode": "balanced",
            "phase": "default",
            "query": "q1",
            "expected_ids": ["b"],
            "retrieved_ids": ["y"],
        },
        {
            "source": "holdout",
            "mode": "performance",
            "phase": "baseline",
            "query": "q2",
            "expected_ids": ["c"],
            "retrieved_ids": ["z"],
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    failures = load_failures_jsonl(str(p))
    focused = failures_to_focused_dataset(failures)
    assert focused[0]["query"] == "q1"
    assert focused[0]["misses"] == 2
    assert focused[0]["expected_ids"] == ["a", "b"]

    focused_bal = failures_to_focused_dataset(failures, mode="balanced")
    assert len(focused_bal) == 1
    assert focused_bal[0]["query"] == "q1"
