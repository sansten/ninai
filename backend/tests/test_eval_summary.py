from __future__ import annotations

import json

from app.utils.eval_summary import EvalRunSummary, write_summary_csv, write_summary_json


def test_write_summary_json(tmp_path):
    p = tmp_path / "out.json"
    summaries = [
        EvalRunSummary(mode="balanced", kind="simple", payload={"queries": 2, "recall_at_k": 0.5, "mrr": 0.25}),
    ]
    write_summary_json(str(p), summaries=summaries, meta={"x": 1})
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["meta"]["x"] == 1
    assert len(obj["summaries"]) == 1
    assert obj["summaries"][0]["mode"] == "balanced"


def test_write_summary_csv(tmp_path):
    p = tmp_path / "out.csv"
    summaries = [
        EvalRunSummary(mode="balanced", kind="simple", payload={"queries": 2, "recall_at_k": 0.5, "mrr": 0.25}),
        EvalRunSummary(
            mode="performance",
            kind="holdout",
            payload={
                "train_queries": 3,
                "test_queries": 1,
                "baseline": {"recall_at_k": 0.0, "mrr": 0.0},
                "iterations": [{"post_feedback": {"recall_at_k": 1.0, "mrr": 1.0}}],
                "dry_run": True,
            },
        ),
    ]
    write_summary_csv(str(p), summaries=summaries)
    txt = p.read_text(encoding="utf-8")
    assert "mode,kind" in txt
    assert "balanced" in txt
    assert "performance" in txt
