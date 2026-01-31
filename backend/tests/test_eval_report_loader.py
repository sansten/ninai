from __future__ import annotations

import json

from app.utils.eval_report import load_eval_report_jsonl


def test_load_eval_report_jsonl(tmp_path):
    p = tmp_path / "report.jsonl"
    lines = [
        {
            "mode": "performance",
            "query": "hello",
            "expected_ids": ["e1"],
            "retrieved_ids": ["r1", "r2"],
            "phase": "baseline",
            "ranking_meta": {"hnms_mode_effective": "performance"},
        },
        {
            "mode": "research",
            "query": "world",
            "expected_ids": "e2",
            "retrieved_ids": ["r3"],
            "ranking_meta": None,
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")

    rows = load_eval_report_jsonl(str(p))
    assert len(rows) == 2
    assert rows[0].mode == "performance"
    assert rows[0].phase == "baseline"
    assert rows[0].query == "hello"
    assert rows[0].expected_ids == ["e1"]
    assert rows[0].retrieved_ids == ["r1", "r2"]
    assert rows[0].ranking_meta and rows[0].ranking_meta["hnms_mode_effective"] == "performance"

    assert rows[1].mode == "research"
    assert rows[1].expected_ids == ["e2"]
    assert rows[1].ranking_meta is None
