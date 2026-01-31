from __future__ import annotations

from app.utils.retrieval_eval import EvalRow, aggregate_metrics, mrr, recall_at_k


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], ["b"], 1) == 0.0
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b", "c"], ["x"], 3) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert mrr(["a", "b", "c"], ["c"]) == 1.0 / 3.0
    assert mrr(["a", "b", "c"], ["x"]) == 0.0


def test_aggregate_metrics():
    rows = [
        EvalRow(query="q1", expected_ids=["b"]),
        EvalRow(query="q2", expected_ids=["x"]),
    ]
    retrieved = [
        ["a", "b"],
        ["x", "y"],
    ]
    metrics = aggregate_metrics(rows, retrieved, k=1)
    assert metrics.queries == 2
    assert metrics.recall_at_k == 0.5
    assert metrics.mrr == (0.5 + 1.0) / 2.0
