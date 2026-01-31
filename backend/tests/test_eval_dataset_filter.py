from __future__ import annotations

from app.utils.eval_dataset_filter import filter_rows, sample_rows
from app.utils.retrieval_eval import EvalRow


def test_filter_rows_contains_and_regex():
    rows = [
        EvalRow(query="Alpha one", expected_ids=["1"]),
        EvalRow(query="beta two", expected_ids=["2"]),
        EvalRow(query="Gamma three", expected_ids=["3"]),
    ]

    out = filter_rows(rows, query_contains="beta")
    assert [r.query for r in out] == ["beta two"]

    out = filter_rows(rows, query_regex=r"gamma")
    assert [r.query for r in out] == ["Gamma three"]


def test_sample_rows_is_deterministic():
    rows = [EvalRow(query=f"q{i}", expected_ids=[str(i)]) for i in range(10)]
    a = sample_rows(rows, max_queries=3, seed=123)
    b = sample_rows(rows, max_queries=3, seed=123)
    assert [r.query for r in a] == [r.query for r in b]


def test_sample_rows_handles_limits():
    rows = [EvalRow(query=f"q{i}", expected_ids=[str(i)]) for i in range(5)]
    assert sample_rows(rows, max_queries=None, seed=1) == rows
    assert sample_rows(rows, max_queries=0, seed=1) == []
    assert sample_rows(rows, max_queries=10, seed=1) == rows
