from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Recall@k for a single query.

    Returns 1.0 if any relevant id appears in top-k retrieved, else 0.0.
    """

    if k <= 0:
        return 0.0

    topk = [str(x) for x in (retrieved_ids or [])[:k]]
    relevant = {str(x) for x in (relevant_ids or [])}
    if not relevant:
        return 0.0

    return 1.0 if any(x in relevant for x in topk) else 0.0


def mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    """Mean Reciprocal Rank for a single query.

    If the first relevant result is at rank r (1-indexed), returns 1/r, else 0.
    """

    relevant = {str(x) for x in (relevant_ids or [])}
    if not relevant:
        return 0.0

    for idx, rid in enumerate(retrieved_ids or []):
        if str(rid) in relevant:
            return 1.0 / float(idx + 1)
    return 0.0


@dataclass(frozen=True)
class EvalRow:
    query: str
    expected_ids: list[str]


@dataclass(frozen=True)
class EvalMetrics:
    queries: int
    recall_at_k: float
    mrr: float


def aggregate_metrics(
    rows: Sequence[EvalRow],
    retrieved_ids_per_row: Sequence[Sequence[str]],
    *,
    k: int,
) -> EvalMetrics:
    if len(rows) != len(retrieved_ids_per_row):
        raise ValueError("rows and retrieved_ids_per_row must be the same length")

    n = len(rows)
    if n == 0:
        return EvalMetrics(queries=0, recall_at_k=0.0, mrr=0.0)

    recalls = []
    rrs = []
    for row, retrieved in zip(rows, retrieved_ids_per_row, strict=True):
        recalls.append(recall_at_k(list(retrieved), row.expected_ids, k))
        rrs.append(mrr(list(retrieved), row.expected_ids))

    return EvalMetrics(
        queries=n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
    )
