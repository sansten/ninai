from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RelevanceFeedbackAction:
    memory_id: str
    relevant: bool


def plan_relevance_feedback(
    *,
    expected_ids: Iterable[str],
    retrieved_ids: Sequence[str],
    k: int,
    include_negative_top1: bool = True,
) -> list[RelevanceFeedbackAction]:
    """Plan relevance feedback actions from labeled eval data.

    Safety/behavior:
    - Emits at most one negative (top-1) and one positive (first relevant in top-k).
    - If expected_ids is empty, returns [].

    Intended use:
    - In staging, to simulate thumbs up/down feedback loops.
    """

    expected = {str(x) for x in (expected_ids or []) if x is not None}
    if not expected:
        return []

    if k <= 0:
        k = 1

    topk = [str(x) for x in (retrieved_ids or [])[:k] if x is not None]
    if not topk:
        return []

    actions: list[RelevanceFeedbackAction] = []

    top1 = topk[0]
    if include_negative_top1 and (top1 not in expected):
        actions.append(RelevanceFeedbackAction(memory_id=top1, relevant=False))

    first_relevant = next((mid for mid in topk if mid in expected), None)
    if first_relevant is not None:
        actions.append(RelevanceFeedbackAction(memory_id=first_relevant, relevant=True))

    # Avoid duplicates (e.g., top1 relevant -> only emit positive).
    dedup: dict[str, RelevanceFeedbackAction] = {}
    for a in actions:
        dedup[a.memory_id] = a

    # Keep stable order: negative (if any) first, then positive.
    out: list[RelevanceFeedbackAction] = []
    if include_negative_top1 and (top1 in dedup) and (dedup[top1].relevant is False):
        out.append(dedup[top1])

    if first_relevant is not None and first_relevant in dedup and dedup[first_relevant].relevant is True:
        out.append(dedup[first_relevant])

    return out
