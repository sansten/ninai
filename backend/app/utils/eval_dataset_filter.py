from __future__ import annotations

import random
import re
from typing import Iterable, Optional

from app.utils.retrieval_eval import EvalRow


def filter_rows(
    rows: Iterable[EvalRow],
    *,
    query_contains: Optional[str] = None,
    query_regex: Optional[str] = None,
) -> list[EvalRow]:
    out: list[EvalRow] = []
    rx = re.compile(query_regex, flags=re.IGNORECASE) if query_regex else None
    contains = (query_contains or "").lower().strip() or None

    for r in rows:
        q = (r.query or "").strip()
        if not q:
            continue
        if contains is not None and contains not in q.lower():
            continue
        if rx is not None and rx.search(q) is None:
            continue
        out.append(r)

    return out


def sample_rows(rows: list[EvalRow], *, max_queries: Optional[int], seed: int) -> list[EvalRow]:
    if max_queries is None:
        return rows
    n = int(max_queries)
    if n <= 0:
        return []
    if n >= len(rows):
        return rows

    rng = random.Random(int(seed))
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    pick = sorted(idx[:n])
    return [rows[i] for i in pick]
