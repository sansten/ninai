from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class FailureRow:
    source: str
    mode: str
    phase: str
    query: str
    expected_ids: list[str]
    retrieved_ids: list[str]


def _as_list_str(val: Any) -> list[str]:
    if isinstance(val, str):
        return [val]
    if not isinstance(val, list):
        return []
    return [str(x) for x in val if x is not None]


def load_failures_jsonl(path: str) -> list[FailureRow]:
    rows: list[FailureRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            query = str(obj.get("query", "")).strip()
            if not query:
                continue
            rows.append(
                FailureRow(
                    source=str(obj.get("source", "dataset") or "dataset"),
                    mode=str(obj.get("mode", "balanced") or "balanced"),
                    phase=str(obj.get("phase", "default") or "default"),
                    query=query,
                    expected_ids=_as_list_str(obj.get("expected_ids")),
                    retrieved_ids=_as_list_str(obj.get("retrieved_ids")),
                )
            )
    return rows


def failures_to_focused_dataset(
    failures: Iterable[FailureRow],
    *,
    mode: Optional[str] = None,
    phase: Optional[str] = None,
    max_queries: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Aggregate failures into a focused dataset.

    Output rows look like:
      {"query": str, "expected_ids": [..], "misses": int, "modes": [...], "phases": [...]}.

    Sorting: highest misses first, then query asc.
    """

    buckets: dict[str, dict[str, Any]] = {}
    for f in failures:
        if mode is not None and f.mode != mode:
            continue
        if phase is not None and f.phase != phase:
            continue

        b = buckets.get(f.query)
        if b is None:
            b = {
                "query": f.query,
                "expected_ids": set(),
                "misses": 0,
                "modes": set(),
                "phases": set(),
            }
            buckets[f.query] = b

        b["expected_ids"].update(f.expected_ids)
        b["misses"] += 1
        b["modes"].add(f.mode)
        b["phases"].add(f.phase)

    rows: list[dict[str, Any]] = []
    for b in buckets.values():
        rows.append(
            {
                "query": b["query"],
                "expected_ids": sorted(b["expected_ids"]),
                "misses": int(b["misses"]),
                "modes": sorted(b["modes"]),
                "phases": sorted(b["phases"]),
            }
        )

    rows.sort(key=lambda r: (-int(r.get("misses", 0)), str(r.get("query", ""))))

    if max_queries is not None:
        n = max(0, int(max_queries))
        rows = rows[:n]

    return rows


def write_focused_dataset_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"query": r["query"], "expected_ids": r["expected_ids"]}) + "\n")
