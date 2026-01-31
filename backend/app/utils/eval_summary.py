from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class EvalRunSummary:
    """Structured per-mode summary for eval_search_ranking.py."""

    mode: str
    kind: str  # simple | holdout
    payload: dict[str, Any]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_summary_json(path: str, *, summaries: Iterable[EvalRunSummary], meta: Optional[dict[str, Any]] = None) -> None:
    out = {
        "generated_at": _utc_iso(),
        "meta": meta or {},
        "summaries": [
            {"mode": s.mode, "kind": s.kind, **s.payload}
            for s in summaries
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _flatten(summary: EvalRunSummary) -> dict[str, Any]:
    row: dict[str, Any] = {"mode": summary.mode, "kind": summary.kind}
    payload = summary.payload

    if summary.kind == "simple":
        # payload: {queries, recall_at_k, mrr, ranking_meta, ...}
        row["queries"] = payload.get("queries")
        row["recall_at_k"] = payload.get("recall_at_k")
        row["mrr"] = payload.get("mrr")
        return row

    if summary.kind == "holdout":
        baseline = payload.get("baseline") or {}
        row["train_queries"] = payload.get("train_queries")
        row["test_queries"] = payload.get("test_queries")
        row["baseline_recall_at_k"] = baseline.get("recall_at_k")
        row["baseline_mrr"] = baseline.get("mrr")

        iterations = payload.get("iterations") or []
        if isinstance(iterations, list) and iterations:
            last = iterations[-1]
            post = (last.get("post_feedback") or {}) if isinstance(last, dict) else {}
            row["post_recall_at_k"] = post.get("recall_at_k")
            row["post_mrr"] = post.get("mrr")
            row["iterations"] = len(iterations)
        else:
            row["post_recall_at_k"] = None
            row["post_mrr"] = None
            row["iterations"] = 0

        row["stopped_early"] = payload.get("stopped_early")
        row["stopped_on_regression"] = payload.get("stopped_on_regression")
        row["stopped_iteration"] = payload.get("stopped_iteration")
        row["dry_run"] = payload.get("dry_run")
        return row

    # Fallback: best-effort
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[str(k)] = v
    return row


def write_summary_csv(path: str, *, summaries: Iterable[EvalRunSummary]) -> None:
    rows = [_flatten(s) for s in summaries]
    if not rows:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("mode,kind\n")
        return

    # stable, union header
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
