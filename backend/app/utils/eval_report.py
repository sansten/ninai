from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class EvalReportRow:
    mode: str
    query: str
    expected_ids: list[str]
    retrieved_ids: list[str]
    phase: str = "default"
    ranking_meta: Optional[dict[str, Any]] = None


def _as_list_str(val: Any) -> list[str]:
    if isinstance(val, str):
        return [val]
    if not isinstance(val, list):
        return []
    return [str(x) for x in val if x is not None]


def load_eval_report_jsonl(path: str) -> list[EvalReportRow]:
    """Load a JSONL report previously produced by eval_search_ranking.py."""

    rows: list[EvalReportRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue

            mode = str(obj.get("mode", "")).strip() or "balanced"
            query = str(obj.get("query", "")).strip()
            if not query:
                # skip malformed row
                continue

            phase = str(obj.get("phase", "default") or "default").strip() or "default"

            expected_ids = _as_list_str(obj.get("expected_ids"))
            retrieved_ids = _as_list_str(obj.get("retrieved_ids"))
            ranking_meta = obj.get("ranking_meta")
            if not isinstance(ranking_meta, dict):
                ranking_meta = None

            rows.append(
                EvalReportRow(
                    mode=mode,
                    query=query,
                    expected_ids=expected_ids,
                    retrieved_ids=retrieved_ids,
                    phase=phase,
                    ranking_meta=ranking_meta,
                )
            )

    return rows
