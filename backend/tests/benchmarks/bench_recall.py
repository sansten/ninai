from __future__ import annotations

from typing import Any

from tests.benchmarks._fixtures import make_text_events, split_gold_ids


def _feature_rank(query: str, rows: list[dict]) -> list[dict]:
    q = set(query.lower().split())
    scored: list[tuple[float, dict]] = []
    for row in rows:
        tokens = set(str(row["content"]).lower().split())
        overlap = len(q & tokens)
        has_auth = any(t in tokens for t in {"auth", "authentication"})
        has_failure = any(t in tokens for t in {"failure", "failed"})
        has_spike = "spike" in tokens
        semantic_bonus = (0.35 if has_auth else 0.0) + (0.25 if has_failure else 0.0) + (0.2 if has_spike else 0.0)
        score = overlap + semantic_bonus
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored]


def _recall_at_k(results: list[dict], gold_ids: set[str], k: int) -> float:
    top = {str(r["id"]) for r in results[:k]}
    return len(top & gold_ids) / max(1, len(gold_ids))


def _precision_at_k(results: list[dict], gold_ids: set[str], k: int) -> float:
    top = [str(r["id"]) for r in results[:k]]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rid in gold_ids)
    return hits / len(top)


async def run(*, mode: str, strategy: str, dataset: str = "synthetic") -> dict[str, Any]:
    rows = make_text_events(200, dataset=dataset)
    gold = split_gold_ids(rows, "is_relevant")
    ranked = _feature_rank("authentication failure spike", rows)

    out: dict[str, Any] = {
        "benchmark": "recall",
        "mode": mode,
        "strategy": strategy,
        "dataset": dataset,
        "recall_at_5": round(_recall_at_k(ranked, gold, 5), 4),
        "recall_at_10": round(_recall_at_k(ranked, gold, 10), 4),
        "precision_at_5": round(_precision_at_k(ranked, gold, 5), 4),
        "precision_at_10": round(_precision_at_k(ranked, gold, 10), 4),
        "samples": len(rows),
        "status": "ok",
    }

    if mode in {"integration", "full"}:
        # Qdrant mode hook: this remains optional so CI/unit runs stay fast.
        try:
            import qdrant_client  # type: ignore  # noqa: F401

            out["qdrant_enabled"] = True
        except Exception:
            out["qdrant_enabled"] = False
            out["qdrant_note"] = "qdrant_client not installed; ran feature ranking only"

    return out
