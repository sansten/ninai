from __future__ import annotations

from typing import Any

from tests.benchmarks._fixtures import make_text_events


def _credibility_score(content: str) -> float:
    txt = content.lower()
    base = 0.65
    if "auth" in txt:
        base += 0.15
    if "queue" in txt:
        base -= 0.1
    return max(0.0, min(1.0, round(base, 4)))


async def run(*, mode: str, strategy: str, dataset: str = "synthetic") -> dict[str, Any]:
    rows = make_text_events(100, dataset=dataset)
    scores = [_credibility_score(str(r["content"])) for r in rows]
    high = sum(1 for s in scores if s >= 0.75)
    return {
        "benchmark": "credibility",
        "mode": mode,
        "strategy": strategy,
        "dataset": dataset,
        "avg_score": round(sum(scores) / len(scores), 4),
        "high_score_rate": round(high / len(scores), 4),
        "samples": len(rows),
        "status": "ok",
    }
