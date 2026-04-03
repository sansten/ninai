from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from tests.benchmarks._fixtures import make_text_events


def _events(n: int = 100) -> list[dict]:
    now = datetime.now(timezone.utc)
    out: list[dict] = []
    for i in range(n):
        ts = now - timedelta(hours=(i % 24))
        out.append({"id": f"t-{i}", "created_at": ts})
    return out


async def run(*, mode: str, strategy: str, dataset: str = "synthetic") -> dict[str, Any]:
    if dataset == "kaggle":
        rows = make_text_events(120, dataset="kaggle")
        counts = Counter(row["created_at"].hour for row in rows if row.get("created_at") is not None)
    else:
        rows = _events(120)
        counts = Counter(row["created_at"].hour for row in rows)
    peak_hour, peak_count = counts.most_common(1)[0]
    return {
        "benchmark": "temporal",
        "mode": mode,
        "strategy": strategy,
        "dataset": dataset,
        "peak_hour": int(peak_hour),
        "peak_count": int(peak_count),
        "samples": len(rows),
        "status": "ok",
    }
