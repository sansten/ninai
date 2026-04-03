from __future__ import annotations

import time
from statistics import mean
from typing import Any

from tests.benchmarks._fixtures import make_text_events


async def run(*, mode: str, strategy: str, dataset: str = "synthetic") -> dict[str, Any]:
    rows = make_text_events(200, dataset=dataset)
    timings: list[float] = []
    for row in rows:
        start = time.perf_counter()
        _ = str(row["content"]).split()
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "benchmark": "latency",
        "mode": mode,
        "strategy": strategy,
        "dataset": dataset,
        "p50_ms": round(sorted(timings)[len(timings) // 2], 4),
        "avg_ms": round(mean(timings), 4),
        "samples": len(rows),
        "status": "ok",
    }
