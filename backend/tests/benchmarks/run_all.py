from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from tests.benchmarks import (
    bench_conflict,
    bench_credibility,
    bench_goal,
    bench_latency,
    bench_recall,
    bench_temporal,
)


BENCHMARKS = [
    bench_conflict.run,
    bench_credibility.run,
    bench_goal.run,
    bench_latency.run,
    bench_recall.run,
    bench_temporal.run,
]


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nBenchmark Summary")
    print("-" * 96)
    print(f"{'benchmark':<14} {'mode':<12} {'strategy':<10} {'status':<8} metrics")
    print("-" * 96)
    for row in rows:
        metric_keys = [k for k in row.keys() if k not in {"benchmark", "mode", "strategy", "status"}]
        metric_str = ", ".join(f"{k}={row[k]}" for k in metric_keys)
        print(f"{row['benchmark']:<14} {row['mode']:<12} {row['strategy']:<10} {row['status']:<8} {metric_str}")


def _aggregate_runs(all_runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge N benchmark runs into one row per benchmark with mean ± stddev for numeric fields."""
    if len(all_runs) == 1:
        return all_runs[0]

    # Group by benchmark name
    by_bench: dict[str, list[dict]] = {}
    for run in all_runs:
        for row in run:
            by_bench.setdefault(row["benchmark"], []).append(row)

    aggregated: list[dict[str, Any]] = []
    for bench_name, rows in by_bench.items():
        merged: dict[str, Any] = {
            "benchmark": bench_name,
            "mode": rows[0].get("mode", ""),
            "strategy": rows[0].get("strategy", ""),
            "status": "ok" if all(r.get("status") == "ok" for r in rows) else "error",
            "runs": len(rows),
        }
        # Aggregate numeric fields
        all_keys = {k for row in rows for k in row if k not in {"benchmark", "mode", "strategy", "status", "runs", "domain_confusion"}}
        for k in sorted(all_keys):
            vals = [row[k] for row in rows if isinstance(row.get(k), (int, float))]
            if not vals:
                merged[k] = rows[0].get(k)
                continue
            mean_val = sum(vals) / len(vals)
            if len(vals) > 1:
                variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
                std_val = math.sqrt(variance)
                merged[f"{k}_mean"] = round(mean_val, 4)
                merged[f"{k}_std"] = round(std_val, 4)
            else:
                merged[k] = round(mean_val, 4)
        aggregated.append(merged)
    return aggregated


def _composite_score(rows: list[dict[str, Any]], strategy: str) -> float:
    """Compute quality × reliability composite score (0–1).

    quality  = mean of goal accuracy, conflict f1, recall@10 (where available)
    reliability = llm_success_rate for llm strategy; 1.0 for heuristic
    """
    quality_vals: list[float] = []
    reliability = 1.0

    for row in rows:
        bname = row.get("benchmark", "")
        # Use _mean suffix if multi-run aggregation produced it, else raw value
        def _get(key: str) -> float | None:
            v = row.get(f"{key}_mean", row.get(key))
            return float(v) if isinstance(v, (int, float)) else None

        if bname == "goal":
            v = _get("accuracy")
            if v is not None:
                quality_vals.append(v)
            if strategy == "llm":
                lr = _get("llm_success_rate")
                if lr is not None:
                    reliability = lr
        elif bname == "conflict":
            v = _get("f1")
            if v is not None:
                quality_vals.append(v)
        elif bname == "recall":
            v = _get("recall_at_10") or _get("Recall@10")
            if v is not None:
                quality_vals.append(v)

    quality = (sum(quality_vals) / len(quality_vals)) if quality_vals else 0.0
    return round(quality * reliability, 4)


async def _run(mode: str, strategy: str, dataset: str) -> list[dict[str, Any]]:
    os.environ["AGENT_STRATEGY"] = strategy
    try:
        settings.AGENT_STRATEGY = strategy
    except Exception:
        pass
    results: list[dict[str, Any]] = []

    for runner in BENCHMARKS:
        try:
            results.append(await runner(mode=mode, strategy=strategy, dataset=dataset))
        except Exception as exc:  # pragma: no cover - defensive benchmarking guard
            results.append(
                {
                    "benchmark": runner.__module__.split(".")[-1].replace("bench_", ""),
                    "mode": mode,
                    "strategy": strategy,
                    "dataset": dataset,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ninai benchmark suite")
    parser.add_argument("--mode", default="unit", choices=["unit", "integration", "full"])
    parser.add_argument("--strategy", default="heuristic", choices=["heuristic", "llm"])
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "kaggle"])
    parser.add_argument("--ollama-model", default=None, help="Optional override for OLLAMA_MODEL_AGENTS")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs (≥2 adds mean±stddev)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    args = parser.parse_args()

    if args.ollama_model:
        os.environ["OLLAMA_MODEL_AGENTS"] = str(args.ollama_model)
        try:
            settings.OLLAMA_MODEL_AGENTS = str(args.ollama_model)
        except Exception:
            pass

    num_runs = max(1, args.runs)
    started = datetime.now(timezone.utc)
    all_runs: list[list[dict[str, Any]]] = []
    for _ in range(num_runs):
        all_runs.append(asyncio.run(_run(mode=args.mode, strategy=args.strategy, dataset=args.dataset)))
    finished = datetime.now(timezone.utc)

    rows = _aggregate_runs(all_runs)
    composite = _composite_score(rows, args.strategy)

    _print_table(rows)
    print(f"\nComposite score (quality × reliability): {composite}")

    summary = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "mode": args.mode,
        "strategy": args.strategy,
        "dataset": args.dataset,
        "runs": num_runs,
        "composite_score": composite,
        "results": rows,
    }

    if args.json:
        print("\nJSON Summary")
        print(json.dumps(summary, indent=2))

    has_error = any(row.get("status") == "error" for row in rows)
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
