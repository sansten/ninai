from __future__ import annotations

import argparse

from app.utils.failures_to_dataset import (
    failures_to_focused_dataset,
    load_failures_jsonl,
    write_focused_dataset_jsonl,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Convert eval failures JSONL into a focused dataset JSONL.")
    p.add_argument("--failures-jsonl", required=True, help="Path produced by eval_search_ranking.py --out-failures-jsonl")
    p.add_argument("--out-dataset-jsonl", required=True, help="Output dataset JSONL (one row per query)")
    p.add_argument("--mode", default=None, help="Optional: only include failures from this mode")
    p.add_argument("--phase", default=None, help="Optional: only include failures from this phase")
    p.add_argument("--max-queries", type=int, default=None, help="Optional: cap number of queries (highest misses first)")
    args = p.parse_args()

    failures = load_failures_jsonl(str(args.failures_jsonl))
    focused = failures_to_focused_dataset(
        failures,
        mode=(str(args.mode) if args.mode else None),
        phase=(str(args.phase) if args.phase else None),
        max_queries=args.max_queries,
    )
    write_focused_dataset_jsonl(str(args.out_dataset_jsonl), focused)
    print(f"Wrote {len(focused)} queries to {args.out_dataset_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
