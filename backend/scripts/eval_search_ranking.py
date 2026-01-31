from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from app.utils.retrieval_eval import EvalRow, aggregate_metrics
from app.utils.retrieval_feedback import plan_relevance_feedback
from app.utils.eval_report import load_eval_report_jsonl
from app.utils.eval_split import split_deterministic
from app.utils.eval_early_stop import EarlyStopConfig, update_plateau_streak
from app.utils.eval_regression_stop import RegressionStopConfig, should_stop_on_regression
from app.utils.eval_summary import EvalRunSummary, write_summary_csv, write_summary_json
from app.utils.run_meta import collect_run_meta
from app.utils.eval_dataset_filter import filter_rows, sample_rows


def _load_dataset(path: str) -> list[EvalRow]:
    rows: list[EvalRow] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(_row_from_obj(obj))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        for x in obj:
            rows.append(_row_from_obj(x))
    else:
        raise ValueError("dataset must be a JSON array or JSONL")

    return rows


def _row_from_obj(obj: Any) -> EvalRow:
    if not isinstance(obj, dict):
        raise ValueError("dataset row must be an object")

    query = str(obj.get("query", "")).strip()
    if not query:
        raise ValueError("dataset row missing 'query'")

    expected = obj.get("expected_ids") or obj.get("expected_id") or obj.get("expected")
    if isinstance(expected, str):
        expected_ids = [expected]
    elif isinstance(expected, list):
        expected_ids = [str(x) for x in expected if x is not None]
    else:
        expected_ids = []

    return EvalRow(query=query, expected_ids=expected_ids)


def _search(
    client: httpx.Client,
    *,
    base_url: str,
    query: str,
    limit: int,
    hnms_mode: str | None,
) -> tuple[list[str], dict[str, Any] | None]:
    params: dict[str, Any] = {"query": query, "limit": limit}
    if hnms_mode:
        params["hnms_mode"] = hnms_mode

    resp = client.get(f"{base_url.rstrip('/')}/api/v1/memories/search", params=params)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []
    ranking_meta = data.get("ranking_meta")
    if not isinstance(ranking_meta, dict):
        ranking_meta = None
    ids: list[str] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        mid = r.get("id")
        if mid is None:
            continue
        ids.append(str(mid))
    return ids, ranking_meta


def _submit_relevance(
    client: httpx.Client,
    *,
    base_url: str,
    memory_id: str,
    relevant: bool,
    query: str,
    hnms_mode: str,
    trace_id: str,
    target_agent: str,
) -> None:
    payload: dict[str, Any] = {
        "relevant": bool(relevant),
        "query": query,
        "hnms_mode": hnms_mode,
        "trace_id": trace_id,
        "target_agent": target_agent,
    }
    resp = client.post(
        f"{base_url.rstrip('/')}/api/v1/memories/{memory_id}/relevance",
        json=payload,
    )
    resp.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate search ranking modes/decay via the API.")
    parser.add_argument("--dataset", required=False, help="Path to dataset (.json or .jsonl)")
    parser.add_argument(
        "--resume-from-jsonl",
        default=None,
        help="Path to a prior --out-jsonl report to replay metrics/feedback deterministically (no re-search).",
    )
    parser.add_argument(
        "--resume-phase",
        default=None,
        help="When resuming from JSONL that contains phases, pick one (e.g. baseline, post_feedback). Default: auto.",
    )
    parser.add_argument(
        "--query-contains",
        default=None,
        help="Filter dataset/resume rows to queries containing this substring (case-insensitive).",
    )
    parser.add_argument(
        "--query-regex",
        default=None,
        help="Filter dataset/resume rows to queries matching this regex (case-insensitive).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="After filtering, cap the number of queries (deterministic sampling).",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic query sampling when --max-queries is set (default 42).",
    )

    parser.add_argument(
        "--holdout-eval",
        action="store_true",
        help="Run a holdout evaluation: baseline on test split, apply feedback on train split, then re-evaluate test.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="For --holdout-eval, number of feedback/eval iterations to run (default 1).",
    )
    parser.add_argument(
        "--early-stop-mrr-delta",
        type=float,
        default=None,
        help="For --holdout-eval, stop if iteration-to-iteration MRR delta is below this threshold for --early-stop-patience iterations.",
    )
    parser.add_argument(
        "--early-stop-recall-delta",
        type=float,
        default=None,
        help="For --holdout-eval, stop if iteration-to-iteration recall@k delta is below this threshold for --early-stop-patience iterations.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="For --holdout-eval, number of consecutive plateau iterations before stopping (default 2).",
    )
    parser.add_argument(
        "--stop-on-regression-mrr",
        type=float,
        default=None,
        help="For --holdout-eval, abort if iteration-to-iteration MRR drops by at least this amount.",
    )
    parser.add_argument(
        "--stop-on-regression-recall",
        type=float,
        default=None,
        help="For --holdout-eval, abort if iteration-to-iteration recall@k drops by at least this amount.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Train fraction for --holdout-eval (default 0.7).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic train/test split (default 42).",
    )
    parser.add_argument("--base-url", default=os.getenv("EVAL_API_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--token", default=os.getenv("EVAL_AUTH_TOKEN"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional path to write per-query results as JSONL (one line per query per mode)",
    )
    parser.add_argument(
        "--out-failures-jsonl",
        default=None,
        help="Optional path to write hard cases (misses where expected_ids are not in top-k).",
    )
    parser.add_argument(
        "--out-summary-json",
        default=None,
        help="Optional path to write run summary as JSON (aggregated per-mode metrics).",
    )
    parser.add_argument(
        "--out-summary-csv",
        default=None,
        help="Optional path to write run summary as CSV (flattened per-mode metrics).",
    )
    parser.add_argument(
        "--include-run-meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include reproducibility metadata in JSON summary (git SHA, dataset hash, host, python).",
    )
    parser.add_argument(
        "--apply-feedback",
        action="store_true",
        help="If set, generate and submit relevance feedback (thumbs up/down) from dataset labels.",
    )
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Skip metrics and only apply planned relevance feedback (requires --apply-feedback).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When applying feedback, keep dry-run on by default. Use --no-dry-run to actually POST feedback.",
    )
    parser.add_argument(
        "--feedback-k",
        type=int,
        default=10,
        help="Top-k window used to plan relevance feedback (independent of metric k).",
    )
    parser.add_argument(
        "--feedback-target-agent",
        default="eval_harness",
        help="target_agent value stored with feedback.",
    )
    parser.add_argument(
        "--feedback-max-total",
        type=int,
        default=200,
        help="Max number of feedback POSTs allowed across the whole run (safety cap).",
    )
    parser.add_argument(
        "--feedback-max-per-query",
        type=int,
        default=2,
        help="Max number of feedback POSTs allowed per query (safety cap).",
    )
    parser.add_argument(
        "--feedback-sleep-ms",
        type=int,
        default=50,
        help="Sleep this many ms between feedback POSTs (rate limit).",
    )
    parser.add_argument(
        "--modes",
        default="balanced,performance,research",
        help="Comma-separated modes to test (use empty string for no mode param)",
    )
    args = parser.parse_args()

    if args.feedback_only and not args.apply_feedback:
        print("--feedback-only requires --apply-feedback")
        return 2

    if args.holdout_eval and args.resume_from_jsonl is not None:
        print("--holdout-eval is not supported with --resume-from-jsonl (needs a second post-feedback search run)")
        return 2

    if args.holdout_eval and not args.apply_feedback:
        print("--holdout-eval requires --apply-feedback (to apply training feedback between evaluations)")
        return 2

    if args.holdout_eval and int(args.iterations or 0) < 1:
        print("--iterations must be >= 1")
        return 2

    if not args.dataset and not args.resume_from_jsonl:
        print("Missing dataset. Provide --dataset or --resume-from-jsonl.")
        return 2

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]

    needs_api_reads = args.resume_from_jsonl is None
    needs_api_writes = bool(args.apply_feedback and not args.dry_run)
    if (needs_api_reads or needs_api_writes) and not args.token:
        print("Missing auth token. Set EVAL_AUTH_TOKEN or pass --token.")
        return 2

    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    out_fp = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    failures_fp = open(args.out_failures_jsonl, "w", encoding="utf-8") if args.out_failures_jsonl else None
    feedback_sent_total = 0
    summaries: list[EvalRunSummary] = []

    try:
        with httpx.Client(headers=headers, timeout=30.0) as client:
            if args.resume_from_jsonl is not None:
                report = load_eval_report_jsonl(str(args.resume_from_jsonl))
                # Apply query filtering/sampling at resume level.
                resume_rows = [EvalRow(query=r.query, expected_ids=list(r.expected_ids)) for r in report]
                resume_rows = filter_rows(resume_rows, query_contains=args.query_contains, query_regex=args.query_regex)
                resume_rows = sample_rows(resume_rows, max_queries=args.max_queries, seed=int(args.sample_seed))
                keep_queries = {r.query for r in resume_rows}
                report = [r for r in report if r.query in keep_queries]
                for mode in modes:
                    mode_rows = [r for r in report if r.mode == mode]
                    phases = {r.phase for r in mode_rows}
                    if args.resume_phase:
                        mode_rows = [r for r in mode_rows if r.phase == str(args.resume_phase)]
                    elif len(phases) > 1 and "post_feedback" in phases:
                        # Prefer post-feedback when resuming from holdout reports.
                        mode_rows = [r for r in mode_rows if r.phase == "post_feedback"]
                    if not mode_rows:
                        continue

                    if args.apply_feedback:
                        for r in mode_rows:
                            actions = plan_relevance_feedback(
                                expected_ids=r.expected_ids,
                                retrieved_ids=r.retrieved_ids,
                                k=args.feedback_k,
                                include_negative_top1=True,
                            )
                            actions = list(actions)[: max(0, int(args.feedback_max_per_query or 0))]
                            if not args.dry_run:
                                for a in actions:
                                    if int(args.feedback_max_total or 0) > 0 and feedback_sent_total >= int(
                                        args.feedback_max_total
                                    ):
                                        break
                                    _submit_relevance(
                                        client,
                                        base_url=args.base_url,
                                        memory_id=a.memory_id,
                                        relevant=a.relevant,
                                        query=r.query,
                                        hnms_mode=mode,
                                        trace_id=f"eval:{mode}",
                                        target_agent=str(args.feedback_target_agent),
                                    )
                                    feedback_sent_total += 1
                                    sleep_ms = int(args.feedback_sleep_ms or 0)
                                    if sleep_ms > 0:
                                        time.sleep(sleep_ms / 1000.0)

                    if args.feedback_only:
                        continue

                    retrieved_per_row = [list(r.retrieved_ids) for r in mode_rows]
                    eval_rows = [EvalRow(query=r.query, expected_ids=list(r.expected_ids)) for r in mode_rows]
                    metrics = aggregate_metrics(eval_rows, retrieved_per_row, k=args.k)
                    if failures_fp is not None:
                        for r, retrieved in zip(eval_rows, retrieved_per_row):
                            exp = set(r.expected_ids)
                            topk = set(retrieved[: args.k])
                            if exp and exp.isdisjoint(topk):
                                failures_fp.write(
                                    json.dumps(
                                        {
                                            "source": "resume",
                                            "mode": mode,
                                            "phase": next((x.phase for x in mode_rows if x.query == r.query), "default"),
                                            "query": r.query,
                                            "expected_ids": r.expected_ids,
                                            "retrieved_ids": retrieved[: args.k],
                                        }
                                    )
                                    + "\n"
                                )
                    meta_first = next((r.ranking_meta for r in mode_rows if isinstance(r.ranking_meta, dict)), None)
                    out = {"mode": mode, **asdict(metrics), "ranking_meta": meta_first}
                    print(json.dumps(out, indent=2))
                    summaries.append(EvalRunSummary(mode=mode, kind="simple", payload=out))

            else:
                rows = _load_dataset(str(args.dataset))
                rows = filter_rows(rows, query_contains=args.query_contains, query_regex=args.query_regex)
                rows = sample_rows(rows, max_queries=args.max_queries, seed=int(args.sample_seed))
                if args.holdout_eval:
                    train_rows, test_rows = split_deterministic(rows, train_frac=float(args.train_frac), seed=int(args.split_seed))
                    if not test_rows:
                        print("Holdout split produced empty test set; lower --train-frac or increase dataset size")
                        return 2

                    for mode in modes:
                        # Baseline: evaluate test split.
                        test_retrieved_baseline: list[list[str]] = []
                        meta_first: dict[str, Any] | None = None
                        for row in test_rows:
                            retrieved, ranking_meta = _search(
                                client,
                                base_url=args.base_url,
                                query=row.query,
                                limit=max(args.k, 1),
                                hnms_mode=mode,
                            )
                            test_retrieved_baseline.append(retrieved)
                            if meta_first is None and ranking_meta is not None:
                                meta_first = ranking_meta
                            if out_fp is not None:
                                out_fp.write(
                                    json.dumps(
                                        {
                                            "phase": "baseline",
                                            "mode": mode,
                                            "query": row.query,
                                            "expected_ids": row.expected_ids,
                                            "retrieved_ids": retrieved[: args.k],
                                            "ranking_meta": ranking_meta,
                                        }
                                    )
                                    + "\n"
                                )

                        baseline_metrics = aggregate_metrics(test_rows, test_retrieved_baseline, k=args.k)
                        if failures_fp is not None:
                            for r, retrieved in zip(test_rows, test_retrieved_baseline):
                                exp = set(r.expected_ids)
                                topk = set(retrieved[: args.k])
                                if exp and exp.isdisjoint(topk):
                                    failures_fp.write(
                                        json.dumps(
                                            {
                                                "source": "holdout",
                                                "mode": mode,
                                                "phase": "baseline",
                                                "query": r.query,
                                                "expected_ids": r.expected_ids,
                                                "retrieved_ids": retrieved[: args.k],
                                            }
                                        )
                                        + "\n"
                                    )

                        iterations_out: list[dict[str, Any]] = []
                        prev_metrics = baseline_metrics
                        plateau_streak = 0
                        stopped_early = False
                        stopped_iteration: int | None = None
                        stopped_on_regression = False

                        early_cfg = EarlyStopConfig(
                            mrr_delta=args.early_stop_mrr_delta,
                            recall_at_k_delta=args.early_stop_recall_delta,
                            patience=int(args.early_stop_patience),
                        )
                        reg_cfg = RegressionStopConfig(
                            mrr_drop=args.stop_on_regression_mrr,
                            recall_at_k_drop=args.stop_on_regression_recall,
                        )

                        for it in range(1, int(args.iterations) + 1):
                            # Apply feedback from train split (based on current retrieval).
                            for row in train_rows:
                                retrieved, _ = _search(
                                    client,
                                    base_url=args.base_url,
                                    query=row.query,
                                    limit=max(args.k, 1),
                                    hnms_mode=mode,
                                )
                                actions = plan_relevance_feedback(
                                    expected_ids=row.expected_ids,
                                    retrieved_ids=retrieved,
                                    k=args.feedback_k,
                                    include_negative_top1=True,
                                )
                                actions = list(actions)[: max(0, int(args.feedback_max_per_query or 0))]
                                if not args.dry_run:
                                    for a in actions:
                                        if int(args.feedback_max_total or 0) > 0 and feedback_sent_total >= int(
                                            args.feedback_max_total
                                        ):
                                            break
                                        _submit_relevance(
                                            client,
                                            base_url=args.base_url,
                                            memory_id=a.memory_id,
                                            relevant=a.relevant,
                                            query=row.query,
                                            hnms_mode=mode,
                                            trace_id=f"eval:holdout:{mode}:iter{it}",
                                            target_agent=str(args.feedback_target_agent),
                                        )
                                        feedback_sent_total += 1
                                        sleep_ms = int(args.feedback_sleep_ms or 0)
                                        if sleep_ms > 0:
                                            time.sleep(sleep_ms / 1000.0)

                            # Re-evaluate test split after this iteration's feedback.
                            test_retrieved_post: list[list[str]] = []
                            for row in test_rows:
                                retrieved, ranking_meta = _search(
                                    client,
                                    base_url=args.base_url,
                                    query=row.query,
                                    limit=max(args.k, 1),
                                    hnms_mode=mode,
                                )
                                test_retrieved_post.append(retrieved)
                                if out_fp is not None:
                                    phase = "post_feedback" if it == 1 else f"post_feedback_iter{it}"
                                    out_fp.write(
                                        json.dumps(
                                            {
                                                "phase": phase,
                                                "mode": mode,
                                                "query": row.query,
                                                "expected_ids": row.expected_ids,
                                                "retrieved_ids": retrieved[: args.k],
                                                "ranking_meta": ranking_meta,
                                            }
                                        )
                                        + "\n"
                                    )

                                if failures_fp is not None:
                                    phase = "post_feedback" if it == 1 else f"post_feedback_iter{it}"
                                    for r, retrieved in zip(test_rows, test_retrieved_post):
                                        exp = set(r.expected_ids)
                                        topk = set(retrieved[: args.k])
                                        if exp and exp.isdisjoint(topk):
                                            failures_fp.write(
                                                json.dumps(
                                                    {
                                                        "source": "holdout",
                                                        "mode": mode,
                                                        "phase": phase,
                                                        "query": r.query,
                                                        "expected_ids": r.expected_ids,
                                                        "retrieved_ids": retrieved[: args.k],
                                                    }
                                                )
                                                + "\n"
                                            )

                            post_metrics = aggregate_metrics(test_rows, test_retrieved_post, k=args.k)
                            iterations_out.append(
                                {
                                    "iteration": it,
                                    "post_feedback": asdict(post_metrics),
                                    "delta_from_baseline": {
                                        "recall_at_k": post_metrics.recall_at_k - baseline_metrics.recall_at_k,
                                        "mrr": post_metrics.mrr - baseline_metrics.mrr,
                                    },
                                    "delta_from_prev": {
                                        "recall_at_k": post_metrics.recall_at_k - prev_metrics.recall_at_k,
                                        "mrr": post_metrics.mrr - prev_metrics.mrr,
                                    },
                                }
                            )

                            delta_prev_recall = post_metrics.recall_at_k - prev_metrics.recall_at_k
                            delta_prev_mrr = post_metrics.mrr - prev_metrics.mrr

                            if should_stop_on_regression(
                                delta_mrr=delta_prev_mrr,
                                delta_recall_at_k=delta_prev_recall,
                                config=reg_cfg,
                            ):
                                stopped_on_regression = True
                                stopped_iteration = it
                                break

                            plateau_streak, should_stop = update_plateau_streak(
                                current_streak=plateau_streak,
                                delta_mrr=delta_prev_mrr,
                                delta_recall_at_k=delta_prev_recall,
                                config=early_cfg,
                            )
                            prev_metrics = post_metrics

                            if should_stop:
                                stopped_early = True
                                stopped_iteration = it
                                break

                        out = {
                            "mode": mode,
                            "baseline": asdict(baseline_metrics),
                            "iterations": iterations_out,
                            "ranking_meta": meta_first,
                            "train_queries": len(train_rows),
                            "test_queries": len(test_rows),
                            "dry_run": bool(args.dry_run),
                            "stopped_early": stopped_early,
                            "stopped_iteration": stopped_iteration,
                            "stopped_on_regression": stopped_on_regression,
                        }
                        print(json.dumps(out, indent=2))
                        summaries.append(EvalRunSummary(mode=mode, kind="holdout", payload=out))

                    if args.out_summary_json:
                        run_meta = (
                            collect_run_meta(
                                base_url=str(args.base_url),
                                dataset_path=str(args.dataset) if args.dataset else None,
                                resume_jsonl_path=str(args.resume_from_jsonl) if args.resume_from_jsonl else None,
                                repo_root=str(Path(__file__).resolve().parents[2]),
                            )
                            if bool(args.include_run_meta)
                            else {}
                        )
                        write_summary_json(
                            str(args.out_summary_json),
                            summaries=summaries,
                            meta={
                                "kind": "holdout",
                                "dataset": args.dataset,
                                "modes": modes,
                                "k": args.k,
                                "iterations": args.iterations,
                                "dry_run": bool(args.dry_run),
                                "run_meta": run_meta,
                            },
                        )
                    if args.out_summary_csv:
                        write_summary_csv(str(args.out_summary_csv), summaries=summaries)

                    return 0

                for mode in modes:
                    retrieved_per_row: list[list[str]] = []
                    meta_first: dict[str, Any] | None = None

                    for row in rows:
                        retrieved, ranking_meta = _search(
                            client,
                            base_url=args.base_url,
                            query=row.query,
                            limit=max(args.k, 1),
                            hnms_mode=mode,
                        )
                        retrieved_per_row.append(retrieved)
                        if meta_first is None and ranking_meta is not None:
                            meta_first = ranking_meta

                        if out_fp is not None:
                            out_fp.write(
                                json.dumps(
                                    {
                                        "mode": mode,
                                        "query": row.query,
                                        "expected_ids": row.expected_ids,
                                        "retrieved_ids": retrieved[: args.k],
                                        "ranking_meta": ranking_meta,
                                    }
                                )
                                + "\n"
                            )

                        if args.apply_feedback:
                            actions = plan_relevance_feedback(
                                expected_ids=row.expected_ids,
                                retrieved_ids=retrieved,
                                k=args.feedback_k,
                                include_negative_top1=True,
                            )
                            actions = list(actions)[: max(0, int(args.feedback_max_per_query or 0))]
                            if not args.dry_run:
                                for a in actions:
                                    if int(args.feedback_max_total or 0) > 0 and feedback_sent_total >= int(
                                        args.feedback_max_total
                                    ):
                                        break
                                    _submit_relevance(
                                        client,
                                        base_url=args.base_url,
                                        memory_id=a.memory_id,
                                        relevant=a.relevant,
                                        query=row.query,
                                        hnms_mode=mode,
                                        trace_id=f"eval:{mode}",
                                        target_agent=str(args.feedback_target_agent),
                                    )
                                    feedback_sent_total += 1
                                    sleep_ms = int(args.feedback_sleep_ms or 0)
                                    if sleep_ms > 0:
                                        time.sleep(sleep_ms / 1000.0)

                    if args.feedback_only:
                        continue

                    metrics = aggregate_metrics(rows, retrieved_per_row, k=args.k)
                    if failures_fp is not None:
                        for r, retrieved in zip(rows, retrieved_per_row):
                            exp = set(r.expected_ids)
                            topk = set(retrieved[: args.k])
                            if exp and exp.isdisjoint(topk):
                                failures_fp.write(
                                    json.dumps(
                                        {
                                            "source": "dataset",
                                            "mode": mode,
                                            "phase": "default",
                                            "query": r.query,
                                            "expected_ids": r.expected_ids,
                                            "retrieved_ids": retrieved[: args.k],
                                        }
                                    )
                                    + "\n"
                                )
                    out = {"mode": mode, **asdict(metrics), "ranking_meta": meta_first}
                    print(json.dumps(out, indent=2))
                    summaries.append(EvalRunSummary(mode=mode, kind="simple", payload=out))
    finally:
        if out_fp is not None:
            out_fp.close()
        if failures_fp is not None:
            failures_fp.close()

        if args.out_summary_json and summaries:
            run_meta = (
                collect_run_meta(
                    base_url=str(args.base_url),
                    dataset_path=str(args.dataset) if args.dataset else None,
                    resume_jsonl_path=str(args.resume_from_jsonl) if args.resume_from_jsonl else None,
                    repo_root=str(Path(__file__).resolve().parents[2]),
                )
                if bool(args.include_run_meta)
                else {}
            )
            write_summary_json(
                str(args.out_summary_json),
                summaries=summaries,
                meta={
                    "kind": "resume" if args.resume_from_jsonl else "simple",
                    "dataset": args.dataset,
                    "resume_from_jsonl": args.resume_from_jsonl,
                    "modes": modes,
                    "k": args.k,
                    "dry_run": bool(args.dry_run),
                    "run_meta": run_meta,
                },
            )
        if args.out_summary_csv and summaries:
            write_summary_csv(str(args.out_summary_csv), summaries=summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
