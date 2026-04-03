from __future__ import annotations

import time
from statistics import mean
from typing import Any

from tests.benchmarks._fixtures import make_text_events


def _incident_conflict(text: str) -> bool:
    return any(term in text for term in ["incident", "outage", "sev", "failed", "down", "timeout", "rollback", "degraded"])


def _support_conflict(text: str) -> bool:
    return any(term in text for term in [
        # Escalation signals
        "escalation", "escalated", "urgent", "manager", "executive", "breach",
        # Resolution failure
        "blocked", "cannot", "unable", "failure", "outage", "unresolved", "no update", "no response",
        # Business impact / customer dissatisfaction
        "sla", "complaint", "frustrated", "dissatisfied", "not resolved", "reopened",
        # Severity keywords common in support tickets
        "critical", "high priority", "missed deadline",
    ])


def _deployment_conflict(text: str) -> bool:
    return any(term in text for term in [
        "rollback", "migration failed", "deploy failed", "post-deploy incident", "regression",
        "hotfix", "broken build", "revert", "failed deploy",
    ])


def _hr_conflict(text: str) -> bool:
    return any(term in text for term in [
        "grievance", "complaint", "policy violation", "misconduct", "termination",
        "harassment", "dispute", "investigation",
    ])


def _predict_conflict(content: str, domain: str) -> bool:
    text = content.lower()
    dom = (domain or "").lower()
    if dom == "incident":
        return _incident_conflict(text)
    if dom == "support":
        return _support_conflict(text)
    if dom == "deployment":
        return _deployment_conflict(text)
    if dom == "hr":
        return _hr_conflict(text)

    # Fallback detector for mixed/unknown domains.
    return any(term in text for term in ["incident", "outage", "rollback", "timeout", "failed", "error", "critical"])


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


async def run(*, mode: str, strategy: str, dataset: str = "synthetic") -> dict[str, Any]:
    data = make_text_events(140, dataset=dataset)
    timings: list[float] = []
    tp = fp = fn = tn = 0
    domain_confusion: dict[str, dict[str, int]] = {}

    for row in data:
        start = time.perf_counter()
        domain = str(row.get("domain", "unknown") or "unknown")
        pred = _predict_conflict(str(row["content"]), domain)
        timings.append((time.perf_counter() - start) * 1000)
        truth = bool(row["is_conflict"])

        bucket = domain_confusion.setdefault(domain, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if pred and truth:
            tp += 1
            bucket["tp"] += 1
        elif pred and not truth:
            fp += 1
            bucket["fp"] += 1
        elif (not pred) and truth:
            fn += 1
            bucket["fn"] += 1
        else:
            tn += 1
            bucket["tn"] += 1

    out = _metrics(tp, fp, fn)
    out.update(
        {
            "benchmark": "conflict",
            "mode": mode,
            "strategy": strategy,
            "dataset": dataset,
            "avg_latency_ms": round(mean(timings), 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "domain_confusion": domain_confusion,
            "samples": len(data),
            "status": "ok",
        }
    )
    return out
