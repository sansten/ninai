from __future__ import annotations

from datetime import datetime, timedelta, timezone
from random import Random


CONFLICT_TERMS = ["incident", "outage", "rollback", "timeout", "failed", "error"]
# Restricted to terms that overlap the recall benchmark query ("authentication failure spike")
# so the gold set is always retrievable by the keyword ranker.
RELEVANCE_TERMS = ["authentication", "failure", "spike"]


def _make_synthetic_events(n: int = 120, seed: int = 42) -> list[dict]:
    rng = Random(seed)
    events: list[dict] = []
    for i in range(n):
        token = rng.choice(["auth", "billing", "cache", "queue", "search"])
        is_conflict = token in {"billing", "queue"} and i % 4 == 0
        # Sparser relevance criterion: ~2/5 tokens × 1/10 positions ≈ 8 gold items per 200 events,
        # small enough for recall@10 to cover the full gold set.
        is_relevant = token in {"auth", "search"} and i % 10 == 0

        conflict_phrase = rng.choice(CONFLICT_TERMS) if is_conflict else "healthy"
        relevance_phrase = rng.choice(RELEVANCE_TERMS) if is_relevant else "routine"
        content = f"{token} service event {i}: {relevance_phrase} telemetry, {conflict_phrase} signal"

        events.append(
            {
                "id": f"e-{i}",
                "content": content,
                "tags": [token],
                "is_conflict": is_conflict,
                "is_relevant": is_relevant,
            }
        )
    return events


_SUPPORT_CONFLICT_TERMS = [
    "escalation", "escalated", "urgent", "breach", "sla", "complaint",
    "frustrated", "dissatisfied", "not resolved", "reopened", "unresolved",
    "no update", "no response", "blocked", "cannot", "unable",
    "failure", "outage", "critical", "high priority", "missed deadline",
]


def _kaggle_is_conflict(text: str, domain: str, priority: str) -> bool:
    text_l = (text or "").lower()
    if domain == "incident" and priority in {"high", "critical"}:
        return True
    if domain == "support":
        return any(term in text_l for term in _SUPPORT_CONFLICT_TERMS)
    return any(term in text_l for term in ["outage", "down", "failed", "timeout", "critical", "incident"])


def _kaggle_is_conflict_structured(*, domain: str, priority: str, source_dataset: str, resolution: str | None) -> bool | None:
    ds = (source_dataset or "").lower()
    dom = (domain or "").lower()
    prio = (priority or "").lower()
    has_resolution = bool(str(resolution or "").strip())

    if ds == "incident_event_log":
        if prio in {"high", "critical"}:
            return True
        if prio == "low" and has_resolution:
            return False

    if dom == "incident":
        if prio in {"high", "critical"}:
            return True
        if prio == "low" and has_resolution:
            return False

    if dom == "deployment":
        if has_resolution and prio in {"low", "medium"}:
            return False

    if dom == "support":
        if prio in {"critical", "high"}:
            return True
        if has_resolution and prio in {"low", "medium"}:
            return False

    return None


def _kaggle_is_relevant(text: str) -> bool:
    text_l = (text or "").lower()
    return any(term in text_l for term in ["auth", "authentication", "login", "jwt", "failure", "spike", "credential"])


def _make_kaggle_events(n: int = 120) -> list[dict]:
    from tests.e2e.data import HELPDESK_TICKETS  # noqa: PLC0415

    tickets = list(HELPDESK_TICKETS)
    if not tickets:
        return []

    now = datetime.now(timezone.utc)
    events: list[dict] = []
    for i in range(min(n, len(tickets))):
        ticket = tickets[i]
        subject = str(ticket.get("subject", ""))
        description = str(ticket.get("description", ""))
        text = f"{subject} {description}".strip()
        domain = str(ticket.get("domain", "")).lower()
        priority = str(ticket.get("priority", "")).lower()
        source_dataset = str(ticket.get("_source_dataset", ""))
        resolution = str(ticket.get("resolution")) if ticket.get("resolution") is not None else None
        created_at = now - timedelta(days=int(ticket.get("days_ago", 0) or 0))

        structured_conflict = _kaggle_is_conflict_structured(
            domain=domain,
            priority=priority,
            source_dataset=source_dataset,
            resolution=resolution,
        )
        conflict_label = structured_conflict if structured_conflict is not None else _kaggle_is_conflict(text, domain, priority)

        events.append(
            {
                "id": str(ticket.get("ticket_id", f"k-{i}")),
                "content": text,
                "tags": list(ticket.get("tags") or []),
                "is_conflict": bool(conflict_label),
                "is_relevant": _kaggle_is_relevant(text),
                "created_at": created_at,
                "domain": domain,
                "priority": priority,
                "source_dataset": source_dataset,
            }
        )
    return events


def make_text_events(n: int = 120, seed: int = 42, dataset: str = "synthetic") -> list[dict]:
    dataset_name = str(dataset or "synthetic").strip().lower()
    if dataset_name == "kaggle":
        rows = _make_kaggle_events(n=n)
        if rows:
            return rows
    return _make_synthetic_events(n=n, seed=seed)


# Human-reviewed gold labels independent of the keyword heuristic.
# These cover cases the heuristic misclassifies: confidential content without
# obvious trigger words, and non-sensitive content that contains trigger-like language.
GOAL_GOLD_LABELS: list[tuple[str, str]] = [
    # Confidential — no obvious keywords; classified by context
    ("Employee compensation bands and salary ranges were shared in the Q3 HR sync.", "confidential"),
    ("Customer health insurance claim details reviewed during benefit enrollment.", "confidential"),
    ("Board meeting minutes include acquisition target valuation and NDA terms.", "confidential"),
    ("User financial account balance and transaction history exported for audit.", "confidential"),
    ("Personal identification documents uploaded to the compliance review folder.", "confidential"),
    ("Source code for unreleased product feature sent to external contractor.", "confidential"),
    # Internal — contain trigger-like words but are not sensitive
    ("The password manager product roadmap was published to the engineering blog.", "internal"),
    ("Incident postmortem for the login failure published in the public wiki.", "internal"),
    ("API key rotation best practices documented in the developer handbook.", "internal"),
    ("Training video on how to identify and avoid social engineering attacks.", "internal"),
    ("Runbook updated to describe how to recover from an authentication outage.", "internal"),
    ("Credit card processing flow diagram shared during onboarding session.", "internal"),
]


def make_goal_samples(n: int = 8, dataset: str = "synthetic") -> list[tuple[str, str]]:
    dataset_name = str(dataset or "synthetic").strip().lower()
    if dataset_name == "kaggle":
        # Start with gold-labeled samples (harder, independent of keyword heuristic).
        samples: list[tuple[str, str]] = list(GOAL_GOLD_LABELS)
        # Fill remaining slots from Kaggle tickets using keyword heuristic.
        remaining = max(0, n - len(samples))
        if remaining > 0:
            rows = make_text_events(n=max(remaining, 80), dataset="kaggle")
            for row in rows:
                content = str(row.get("content", ""))
                lowered = content.lower()
                truth = "confidential" if any(term in lowered for term in ["password", "credit card", "api key", "ssn", "secret"]) else "internal"
                samples.append((content, truth))
                if len(samples) >= n:
                    break
        if samples:
            return samples[:n]

    base = [
        ("Customer provided credit card details and requested urgent password reset.", "confidential"),
        ("Status note: authentication failure spike detected in production.", "internal"),
        ("Confidential contract and API key shared in meeting recap.", "confidential"),
        ("Weekly planning update for support operations and queue health.", "internal"),
        ("SSN and payroll records were shared during incident triage.", "confidential"),
        ("Public release notes draft for internal search improvements.", "internal"),
        ("Password and token reset request from customer account team.", "confidential"),
        ("Runbook update for queue processor backlog handling.", "internal"),
    ]
    if n <= len(base):
        return base[:n]

    out: list[tuple[str, str]] = []
    i = 0
    while len(out) < n:
        content, label = base[i % len(base)]
        out.append((f"{content} sample-{i}", label))
        i += 1
    return out


def split_gold_ids(items: list[dict], key: str) -> set[str]:
    return {str(row["id"]) for row in items if bool(row.get(key))}
