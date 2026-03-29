"""Real-world IT helpdesk and enterprise dataset for end-to-end tests.

Loads from three Kaggle datasets downloaded to tests/e2e/kaggle_data/:
  - customer_support_tickets.csv       (suraj520/customer-support-ticket-dataset)
  - all_tickets_processed_improved_v3.csv  (adisongoh/it-service-ticket-classification-dataset)
  - incident_event_log.csv             (vipulshinde/incident-response-log)

Falls back to a small embedded sample set if the CSVs are not present so
tests remain runnable in CI without a Kaggle token.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "kaggle_data"

# ---------------------------------------------------------------------------
# Column / value mappings from raw CSV fields to Ninai domain model
# ---------------------------------------------------------------------------

_TICKET_TYPE_TO_DOMAIN = {
    "Technical issue": "incident",
    "Billing inquiry": "support",
    "Refund request": "support",
    "Product inquiry": "support",
    "Cancellation request": "support",
}

_PRIORITY_MAP = {
    "Critical": "critical",
    "High": "high",
    "Medium": "medium",
    "Low": "low",
    "1 - Critical": "critical",
    "2 - High": "high",
    "3 - Moderate": "medium",
    "4 - Low": "low",
}

_CHANNEL_TO_SOURCE = {
    "Email": "email",
    "Phone": "api",
    "Social media": "chat",
    "Chat": "chat",
    "Phone ": "api",
}

_TOPIC_TO_DOMAIN = {
    "Hardware": "incident",
    "HR Support": "hr",
    "Access": "incident",
    "Miscellaneous": "support",
    "Storage": "engineering",
    "Purchase": "support",
    "Internal Project": "project",
    "Administrative rights": "engineering",
}

# Source/role defaults per domain
_DOMAIN_SOURCE = {
    "incident": "api",
    "support": "email",
    "engineering": "api",
    "hr": "email",
    "project": "email",
    "strategy": "email",
    "document": "email",
}

_DOMAIN_ROLE = {
    "incident": "engineer",
    "support": "support_agent",
    "engineering": "engineer",
    "hr": "manager",
    "project": "manager",
    "strategy": "manager",
    "document": "engineer",
}


def _clean_text(text: str) -> str:
    text = re.sub(r"\{product_purchased\}", "the device", str(text))
    text = re.sub(r"icon\s+icon|icon\s+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:400]


# ---------------------------------------------------------------------------
# Loader: customer_support_tickets.csv
# ---------------------------------------------------------------------------

def _load_customer_support(n: int = 25) -> list[dict[str, Any]]:
    csv_path = _DATA_DIR / "customer_support_tickets.csv"
    if not csv_path.exists():
        return []

    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Ticket Description"])

    # Sample across all priority x type combos for coverage
    sampled = (
        df.groupby(["Ticket Priority", "Ticket Type"], group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
        .reset_index(drop=True)
        .head(n)
    )

    tickets: list[dict[str, Any]] = []
    for i, row in sampled.iterrows():
        priority = _PRIORITY_MAP.get(str(row.get("Ticket Priority", "medium")), "medium")
        domain = _TICKET_TYPE_TO_DOMAIN.get(str(row.get("Ticket Type", "")), "support")
        source = _CHANNEL_TO_SOURCE.get(str(row.get("Ticket Channel", "email")), "email")

        # Critical/high tickets are more recent; low/medium are older
        days_map = {"critical": 2, "high": 5, "medium": 12, "low": 20}
        days_ago = days_map[priority] + (int(i) % 7)

        resolution_raw = row.get("Resolution")
        resolution = str(resolution_raw) if pd.notna(resolution_raw) else None

        tickets.append({
            "ticket_id": f"CS-{int(row.get('Ticket ID', i))}",
            "days_ago": days_ago,
            "domain": domain,
            "priority": priority,
            "source": source,
            "author_role": "support_agent" if domain == "support" else "engineer",
            "subject": _clean_text(str(row.get("Ticket Subject", "Support request"))),
            "description": _clean_text(str(row.get("Ticket Description", ""))),
            "resolution": resolution,
            "entities": {},
            "reference_count": {"critical": 5, "high": 3, "medium": 2, "low": 1}[priority],
            "tags": [domain, priority],
            "_source_dataset": "customer_support_tickets",
        })
    return tickets


# ---------------------------------------------------------------------------
# Loader: all_tickets_processed_improved_v3.csv (IT service classification)
# ---------------------------------------------------------------------------

def _load_it_service_tickets(n: int = 20) -> list[dict[str, Any]]:
    csv_path = _DATA_DIR / "all_tickets_processed_improved_v3.csv"
    if not csv_path.exists():
        return []

    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Document"])
    df = df[df["Document"].str.len() > 20]

    # Two samples per topic group for coverage
    sampled = (
        df.groupby("Topic_group", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 3), random_state=42))
        .reset_index(drop=True)
        .head(n)
    )

    tickets: list[dict[str, Any]] = []
    for i, row in sampled.iterrows():
        topic = str(row.get("Topic_group", "Miscellaneous"))
        domain = _TOPIC_TO_DOMAIN.get(topic, "support")

        # Vary staleness: project/hr/engineering are older
        age_by_domain = {
            "incident": 3, "support": 8, "engineering": 25,
            "hr": 60, "project": 45, "document": 120,
        }
        days_ago = age_by_domain.get(domain, 10) + (int(i) % 10)

        tickets.append({
            "ticket_id": f"IT-{i:04d}",
            "days_ago": days_ago,
            "domain": domain,
            "priority": "medium",
            "source": _DOMAIN_SOURCE.get(domain, "email"),
            "author_role": _DOMAIN_ROLE.get(domain, "engineer"),
            "subject": f"{topic} request",
            "description": _clean_text(str(row.get("Document", ""))),
            "resolution": None,
            "entities": {},
            "reference_count": 2,
            "tags": [domain, topic.lower().replace(" ", "_")],
            "_source_dataset": "it_service_tickets",
            "_topic_group": topic,
        })
    return tickets


# ---------------------------------------------------------------------------
# Loader: incident_event_log.csv (ServiceNow)
# ---------------------------------------------------------------------------

def _load_incident_log(n: int = 15) -> list[dict[str, Any]]:
    csv_path = _DATA_DIR / "incident_event_log.csv"
    if not csv_path.exists():
        return []

    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(csv_path)
    incidents = df.groupby("number").first().reset_index()

    # Parse opened_at, compute relative days within dataset range → map to 1–90 days
    incidents["opened_at_parsed"] = pd.to_datetime(
        incidents["opened_at"], dayfirst=True, errors="coerce"
    )
    incidents = incidents.dropna(subset=["opened_at_parsed"])

    t_min = incidents["opened_at_parsed"].min()
    t_max = incidents["opened_at_parsed"].max()
    span = (t_max - t_min).total_seconds() or 1.0

    # Sample across all priority levels
    sampled = (
        incidents.groupby("priority", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 4), random_state=42))
        .reset_index(drop=True)
        .head(n)
    )

    tickets: list[dict[str, Any]] = []
    for i, row in sampled.iterrows():
        priority = _PRIORITY_MAP.get(str(row.get("priority", "3 - Moderate")), "medium")
        contact = str(row.get("contact_type", "Phone"))
        source = _CHANNEL_TO_SOURCE.get(contact, "api")

        # Map relative dataset position → days_ago (1 = oldest, 90 = newest)
        rel = (row["opened_at_parsed"] - t_min).total_seconds() / span
        days_ago = max(1, int(90 - rel * 89))

        has_resolution = pd.notna(row.get("resolved_at")) and str(row.get("resolved_at", "")).strip() not in ("", "?")

        tickets.append({
            "ticket_id": str(row.get("number", f"INC-{i}")),
            "days_ago": days_ago,
            "domain": "incident",
            "priority": priority,
            "source": source,
            "author_role": "engineer",
            "subject": f"Incident {row.get('number', i)} — {row.get('category', 'system')} / {row.get('subcategory', 'unknown')}",
            "description": (
                f"ServiceNow incident {row.get('number', '')} raised via {contact}. "
                f"Category: {row.get('category', '?')}. Impact: {row.get('impact', '?')}. "
                f"Urgency: {row.get('urgency', '?')}. State: {row.get('incident_state', '?')}."
            ),
            "resolution": "Resolved via standard procedure" if has_resolution else None,
            "entities": {str(row.get("assignment_group", "ops")): "team"},
            "reference_count": {"critical": 6, "high": 4, "medium": 2, "low": 1}[priority],
            "tags": ["incident", priority, str(row.get("category", ""))],
            "_source_dataset": "incident_event_log",
            "_original_priority": str(row.get("priority", "")),
        })
    return tickets


# ---------------------------------------------------------------------------
# Conflict / anomaly enrichment snapshots (small, hand-crafted)
# ---------------------------------------------------------------------------

HEALTHY_ENRICHMENT_SNAPSHOTS: list[dict[str, Any]] = [
    {"credibility_score": 0.88, "uncertainty_score": 0.12, "uncertainty_level": "low",
     "conflict_count": 0, "temporal_confidence": 0.85, "decay_factor": 0.75, "is_stale": False},
    {"credibility_score": 0.82, "uncertainty_score": 0.18, "uncertainty_level": "low",
     "conflict_count": 0, "temporal_confidence": 0.80, "decay_factor": 0.80, "is_stale": False},
    {"credibility_score": 0.90, "uncertainty_score": 0.10, "uncertainty_level": "low",
     "conflict_count": 1, "temporal_confidence": 0.88, "decay_factor": 0.70, "is_stale": False},
    {"credibility_score": 0.75, "uncertainty_score": 0.20, "uncertainty_level": "medium",
     "conflict_count": 0, "temporal_confidence": 0.78, "decay_factor": 0.65, "is_stale": False},
    {"credibility_score": 0.85, "uncertainty_score": 0.15, "uncertainty_level": "low",
     "conflict_count": 0, "temporal_confidence": 0.82, "decay_factor": 0.72, "is_stale": False},
]

CRISIS_ENRICHMENT_SNAPSHOTS: list[dict[str, Any]] = [
    {"credibility_score": 0.08, "uncertainty_score": 0.94, "uncertainty_level": "critical",
     "conflict_count": 9, "temporal_confidence": 0.03, "decay_factor": 0.02, "is_stale": True},
    {"credibility_score": 0.15, "uncertainty_score": 0.88, "uncertainty_level": "critical",
     "conflict_count": 6, "temporal_confidence": 0.10, "decay_factor": 0.05, "is_stale": False},
    {"credibility_score": 0.20, "uncertainty_score": 0.82, "uncertainty_level": "high",
     "conflict_count": 5, "temporal_confidence": 0.08, "decay_factor": 0.04, "is_stale": True},
]


# (HELPDESK_TICKETS is assigned after _SEMANTIC_FIXTURES and _FALLBACK_TICKETS are defined)


def get_ticket(ticket_id: str) -> dict[str, Any]:
    for t in HELPDESK_TICKETS:
        if t["ticket_id"] == ticket_id:
            return t
    raise KeyError(f"ticket {ticket_id!r} not found")


def get_tickets_by_domain(domain: str) -> list[dict[str, Any]]:
    return [t for t in HELPDESK_TICKETS if t["domain"] == domain]


def get_tickets_by_priority(priority: str) -> list[dict[str, Any]]:
    return [t for t in HELPDESK_TICKETS if t["priority"] == priority]


# ---------------------------------------------------------------------------
# Context builders — convert a ticket dict to agent context shapes
# ---------------------------------------------------------------------------

def make_memory_context(ticket: dict[str, Any], strategy: str = "heuristic") -> dict[str, Any]:
    """Build a MemoryDecayAgent context from a ticket record."""
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=ticket["days_ago"])).isoformat()
    return {
        "memory": {
            "content": ticket["description"],
            "created_at": created_at,
            "enrichment": {
                "domain": ticket["domain"],
                "reference_count": ticket.get("reference_count", 1),
                "last_accessed_at": None,
            },
        },
        "runtime": {"job_id": ticket["ticket_id"]},
    }


def make_credibility_context(
    ticket: dict[str, Any],
    conflicts: list | None = None,
    corroborating: list | None = None,
) -> dict[str, Any]:
    """Build a CredibilityAgent context from a ticket record."""
    return {
        "memory": {
            "content": ticket["description"],
            "source_type": ticket.get("source", "email"),
            "author_role": ticket.get("author_role", "user"),
            "enrichment": {
                "entities": ticket.get("entities", {}),
                "conflicts": conflicts or [],
                "corroborating_memories": corroborating or [],
            },
        },
        "runtime": {},
    }


def make_anomaly_context(enrichment: dict[str, Any]) -> dict[str, Any]:
    """Build an AnomalyDetectionAgent context from an enrichment snapshot."""
    return {
        "memory": {"enrichment": enrichment},
        "runtime": {"job_id": "anomaly-test"},
    }


def make_conflict_context(
    ticket: dict[str, Any],
    world_nodes: list | None = None,
    propagated_signals: list | None = None,
) -> dict[str, Any]:
    """Build a ConflictDetectionAgent context from a ticket record."""
    return {
        "tenant": {"org_id": "org-kaggle", "org_slug": "kaggle-test"},
        "memory": {
            "id": ticket["ticket_id"],
            "content": ticket["description"],
            "enrichment": {
                "world_nodes": world_nodes or [],
                "propagated_signals": propagated_signals or [],
                "causal_chains": [],
            },
        },
        "runtime": {"job_id": ticket["ticket_id"]},
    }


# ---------------------------------------------------------------------------
# Semantic test fixtures — fixed IDs required by specific test assertions.
# These are always present regardless of CSV availability.
# ---------------------------------------------------------------------------

_SEMANTIC_FIXTURES: list[dict[str, Any]] = [
    # Fresh incidents
    {
        "ticket_id": "INC-001", "days_ago": 2, "domain": "incident", "priority": "high",
        "source": "slack", "author_role": "engineer",
        "subject": "Production auth service down — JWT validation failing",
        "description": "Auth service is returning 500 errors on all JWT validation requests. All tokens rejected with 401. Engineers investigating token-signing key rotation issue.",
        "resolution": None, "entities": {"AuthService": "service", "JWTValidator": "component"}, "reference_count": 4, "tags": ["incident", "auth", "prod"],
    },
    {
        "ticket_id": "INC-002", "days_ago": 1, "domain": "incident", "priority": "critical",
        "source": "system", "author_role": "system",
        "subject": "Database connection pool exhausted on prod",
        "description": "System alert: PostgreSQL connection pool exhausted. All new connections refused. Max pool size 200 reached. Celery workers backing up.",
        "resolution": None, "entities": {"PostgreSQL": "service", "CeleryWorker": "component"}, "reference_count": 6, "tags": ["incident", "database", "prod"],
    },
    {
        "ticket_id": "INC-003", "days_ago": 3, "domain": "incident", "priority": "critical",
        "source": "api", "author_role": "engineer",
        "subject": "Payment gateway timeout causing checkout failures",
        "description": "Stripe payment gateway timing out after 30s on all checkout requests. Failure rate at 61%. APAC region most affected. Circuit breaker not triggering.",
        "resolution": None, "entities": {"Stripe": "service", "CheckoutService": "component"}, "reference_count": 7, "tags": ["incident", "payment", "prod"],
    },
    {
        "ticket_id": "INC-004", "days_ago": 4, "domain": "incident", "priority": "low",
        "source": "chat", "author_role": "user",
        "subject": "Something seems slow on prod",
        "description": "Hey, things seem a bit slow today? Not sure if it's just me.",
        "resolution": None, "entities": {}, "reference_count": 1, "tags": ["incident"],
    },
    # Stale incidents
    {
        "ticket_id": "INC-010", "days_ago": 45, "domain": "incident", "priority": "high",
        "source": "slack", "author_role": "engineer",
        "subject": "Auth outage post-deploy — 45 days ago",
        "description": "Auth service went down following v2.3.0 deployment. JWT key rotation caused validation failures for 2 hours.",
        "resolution": "Rolled back to v2.2.9. Root cause: HS256 key mismatch.", "entities": {"AuthService": "service"}, "reference_count": 2, "tags": ["incident", "auth"],
    },
    {
        "ticket_id": "INC-011", "days_ago": 80, "domain": "incident", "priority": "medium",
        "source": "api", "author_role": "engineer",
        "subject": "DB migration failure — 80 days ago",
        "description": "Alembic migration 0038 failed on production. Column rename caused FK constraint violation. Rollback executed.",
        "resolution": "Migration fixed in 0039 with explicit FK drop/recreate.", "entities": {"PostgreSQL": "service"}, "reference_count": 1, "tags": ["incident", "database"],
    },
    # Strategy / long half-life
    {
        "ticket_id": "STR-001", "days_ago": 120, "domain": "strategy", "priority": "low",
        "source": "email", "author_role": "manager",
        "subject": "Company OKRs for FY2026 — expand to APAC market",
        "description": "FY2026 objectives finalized: expand to APAC, achieve 99.9% uptime SLA, launch enterprise tier by Q3. Board approved 2026-01-15.",
        "resolution": None, "entities": {}, "reference_count": 10, "tags": ["strategy", "okr"],
    },
    # Conflict scenario pair — claim auth is healthy vs. claim auth is failing
    {
        "ticket_id": "CON-001", "days_ago": 2, "domain": "deployment", "priority": "medium",
        "source": "api", "author_role": "engineer",
        "subject": "Auth service fully operational — all endpoints healthy",
        "description": "Deployment v2.4.1 completed. Auth service is fully operational. All endpoints responding within SLA. JWT validation healthy.",
        "resolution": None, "entities": {"AuthService": "service"}, "reference_count": 3, "tags": ["deployment", "auth"],
    },
    {
        "ticket_id": "CON-002", "days_ago": 2, "domain": "incident", "priority": "critical",
        "source": "slack", "author_role": "engineer",
        "subject": "Auth service experiencing critical failures",
        "description": "Auth service is down. JWT validation returning 500. All login attempts failing. Production impacted.",
        "resolution": None, "entities": {"AuthService": "service"}, "reference_count": 5, "tags": ["incident", "auth"],
    },
    # Deployment
    {
        "ticket_id": "DEP-001", "days_ago": 5, "domain": "deployment", "priority": "medium",
        "source": "api", "author_role": "engineer",
        "subject": "v2.4.0 deployed to production — auth service updated",
        "description": "Release v2.4.0 deployed to all production nodes. Auth service updated with RS256 support. DB migrations 0041-0042 applied.",
        "resolution": "Deployment successful, all health checks passing.", "entities": {"AuthService": "service"}, "reference_count": 4, "tags": ["deployment"],
    },
]


def _build_tickets() -> list[dict[str, Any]]:
    tickets = []
    tickets.extend(_load_customer_support(25))
    tickets.extend(_load_it_service_tickets(20))
    tickets.extend(_load_incident_log(15))

    # Fallback if no CSVs present (CI / no Kaggle token)
    if not tickets:
        tickets = list(_FALLBACK_TICKETS)

    # Always append semantic fixtures (specific IDs required by test assertions)
    existing_ids = {t["ticket_id"] for t in tickets}
    for fix in _SEMANTIC_FIXTURES:
        if fix["ticket_id"] not in existing_ids:
            tickets.append(fix)

    return tickets


# ---------------------------------------------------------------------------
# Fallback embedded sample — used when CSVs are absent (CI environments)
# ---------------------------------------------------------------------------

_FALLBACK_TICKETS: list[dict[str, Any]] = [
    {
        "ticket_id": "FB-001", "days_ago": 2, "domain": "incident", "priority": "high",
        "source": "api", "author_role": "engineer",
        "subject": "Auth service down",
        "description": "Production authentication service is returning 500 errors on all JWT validation requests. Users cannot log in.",
        "resolution": None, "entities": {"AuthService": "service"}, "reference_count": 5, "tags": ["incident", "auth"],
    },
    {
        "ticket_id": "FB-002", "days_ago": 45, "domain": "incident", "priority": "medium",
        "source": "email", "author_role": "engineer",
        "subject": "Database failover",
        "description": "Database failover event occurred. Primary replica promoted. Services restored after 12 minutes.",
        "resolution": "Failover completed successfully", "entities": {"PostgreSQL": "service"}, "reference_count": 2, "tags": ["incident", "db"],
    },
    {
        "ticket_id": "FB-003", "days_ago": 8, "domain": "support", "priority": "medium",
        "source": "email", "author_role": "support_agent",
        "subject": "Password reset not working",
        "description": "User reports the forgot-password link sends an email but the reset link returns a 404 error.",
        "resolution": "Fixed redirect URL in password reset handler", "entities": {}, "reference_count": 3, "tags": ["support"],
    },
    {
        "ticket_id": "FB-004", "days_ago": 120, "domain": "strategy", "priority": "low",
        "source": "email", "author_role": "manager",
        "subject": "FY2026 OKRs finalized",
        "description": "Company objectives for FY2026 finalized: expand to APAC, improve reliability SLAs, launch enterprise tier.",
        "resolution": None, "entities": {}, "reference_count": 8, "tags": ["strategy"],
    },
    {
        "ticket_id": "FB-005", "days_ago": 3, "domain": "deployment", "priority": "high",
        "source": "api", "author_role": "engineer",
        "subject": "v2.4.0 deployed to production",
        "description": "Release v2.4.0 deployed to all production nodes. Auth service updated, database migrations applied.",
        "resolution": "Deployment successful", "entities": {"AuthService": "service"}, "reference_count": 4, "tags": ["deployment"],
    },
]

HELPDESK_TICKETS: list[dict[str, Any]] = _build_tickets()
