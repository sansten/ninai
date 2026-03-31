"""Kaggle data loading and persona builders for all 5 demo cases.

Kaggle sources (in tests/e2e/kaggle_data/):
  customer_support_tickets.csv  — customer-facing tickets
    columns: Ticket ID, Ticket Priority, Ticket Type, Ticket Channel,
             Ticket Subject, Ticket Description, Resolution

  incident_event_log.csv        — IT/internal incident log
    columns: number, priority, contact_type, opened_at, resolved_at,
             category, subcategory, impact, urgency, incident_state,
             assignment_group

All build_* functions return list[dict] in the standard Ninai memory record
format accepted by NinaiClient.memories.create() and by the agent context
builder in agents.py.

Memory record format:
{
    "id":            str,       # local reference key (not DB id)
    "title":         str,
    "content":       str,
    "memory_type":   "semantic",
    "scope":         "personal" | "team" | "organization",
    "source_type":   "email" | "phone" | "portal" | "slack" | "system",
    "author_role":   "customer" | "engineer" | "support_agent" | "manager",
    "tags":          list[str],
    "days_ago":      int,       # synthetic age for temporal reasoning
    "extra_metadata": {
        "org_id":      str,     # from org_data["org_id"]
        "team_id":     str,     # from org_data["teams"][slug]["team_id"]
        "team_slug":   str,
        "user_email":  str,
        "user_id":     str,
        "ticket_id":   str,
        "priority":    str,
        "category":    str,
        "resolution":  str | None,
        "case":        int,     # 1-5
    },
}
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_BACKEND = _HERE.parent.parent
_KAGGLE = _BACKEND / "tests" / "e2e" / "kaggle_data"

SUPPORT_CSV = _KAGGLE / "customer_support_tickets.csv"
INCIDENT_CSV = _KAGGLE / "incident_event_log.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_support(nrows: int | None = None) -> pd.DataFrame:
    """Load customer_support_tickets.csv with normalised column names."""
    df = pd.read_csv(SUPPORT_CSV, nrows=nrows, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # keep useful columns; fill NaN with empty string
    keep = [
        "ticket_id", "ticket_type", "ticket_priority", "ticket_channel",
        "ticket_subject", "ticket_description", "resolution",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].fillna("")
    return df


def load_incident(nrows: int | None = None) -> pd.DataFrame:
    """Load incident_event_log.csv with normalised column names."""
    df = pd.read_csv(INCIDENT_CSV, nrows=nrows, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    keep = [
        "number", "priority", "contact_type", "opened_at", "resolved_at",
        "category", "subcategory", "impact", "urgency", "incident_state",
        "assignment_group",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].fillna("")
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _days_ago(n: int) -> str:
    """ISO timestamp for N days ago (UTC)."""
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _record(
    local_id: str,
    title: str,
    content: str,
    source_type: str,
    author_role: str,
    tags: list[str],
    days_ago: int,
    scope: str,
    org_id: str,
    team_slug: str,
    team_id: str,
    user_email: str,
    user_id: str,
    ticket_id: str,
    priority: str,
    category: str,
    resolution: str | None,
    case: int,
) -> dict[str, Any]:
    """Construct a standard Ninai memory record dict."""
    return {
        "id": local_id,
        "title": title,
        "content": content,
        "memory_type": "semantic",
        "scope": scope,
        "source_type": source_type,
        "author_role": author_role,
        "tags": tags,
        "days_ago": days_ago,
        "extra_metadata": {
            "org_id": org_id,
            "team_id": team_id,
            "team_slug": team_slug,
            "user_email": user_email,
            "user_id": user_id,
            "ticket_id": ticket_id,
            "priority": priority,
            "category": category,
            "resolution": resolution,
            "case": case,
            "created_at": _days_ago(days_ago),
        },
    }


# ---------------------------------------------------------------------------
# Case 1 — The Repeat Caller (Emma Chen)
# ---------------------------------------------------------------------------

def build_case1_records(
    df_support: pd.DataFrame,
    org_data: dict,
) -> list[dict]:
    """Build 3 repeat-caller tickets for Emma Chen.

    Pulls 3 software-type rows from the support CSV and reframes them as
    Emma's escalating login issue.  The content escalates: each ticket
    references the prior failed resolution so Ninai can detect the pattern.
    """
    org_id = org_data["org_id"]
    l1_team = org_data["teams"]["support-l1"]
    carlos = org_data["users"]["carlos.mendez@meridian-tech.com"]
    sarah = org_data["users"]["sarah.okafor@meridian-tech.com"]
    mei = org_data["users"]["mei.zhang@meridian-tech.com"]

    # Pull real descriptions from CSV — prefer tickets about login/access/auth,
    # fall back to any "Technical issue" type (actual Kaggle values, not "software")
    auth_rows = df_support[
        df_support.get("ticket_description", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .str.contains(r"login|password|access|auth|account|sign.in", na=False, regex=True)
    ].head(3)
    if len(auth_rows) < 3:
        tech_rows = df_support[
            df_support.get("ticket_type", pd.Series(dtype=str))
            .astype(str)
            .str.lower()
            .str.contains("technical", na=False)
        ].head(3)
        sw_rows = tech_rows if len(tech_rows) >= 3 else df_support.head(3)
    else:
        sw_rows = auth_rows
    base_descriptions = list(sw_rows["ticket_description"].values) if len(sw_rows) >= 3 else [
        "I cannot log in to my account. The page just shows an error.",
        "Still cannot log in. Password reset did not help.",
        "My login is broken again. The reset does not work.",
    ]

    records = [
        _record(
            local_id="C1-T1",
            title="Emma Chen — Cannot log in (1st contact)",
            content=(
                f"Customer: Emma Chen (emma.chen@meridian-client.com). "
                f"Issue: {base_descriptions[0][:200]} "
                "Assigned to: Carlos Mendez (Support L1). "
                "Resolution attempt: password reset link sent."
            ),
            source_type="email",
            author_role="support_agent",
            tags=["login", "auth", "password-reset", "case-1"],
            days_ago=8,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email="carlos.mendez@meridian-tech.com",
            user_id=carlos["user_id"],
            ticket_id="TKT-2026-0311",
            priority="medium",
            category="software/auth",
            resolution="password reset sent",
            case=1,
        ),
        _record(
            local_id="C1-T2",
            title="Emma Chen — Still cannot log in (2nd contact, 3 days later)",
            content=(
                "Customer: Emma Chen (emma.chen@meridian-client.com). "
                f"Follow-up: {base_descriptions[1][:200]} "
                "Previous ticket TKT-2026-0311 — password reset did not resolve the issue. "
                "Assigned to: Sarah Okafor (Support L1). "
                "Resolution attempt: browser cache cleared, cookies reset."
            ),
            source_type="phone",
            author_role="support_agent",
            tags=["login", "auth", "cache", "repeat-contact", "case-1"],
            days_ago=5,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email="sarah.okafor@meridian-tech.com",
            user_id=sarah["user_id"],
            ticket_id="TKT-2026-0314",
            priority="medium",
            category="software/auth",
            resolution="cache cleared",
            case=1,
        ),
        _record(
            local_id="C1-T3",
            title="Emma Chen — Login broken again (3rd contact, furious)",
            content=(
                "Customer: Emma Chen (emma.chen@meridian-client.com). "
                f"Third contact this week: {base_descriptions[2][:200]} "
                "Previous tickets: TKT-2026-0311 (password reset), TKT-2026-0314 (cache clear) — "
                "BOTH FAILED. Customer is escalating. Account access lost for 8 days. "
                "Assigned to: Mei Zhang (Support L1). "
                "No resolution yet — escalation required."
            ),
            source_type="phone",
            author_role="support_agent",
            tags=["login", "auth", "escalation", "repeat-contact", "case-1", "urgent"],
            days_ago=2,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email="mei.zhang@meridian-tech.com",
            user_id=mei["user_id"],
            ticket_id="TKT-2026-0317",
            priority="high",
            category="software/auth",
            resolution=None,
            case=1,
        ),
    ]
    return records


# ---------------------------------------------------------------------------
# Case 2 — The Team Storm (DevOps Week-12)
# ---------------------------------------------------------------------------

def build_case2_records(
    df_incident: pd.DataFrame,
    org_data: dict,
) -> list[dict]:
    """Build 8 DevOps team incident tickets from a real incident cluster.

    Selects an assignment_group from the incident log that has >= 5 tickets
    in a compressed week window and maps them to Meridian's DevOps team.
    Falls back to synthetic records if no such cluster is found.
    """
    org_id = org_data["org_id"]
    devops_team = org_data["teams"]["devops"]
    devops_members = [
        org_data["users"]["alex.rivera@meridian-tech.com"],
        org_data["users"]["priya.patel@meridian-tech.com"],
        org_data["users"]["marcus.johnson@meridian-tech.com"],
        org_data["users"]["sofia.chen@meridian-tech.com"],
        org_data["users"]["ryan.osei@meridian-tech.com"],
    ]
    member_emails = [
        "alex.rivera@meridian-tech.com",
        "priya.patel@meridian-tech.com",
        "marcus.johnson@meridian-tech.com",
        "sofia.chen@meridian-tech.com",
        "ryan.osei@meridian-tech.com",
    ]

    # Try to find a real cluster from Kaggle data
    if "assignment_group" in df_incident.columns:
        counts = df_incident["assignment_group"].value_counts()
        cluster_group = counts[counts >= 5].index[0] if (counts >= 5).any() else None
        cluster_rows = (
            df_incident[df_incident["assignment_group"] == cluster_group].head(8)
            if cluster_group else pd.DataFrame()
        )
    else:
        cluster_rows = pd.DataFrame()

    # Build records: use real descriptions when available, fall back synthetic
    subjects = [
        ("Deployment pipeline failing — auth service CI broken", "deployment", "high"),
        ("Cannot push to production — permissions denied after key rotation", "access", "medium"),
        ("Auth tokens expiring too fast — LDAP session timeout reduced?", "auth", "high"),
        ("Staging environment login loop — redirects after each auth attempt", "auth", "medium"),
        ("Jenkins auth plugin failing — cannot trigger builds", "deployment", "medium"),
        ("Docker registry authentication rejected — new certs not deployed", "access", "high"),
        ("SSO redirect broken — all devops tools affected post-update", "auth", "critical"),
        ("Git webhooks failing — auth header rejected by new service", "deployment", "medium"),
    ]

    records = []
    for i, (title, category, priority) in enumerate(subjects):
        member_idx = i % len(devops_members)
        member = devops_members[member_idx]
        email = member_emails[member_idx]

        real_desc = ""
        if len(cluster_rows) > i and "subcategory" in cluster_rows.columns:
            row = cluster_rows.iloc[i]
            real_desc = (
                f"[Source: incident #{row.get('number', 'N/A')}, "
                f"group: {row.get('assignment_group', 'N/A')}, "
                f"category: {row.get('category', 'N/A')}/{row.get('subcategory', 'N/A')}] "
            )

        records.append(
            _record(
                local_id=f"C2-T{i+1}",
                title=f"{devops_members[member_idx]['full_name'] if hasattr(devops_members[member_idx], '__getitem__') else 'DevOps'} — {title}",
                content=(
                    f"{real_desc}"
                    f"Reporter: {org_data['users'][email]['full_name']} (DevOps team). "
                    f"Issue: {title}. "
                    "This is part of a cluster of auth/deployment failures affecting the entire "
                    "DevOps team this week. Related to recent certificate rotation on Monday. "
                    f"Priority: {priority}."
                ),
                source_type="portal",
                author_role="engineer",
                tags=["devops", category, "cert-rotation", "week-12", "case-2"],
                days_ago=7 - i,  # spread across the week
                scope="team",
                org_id=org_id,
                team_slug="devops",
                team_id=devops_team["team_id"],
                user_email=email,
                user_id=org_data["users"][email]["user_id"],
                ticket_id=f"INC-2026-03{20+i:02d}",
                priority=priority,
                category=f"engineering/{category}",
                resolution=None,
                case=2,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Case 3 — The Fix That Wasn't (placeholder builders for Copilot)
# ---------------------------------------------------------------------------

def build_case3_records(
    df_support: pd.DataFrame,
    df_incident: pd.DataFrame,
    org_data: dict,
) -> list[dict]:
    """Build records for Case 3: a resolved incident that keeps generating tickets.

    CASE 3 SUMMARY
    --------------
    DevOps closes incident INC-2026-0318 on Monday morning: "auth fixed".
    Support continues receiving the same complaint Tue-Thu from customers.
    Ninai detects the conflict: resolved=True in one memory, ongoing in others.

    Agents: ConflictDetectionAgent → CredibilityAgent → CausalReasoningAgent
            → AdaptiveConflictResolutionAgent → NarrativeSynthesisAgent

    Record mix:
    - 1 "resolved" incident record from DevOps (monday)
    - 4 ongoing customer support tickets (tue-thu)
    - 1 security team note confirming the fix was incomplete

    Copilot instructions:
    - Follow the exact same _record() pattern as build_case1_records and build_case2_records
    - The "resolved" DevOps record should have resolution="auth service restored, JWT keys rotated"
      and days_ago=4 (Monday)
    - The 4 customer tickets should reference the same auth symptoms, days_ago=3,2,2,1
    - The security note should say "key rotation applied to primary only, replica not updated"
      and days_ago=2 (Wednesday)
    - Tags: ["conflict", "auth", "false-resolution", "case-3"] on all records
    - team_slug: "devops" for the resolution record, "support-l1" for customer tickets,
      "security" for the security note
    """
    org_id = org_data["org_id"]
    devops_team = org_data["teams"]["devops"]
    security_team = org_data["teams"]["security"]
    l1_team = org_data["teams"]["support-l1"]

    alex_email = "alex.rivera@meridian-tech.com"
    james_email = "james.kim@meridian-tech.com"
    carlos_email = "carlos.mendez@meridian-tech.com"
    sarah_email = "sarah.okafor@meridian-tech.com"
    mei_email = "mei.zhang@meridian-tech.com"

    alex = org_data["users"][alex_email]
    james = org_data["users"][james_email]
    carlos = org_data["users"][carlos_email]
    sarah = org_data["users"][sarah_email]
    mei = org_data["users"][mei_email]

    # Filter for auth/login tickets first; fall back to "Technical issue" type
    # (actual Kaggle ticket_type values are "Technical issue", not "software")
    auth_cx_rows = df_support[
        df_support.get("ticket_description", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .str.contains(r"login|password|access|auth|account|sign.in", na=False, regex=True)
    ].head(4)
    support_rows = auth_cx_rows if len(auth_cx_rows) >= 4 else df_support[
        df_support.get("ticket_type", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .str.contains("technical", na=False)
    ].head(4)
    support_desc = [str(v) for v in support_rows.get("ticket_description", pd.Series(dtype=str)).tolist()]
    fallback_desc = [
        "Customer cannot log in and sees authentication error after recent maintenance.",
        "Follow-up: login still failing after reset. Session appears invalid.",
        "Third report today: authentication rejected even after cache clear.",
        "Issue persists for multiple customers since Monday deployment.",
    ]
    while len(support_desc) < 4:
        support_desc.append(fallback_desc[len(support_desc)])

    resolved_rows = df_incident[
        df_incident.get("incident_state", pd.Series(dtype=str))
        .astype(str)
        .str.lower()
        .str.contains(r"resolv|clos", na=False, regex=True)
    ].head(1)
    if len(resolved_rows) > 0:
        rr = resolved_rows.iloc[0]
        close_ref = str(rr.get("number", "INC-2026-0318"))
        close_cat = str(rr.get("category", "auth"))
        close_sub = str(rr.get("subcategory", "jwt"))
        close_desc = (
            f"DevOps close note from {close_ref}: auth service marked restored, "
            f"JWT keys rotated, category {close_cat}/{close_sub}."
        )
    else:
        close_ref = "INC-2026-0318"
        close_desc = "DevOps close note: auth service restored, JWT keys rotated, all systems healthy."

    common_tags = ["conflict", "auth", "false-resolution", "case-3"]

    records = [
        _record(
            local_id="C3-CLOSE",
            title="DevOps close note: auth fixed",
            content=(
                f"{close_desc} Resolution entered by Alex Rivera. "
                "Incident marked closed on Monday morning."
            ),
            source_type="portal",
            author_role="engineer",
            tags=common_tags,
            days_ago=4,
            scope="team",
            org_id=org_id,
            team_slug="devops",
            team_id=devops_team["team_id"],
            user_email=alex_email,
            user_id=alex["user_id"],
            ticket_id=close_ref,
            priority="high",
            category="engineering/auth",
            resolution="auth service restored, JWT keys rotated",
            case=3,
        ),
        _record(
            local_id="C3-CX1",
            title="Customer still blocked after closure",
            content=(
                f"Customer report (Tuesday): {support_desc[0][:220]} "
                "System still returns authentication error after Monday fix."
            ),
            source_type="phone",
            author_role="support_agent",
            tags=common_tags,
            days_ago=3,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email=carlos_email,
            user_id=carlos["user_id"],
            ticket_id="TKT-2026-0321",
            priority="medium",
            category="support/auth",
            resolution=None,
            case=3,
        ),
        _record(
            local_id="C3-CX2",
            title="Second day complaint: auth failure persists",
            content=(
                f"Customer follow-up (Wednesday AM): {support_desc[1][:220]} "
                "Issue references Monday closure but behavior unchanged."
            ),
            source_type="portal",
            author_role="support_agent",
            tags=common_tags,
            days_ago=2,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email=sarah_email,
            user_id=sarah["user_id"],
            ticket_id="TKT-2026-0322",
            priority="medium",
            category="support/auth",
            resolution=None,
            case=3,
        ),
        _record(
            local_id="C3-CX3",
            title="Third complaint confirms unresolved issue",
            content=(
                f"Customer report (Wednesday PM): {support_desc[2][:220]} "
                "Support confirms multiple customers still blocked after declared fix."
            ),
            source_type="email",
            author_role="support_agent",
            tags=common_tags,
            days_ago=2,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email=mei_email,
            user_id=mei["user_id"],
            ticket_id="TKT-2026-0323",
            priority="high",
            category="support/auth",
            resolution=None,
            case=3,
        ),
        _record(
            local_id="C3-CX4",
            title="Fourth complaint escalates impact count",
            content=(
                f"Customer escalation (Thursday): {support_desc[3][:220]} "
                "Seven customers affected since Monday despite closure."
            ),
            source_type="phone",
            author_role="support_agent",
            tags=common_tags,
            days_ago=1,
            scope="team",
            org_id=org_id,
            team_slug="support-l1",
            team_id=l1_team["team_id"],
            user_email=carlos_email,
            user_id=carlos["user_id"],
            ticket_id="TKT-2026-0324",
            priority="high",
            category="support/auth",
            resolution=None,
            case=3,
        ),
        _record(
            local_id="C3-SEC",
            title="Security note: replica key mismatch",
            content=(
                "Security investigation: key rotation applied to primary auth node only. "
                "Replica us-east-1b still uses old ES256 key. LDAP users routed to replica "
                "fail token validation and enter login loop."
            ),
            source_type="slack",
            author_role="engineer",
            tags=common_tags + ["root-cause"],
            days_ago=2,
            scope="team",
            org_id=org_id,
            team_slug="security",
            team_id=security_team["team_id"],
            user_email=james_email,
            user_id=james["user_id"],
            ticket_id="SEC-2026-0412",
            priority="critical",
            category="security/auth",
            resolution=None,
            case=3,
        ),
    ]

    return records


# ---------------------------------------------------------------------------
# Case 4 — The Monday Morning Crash (placeholder builders for Copilot)
# ---------------------------------------------------------------------------

def build_case4_records(
    df_incident: pd.DataFrame,
    org_data: dict,
    n_tickets: int = 50,
) -> list[dict]:
    """Build 50 incident records simulating a company-wide auth spike.

    CASE 4 SUMMARY
    --------------
    47 tickets arrive between 09:00–11:00 on a Monday. All involve login or
    access failures. Ninai's AnomalyDetectionAgent detects the spike (+380%),
    OrgAttentionAgent escalates, PlaybookAgent matches MAJOR-INCIDENT-RESPONSE.

    Kaggle source: incident_event_log.csv filtered by category containing
    'access' or 'software', compressed into a 2-hour synthetic window.

    Agents: AnomalyDetectionAgent → TemporalReasoningAgent → OrgAttentionAgent
            → GoalDecompositionAgent → PlaybookAgent → NarrativeSynthesisAgent

    Copilot instructions:
    - Filter df_incident for rows where category contains 'access' or 'software'
      (case-insensitive), take first n_tickets rows
    - Map each row to a _record() with:
        days_ago = 0 (all today)
        source_type = row["contact_type"] or "portal"
        priority = row["priority"] or "high"
        scope = "organization"
        tags = ["auth", "access", "spike", "monday-crash", "case-4"]
        team_slug = "sre"  (SRE owns incident response)
    - Distribute across 3 SRE members (tom.bradley, aisha.hassan, luca.ferrari)
    - Add 10 "baseline" records with days_ago ranging 7-30 so the anomaly
      detector has a comparison baseline
    """
    org_id = org_data["org_id"]
    sre_team = org_data["teams"]["sre"]
    sre_emails = [
        "tom.bradley@meridian-tech.com",
        "aisha.hassan@meridian-tech.com",
        "luca.ferrari@meridian-tech.com",
    ]
    sre_users = [org_data["users"][e] for e in sre_emails]

    if "category" in df_incident.columns:
        filtered = df_incident[
            df_incident["category"].astype(str).str.lower().str.contains(r"access|software", na=False, regex=True)
        ].reset_index(drop=True)
    else:
        filtered = df_incident.reset_index(drop=True)

    if len(filtered) == 0:
        filtered = df_incident.reset_index(drop=True)

    # Ensure enough rows by cycling when source is smaller than needed.
    needed = n_tickets + 10
    rows = [filtered.iloc[i % len(filtered)] for i in range(max(needed, 1))]

    hero = _record(
        local_id="C4-HERO",
        title="LIVE: Auth/access ticket spike - 47 tickets in 2 hours",
        content=(
            "SRE monitoring alert: ticket intake reached 47 in two hours (09:00-11:00 UTC). "
            "Symptoms: login failures, access denied, authentication errors. "
            "Affected departments: Engineering, Operations, Finance. "
            "Baseline: 12 tickets/day. Extrapolated current rate: 280/day."
        ),
        source_type="system",
        author_role="engineer",
        tags=["spike", "auth", "company-wide", "monday-crash", "case-4", "sre"],
        days_ago=0,
        scope="organization",
        org_id=org_id,
        team_slug="sre",
        team_id=sre_team["team_id"],
        user_email=sre_emails[0],
        user_id=sre_users[0]["user_id"],
        ticket_id="ALERT-2026-0330",
        priority="critical",
        category="sre/incident",
        resolution=None,
        case=4,
    )

    spike_records: list[dict[str, Any]] = []
    for i in range(n_tickets):
        row = rows[i]
        owner_idx = i % len(sre_users)
        owner_email = sre_emails[owner_idx]
        owner = sre_users[owner_idx]
        category = str(row.get("category", "access"))
        subcategory = str(row.get("subcategory", "login"))
        ref = str(row.get("number", f"S{i+1:04d}"))
        priority = str(row.get("priority", "high") or "high")
        source_type = str(row.get("contact_type", "portal") or "portal")

        spike_records.append(
            _record(
                local_id=f"C4-S{i+1:02d}",
                title=f"Access failure burst ticket {i+1} of {n_tickets}",
                content=(
                    f"[Kaggle ref: {ref}] Reporter: {owner['full_name']} (SRE on-call). "
                    f"Issue: {category}/{subcategory} failure during Monday spike window. "
                    "Part of company-wide auth disruption affecting multiple departments."
                ),
                source_type=source_type,
                author_role="engineer",
                tags=["auth", "access", "spike", "monday-crash", "case-4"],
                days_ago=0,
                scope="organization",
                org_id=org_id,
                team_slug="sre",
                team_id=sre_team["team_id"],
                user_email=owner_email,
                user_id=owner["user_id"],
                ticket_id=f"INC-{ref}",
                priority=priority,
                category=f"sre/{category}",
                resolution=None,
                case=4,
            )
        )

    baseline_days = [7, 10, 14, 18, 22, 25, 28, 30, 35, 40]
    baseline_records: list[dict[str, Any]] = []
    for i, day in enumerate(baseline_days):
        row = rows[n_tickets + i]
        owner_idx = i % len(sre_users)
        owner_email = sre_emails[owner_idx]
        owner = sre_users[owner_idx]
        category = str(row.get("category", "access"))
        subcategory = str(row.get("subcategory", "login"))
        ref = str(row.get("number", f"B{i+1:04d}"))

        baseline_records.append(
            _record(
                local_id=f"C4-B{i+1:02d}",
                title=f"Baseline auth/access ticket {i+1}",
                content=(
                    f"Historical baseline sample [Kaggle ref: {ref}] {category}/{subcategory}. "
                    "Normal pre-spike ticket volume for comparison."
                ),
                source_type=str(row.get("contact_type", "portal") or "portal"),
                author_role="engineer",
                tags=["baseline", "auth", "case-4"],
                days_ago=day,
                scope="organization",
                org_id=org_id,
                team_slug="sre",
                team_id=sre_team["team_id"],
                user_email=owner_email,
                user_id=owner["user_id"],
                ticket_id=f"BASE-{ref}",
                priority=str(row.get("priority", "medium") or "medium"),
                category=f"sre/{category}",
                resolution=None,
                case=4,
            )
        )

    return [hero] + spike_records + baseline_records


# ---------------------------------------------------------------------------
# Case 5 — The Slow Boil (placeholder builders for Copilot)
# ---------------------------------------------------------------------------

def build_case5_records(
    df_incident: pd.DataFrame,
    org_data: dict,
) -> list[dict]:
    """Build records showing a 5-day exponential ticket volume increase.

    CASE 5 SUMMARY
    --------------
    A specific subcategory (storage/OS) shows 2→4→7→12→19 tickets over 5 days.
    No single day looks alarming. Ninai's TemporalReasoningAgent detects the
    trend on day 3, PredictiveMonitorAgent forecasts SLA breach by day 7,
    AutonomousGoalGenerationAgent creates a proactive investigation goal.

    Kaggle source: incident_event_log.csv, filter by one subcategory with
    enough rows, distribute across 5 days with increasing counts.

    Agents: TemporalReasoningAgent → PredictiveMonitorAgent
            → AnomalyDetectionAgent → AutonomousGoalGenerationAgent
            → MetaCognitivePlanningAgent → NarrativeSynthesisAgent

    Copilot instructions:
    - Pick a subcategory from df_incident with >= 44 rows (to cover 2+4+7+12+19)
    - Create 5 buckets: day_counts = [2, 4, 7, 12, 19]
    - For each bucket b at index i (0-4):
        days_ago = 5 - i  (day 1 is 5 days ago, day 5 is today)
        Create b records with that days_ago value
        Assign to SRE or devops teams alternating
    - All records get tags = ["storage", "degradation", "slow-boil", "case-5"]
    - scope = "organization"
    - The content should mention the subcategory name and "recurring since last week"
    - Total records = 44
    """
    org_id = org_data["org_id"]
    sre_team = org_data["teams"]["sre"]
    devops_team = org_data["teams"]["devops"]

    sre_emails = [
        "tom.bradley@meridian-tech.com",
        "aisha.hassan@meridian-tech.com",
        "luca.ferrari@meridian-tech.com",
    ]
    devops_emails = [
        "alex.rivera@meridian-tech.com",
        "priya.patel@meridian-tech.com",
        "marcus.johnson@meridian-tech.com",
    ]

    # The Kaggle incident dataset uses anonymised subcategory names ("Subcategory 174" etc.)
    # We pick the most frequent eligible subcategory for data distribution realism,
    # but display it as "storage/OS" to match the slow-boil narrative.
    DISPLAY_SUBCATEGORY = "storage/OS"
    if "subcategory" in df_incident.columns:
        sub_counts = df_incident["subcategory"].astype(str).value_counts()
        eligible = sub_counts[sub_counts >= 44]
        chosen_subcategory = str(eligible.index[0]) if len(eligible) else "storage"
        pool = df_incident[df_incident["subcategory"].astype(str) == chosen_subcategory].reset_index(drop=True)
    else:
        chosen_subcategory = "storage"
        pool = df_incident.reset_index(drop=True)

    if len(pool) == 0:
        pool = df_incident.reset_index(drop=True)

    day_counts = [2, 4, 7, 12, 19]
    total_needed = sum(day_counts)
    rows = [pool.iloc[i % len(pool)] for i in range(max(total_needed, 1))]

    records: list[dict[str, Any]] = []
    row_idx = 0
    for day_idx, count in enumerate(day_counts):
        days_ago = 5 - day_idx
        for j in range(count):
            row = rows[row_idx]
            row_idx += 1

            if (day_idx + j) % 2 == 0:
                team_slug = "sre"
                team_id = sre_team["team_id"]
                email = sre_emails[(day_idx + j) % len(sre_emails)]
            else:
                team_slug = "devops"
                team_id = devops_team["team_id"]
                email = devops_emails[(day_idx + j) % len(devops_emails)]

            user = org_data["users"][email]
            number = str(row.get("number", f"SB{row_idx:04d}"))
            # Use the display name for the story narrative; Kaggle ref is in brackets
            subcategory = DISPLAY_SUBCATEGORY

            records.append(
                _record(
                    local_id=f"C5-D{day_idx+1}-T{j+1:02d}",
                    title=f"Day {day_idx+1} {subcategory} degradation signal",
                    content=(
                        f"[Kaggle ref: {number}] Recurring {subcategory} issue. "
                        "Similar reports have appeared since last week with increasing daily volume."
                    ),
                    source_type=str(row.get("contact_type", "portal") or "portal"),
                    author_role="engineer",
                    tags=["storage", "degradation", "slow-boil", "case-5"],
                    days_ago=days_ago,
                    scope="organization",
                    org_id=org_id,
                    team_slug=team_slug,
                    team_id=team_id,
                    user_email=email,
                    user_id=user["user_id"],
                    ticket_id=f"TREND-{number}",
                    priority=str(row.get("priority", "medium") or "medium"),
                    category=f"predictive/{DISPLAY_SUBCATEGORY}",
                    resolution=None,
                    case=5,
                )
            )

    return records
