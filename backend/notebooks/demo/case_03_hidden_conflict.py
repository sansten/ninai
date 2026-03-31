"""Case 3: The Fix That Wasn't — medium complexity.

SCENARIO
--------
Monday 09:00 — DevOps engineer Alex Rivera closes incident INC-2026-0318:
  "Auth service restored. JWT keys rotated. All systems healthy."

Tuesday through Thursday — Support L1 keeps receiving the exact same
complaint from customers: "I still cannot log in."

Wednesday 14:00 — Security lead James Kim adds a quiet internal note:
  "Key rotation script ran on primary node only. Replica at us-east-1b
  was not updated. LDAP users authenticating against replica hit the old key."

Nobody connected these three data sources. The DevOps ticket was closed.
The customer tickets were treated as new issues. James's note sat unread.

Ninai reads all three sources, detects the contradiction between
"auth fixed" and "login still broken", scores source credibility
(customer complaints > closed ticket), traces the causal path
(primary-only rotation → replica mismatch → LDAP loop), identifies
escalation targets, and writes a briefing that surfaces the truth.

COMPLEXITY: MEDIUM
KEY INSIGHT: "The system said it was fixed. The memory says it wasn't."

AGENT PIPELINE
--------------
ConflictDetectionAgent        — finds contradiction between close note and support tickets
CredibilityAgent              — customer tickets score higher than DevOps close note
CausalReasoningAgent          — traces: key rotation → primary only → replica mismatch → login loop
AdaptiveConflictResolutionAgent — identifies escalation targets (security lead, DevOps lead, L2)
NarrativeSynthesisAgent       — writes the CSR briefing

RECORDS (6 total)
-----------------
C3-CLOSE  — DevOps close note, days_ago=4 (Monday), team=devops
            content: "Auth fixed, JWT keys rotated, all healthy"
            resolution: "auth service restored, JWT keys rotated"
            This is the FALSE positive — it will be contradicted by others.

C3-CX1    — Customer ticket, days_ago=3 (Tuesday), team=support-l1
            author: carlos.mendez@meridian-tech.com
            content: "Cannot log in — still getting authentication error"

C3-CX2    — Customer ticket, days_ago=2 (Wednesday AM), team=support-l1
            author: sarah.okafor@meridian-tech.com
            content: "Same issue as yesterday — auth failure persists"

C3-CX3    — Customer ticket, days_ago=2 (Wednesday PM), team=support-l1
            author: mei.zhang@meridian-tech.com
            content: "Third customer today with login failure after 'fix'"

C3-CX4    — Customer ticket, days_ago=1 (Thursday), team=support-l1
            author: carlos.mendez@meridian-tech.com
            content: "Issue ongoing — 7 customers affected this week since Monday"

C3-SEC    — Security internal note, days_ago=2 (Wednesday), team=security
            author: james.kim@meridian-tech.com  ← THIS IS THE HERO RECORD
            content: "Key rotation applied to primary node only. Replica us-east-1b
                      still using old ES256 key. LDAP users on replica hit login loop."
            This is the smoking gun — highest credibility, root cause identified.

HERO RECORD: C3-SEC
  The Security note is the hero because it has the root cause. The CSR briefing
  should surface James Kim's finding to the support team immediately.

KAGGLE DATA USAGE
-----------------
C3-CLOSE : Take 1 row from incident_event_log.csv where incident_state contains
           "resolved" or "closed". Use its description as the base for the close note.
           If no such row, use the synthetic content above verbatim.

C3-CX1-4 : Take 4 rows from customer_support_tickets.csv where ticket_type is
           "Software" or "Technical". Use their ticket_description as the base
           content, then prepend the customer persona text.

C3-SEC   : Always synthetic — the security note has no Kaggle equivalent.

HOW TO IMPLEMENT
----------------
Follow case_01_repeat_caller.py exactly. The structure is identical:

  1. build_records()      — constructs the 6 record dicts using _record() from data_prep.py
  2. run_pipeline()       — calls _run_pipeline() with AGENT_PIPELINE and cross_enrichment
  3. format_comparison()  — calls render_csr_card() + render_comparison_table() + returns str
  4. get_cells()          — returns list of (cell_type, source) tuples (6 cells)

CROSS_ENRICHMENT for run_pipeline()
-------------------------------------
Inject this into the hero record C3-SEC before running agents.
These are the facts Ninai has accumulated by reading all 6 records:

    cross_enrichment = {
        "canonical_entity": "Auth Service / JWT Key Rotation",
        "resolved_entities": [
            {
                "name": "Alex Rivera",
                "email": "alex.rivera@meridian-tech.com",
                "team": "devops",
                "role": "closed incident — false resolution",
            },
            {
                "name": "James Kim",
                "email": "james.kim@meridian-tech.com",
                "team": "security",
                "role": "root cause author",
            },
        ],
        "conflict_count": 2,
        "unresolved_conflicts": [
            "DevOps close note (day -4): auth fixed — but 4 customer tickets show ongoing failure",
            "Key rotation log: primary updated — replica us-east-1b skipped",
        ],
        "causal_chain": [
            "JWT key rotation triggered by deployment INC-2026-0318",
            "Rotation script applied to primary auth node only",
            "Replica node us-east-1b still holds old ES256 key",
            "LDAP users authenticating against replica → token validation failure → login loop",
        ],
        "escalation_targets": [
            "james.kim@meridian-tech.com",    # security lead — owns the fix
            "alex.rivera@meridian-tech.com",  # devops lead   — reopens incident
            "oliver.smith@meridian-tech.com", # support L2    — owns customer escalation
        ],
        "credibility_score": 0.85,      # security note from named engineer > closed ticket
        "source_tier": "primary",
        "uncertainty_level": "low",     # root cause is clearly identified
        "review_recommended": True,
        "conflict_resolution_strategy": "reopen_incident",
        "temporal_anchor": "2026-03-24T14:00:00Z",  # Wednesday security note
        "sequence_position": 3,         # 3rd in the week's sequence of events
    }

COMPARISON QUESTIONS (use in format_comparison)
-------------------------------------------------
questions = [
    "Was the incident really resolved on Monday?",
    "Why are customers still reporting failures?",
    "What is the actual root cause?",
    "Who is responsible for the incomplete fix?",
    "Who needs to be contacted right now?",
]

raw_answers = [
    "Yes — ticket INC-2026-0318 is marked Closed",
    "Unknown — these look like new separate tickets",
    "Unknown — each ticket just says login fails",
    "Unknown — the DevOps team closed the ticket",
    "Unknown — no escalation target visible from the tickets",
]

# Ninai answers — derive from enrichment dict:
# - enrichment.get("conflict_count") for first answer
# - enrichment.get("causal_chain")[-1] for root cause
# - enrichment.get("escalation_targets") for contact list
ninai_answers = [
    "No — Ninai detected a conflict: 2 contradictions between the close note and live complaints",
    "Key rotation applied to primary only — replica us-east-1b still using old key",
    "LDAP users on us-east-1b replica → token validation failure → login loop",
    "DevOps (Alex Rivera) — rotation script skipped replica node",
    "James Kim (security), Alex Rivera (devops), Oliver Smith (support L2) — all three paged",
]

GET_CELLS STRUCTURE (6 cells — same as case_01 and case_02)
------------------------------------------------------------
Cell 1: markdown  — section header with 3-source conflict table:
    | Source | Claim | Credibility |
    |--------|-------|-------------|
    | DevOps close note (Mon) | "auth fixed" | Lower — issue recurred |
    | Customer tickets (Tue-Thu) | "login still broken" | Higher — repeated, corroborated |
    | Security note (Wed) | "replica not updated" | Highest — root cause identified |

Cell 2: code — build_records() call:
    from demo.case_03_hidden_conflict import build_records, run_pipeline, format_comparison
    c3_records = build_records({"support": df_support, "incident": df_incident}, org_data)
    for r in c3_records: print(f"  {r['id']}: {r['title'][:60]} (days_ago={r['days_ago']})")

Cell 3: code — ingest:
    c3_memory_ids = ingest_records(ninai_client, c3_records)

Cell 4: code — run_pipeline:
    c3_result = run_pipeline(c3_records, org_data)
    for agent_name, outputs in c3_result["agent_outputs"].get("C3-SEC", {}).items():
        print(f"  {agent_name}: {'ok' if 'error' not in outputs else 'ERROR'}")

Cell 5: code — display CSR card:
    from IPython.display import Markdown, display
    display(Markdown(format_comparison(c3_result)))

Cell 6: markdown — key insight table:
    | Without Ninai | With Ninai |
    |---|---|
    | INC-2026-0318 stays Closed | Incident reopened within minutes |
    | Customers keep calling | Root cause fixed: replica updated |
    | 3 teams unaware of each other | All 3 escalated simultaneously |

RETURN TYPE REFERENCE
----------------------
build_records()  -> list[dict]   — each dict has keys: id, title, content, source,
                                   role, scope, tags, days_ago, meta
                                   meta has: org_id, team_id, team, user_id, ticket_id,
                                   priority, category, resolution, case=3

run_pipeline()   -> dict         — keys: hero_id, hero, enrichments, agent_outputs
                                   enrichments["C3-SEC"] is the main enrichment dict

format_comparison() -> str       — markdown string (render with display(Markdown(...)))

get_cells()      -> list[tuple[str,str]]  — each tuple is ("markdown"|"code", source_string)
"""

from __future__ import annotations

from typing import Any

from demo.data_prep import build_case3_records, load_support, load_incident
from demo.agents import run_pipeline as _run_pipeline, ingest_records
from demo.formatters import render_csr_card, render_comparison_table

# Agent imports — these are the 5 agents for this case
from app.agents.conflict_detection_agent import ConflictDetectionAgent
from app.agents.credibility_agent import CredibilityAgent
from app.agents.causal_reasoning_agent import CausalReasoningAgent
from app.agents.adaptive_conflict_resolution_agent import AdaptiveConflictResolutionAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent

# ── Module constants (do not change) ─────────────────────────────────────────
CASE_NUMBER    = 3
CASE_TITLE     = "The Fix That Wasn't"
HERO_LOCAL_ID  = "C3-SEC"   # Security note — the smoking gun

AGENT_PIPELINE = [
    ConflictDetectionAgent,
    CredibilityAgent,
    CausalReasoningAgent,
    AdaptiveConflictResolutionAgent,
    NarrativeSynthesisAgent,
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 of 4: build_records
# ─────────────────────────────────────────────────────────────────────────────
def build_records(data: dict, org_data: dict) -> list[dict]:
    """Construct the 6 ticket records for Case 3.

    Parameters
    ----------
    data : dict
        Keys: "support" -> pd.DataFrame (customer_support_tickets.csv)
               "incident" -> pd.DataFrame (incident_event_log.csv)
        Both DataFrames are already loaded and normalised (lowercase column names).

    org_data : dict
        Returned by bootstrap.py subprocess. Structure:
        {
            "org_id": "uuid-string",
            "teams": {
                "devops":     {"team_id": "uuid", "name": "DevOps",     "division": "Engineering"},
                "security":   {"team_id": "uuid", "name": "Security",   "division": "Engineering"},
                "support-l1": {"team_id": "uuid", "name": "Support L1", "division": "Operations"},
                "support-l2": {"team_id": "uuid", "name": "Support L2", "division": "Operations"},
            },
            "users": {
                "alex.rivera@meridian-tech.com":   {"user_id": "uuid", "full_name": "Alex Rivera",   "team": "devops"},
                "james.kim@meridian-tech.com":     {"user_id": "uuid", "full_name": "James Kim",     "team": "security"},
                "carlos.mendez@meridian-tech.com": {"user_id": "uuid", "full_name": "Carlos Mendez", "team": "support-l1"},
                "sarah.okafor@meridian-tech.com":  {"user_id": "uuid", "full_name": "Sarah Okafor",  "team": "support-l1"},
                "mei.zhang@meridian-tech.com":     {"user_id": "uuid", "full_name": "Mei Zhang",     "team": "support-l1"},
                "oliver.smith@meridian-tech.com":  {"user_id": "uuid", "full_name": "Oliver Smith",  "team": "support-l2"},
                ...
            }
        }

    Returns
    -------
    list[dict]
        6 records in this exact order:
        [C3-CLOSE, C3-CX1, C3-CX2, C3-CX3, C3-CX4, C3-SEC]

        Each dict must have these exact keys (see _record() in data_prep.py):
        {
            "id":        str,        # e.g. "C3-CLOSE"
            "title":     str,        # short human-readable title
            "content":   str,        # full ticket/note text
            "source":    str,        # "portal" | "email" | "phone" | "slack"
            "role":      str,        # "engineer" | "support_agent" | "customer"
            "scope":     str,        # "team" | "organization"
            "tags":      list[str],  # e.g. ["auth", "conflict", "case-3"]
            "days_ago":  int,        # how many days ago (4, 3, 2, 2, 1, 2)
            "meta": {
                "org_id":    str,    # org_data["org_id"]
                "team_id":   str,    # org_data["teams"][team_slug]["team_id"]
                "team":      str,    # team slug e.g. "devops"
                "user_id":   str,    # org_data["users"][email]["user_id"]
                "ticket_id": str,    # e.g. "INC-2026-0318"
                "priority":  str,    # "high" | "medium" | "critical"
                "category":  str,    # e.g. "engineering/auth"
                "resolution": str | None,
                "case":      int,    # always 3
                "created_at": str,   # ISO timestamp string
            }
        }

    Implementation steps
    --------------------
    Step 1: Extract org/team/user IDs from org_data (same pattern as case_01 and case_02):
        org_id     = org_data["org_id"]
        devops_tid = org_data["teams"]["devops"]["team_id"]
        sec_tid    = org_data["teams"]["security"]["team_id"]
        l1_tid     = org_data["teams"]["support-l1"]["team_id"]
        alex_uid   = org_data["users"]["alex.rivera@meridian-tech.com"]["user_id"]
        james_uid  = org_data["users"]["james.kim@meridian-tech.com"]["user_id"]
        carlos_uid = org_data["users"]["carlos.mendez@meridian-tech.com"]["user_id"]
        sarah_uid  = org_data["users"]["sarah.okafor@meridian-tech.com"]["user_id"]
        mei_uid    = org_data["users"]["mei.zhang@meridian-tech.com"]["user_id"]

    Step 2: Pull Kaggle rows for the customer tickets (C3-CX1 through C3-CX4):
        sw_rows = data["support"][
            data["support"]["ticket_type"].str.lower().isin(["software","technical"])
        ].head(4)
        descriptions = list(sw_rows["ticket_description"].values)
        # If fewer than 4 rows, pad with synthetic text

    Step 3: Pull 1 Kaggle row for the close note (C3-CLOSE):
        resolved_rows = data["incident"][
            data["incident"].get("incident_state", pd.Series()).str.lower().str.contains("resolv|clos", na=False)
        ].head(1)
        close_base = resolved_rows.iloc[0]["number"] if len(resolved_rows) > 0 else "INC-2026-0318"

    Step 4: Build all 6 records using the _record() helper from data_prep.py.
        Call: from demo.data_prep import _record, _days_ago
        Then call _record(...) for each of the 6 records.

    Step 5: return [c3_close, c3_cx1, c3_cx2, c3_cx3, c3_cx4, c3_sec]
    """
    # TODO: implement following the steps above
    # Reference: build_case1_records() in data_prep.py (lines 140-220) is the exact pattern
    return build_case3_records(data["support"], data["incident"], org_data)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 of 4: run_pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(records: list[dict], org_data: dict) -> dict[str, Any]:
    """Run the Case 3 agent pipeline on all 6 records.

    Parameters
    ----------
    records : list[dict]
        The 6 records returned by build_records().
    org_data : dict
        The bootstrap org structure (same as passed to build_records()).

    Returns
    -------
    dict with keys:
        "hero_id"      : str          — always "C3-SEC"
        "hero"         : dict         — the C3-SEC record dict
        "enrichments"  : dict[str, dict]  — {local_id: enrichment_dict} for all 6 records
        "agent_outputs": dict[str, dict]  — {local_id: {agent_name: outputs}} for all 6 records

    Implementation steps
    --------------------
    Step 1: Define cross_enrichment exactly as the dict in the module docstring above.
        Pull the actual UUIDs from org_data for the user_id fields if desired,
        but the escalation_targets list is email strings, not UUIDs.

    Step 2: Call _run_pipeline() — already imported as `_run_pipeline` at the top of this file:
        return _run_pipeline(
            records=records,
            agent_classes=AGENT_PIPELINE,
            hero_local_id=HERO_LOCAL_ID,
            cross_record_enrichment=cross_enrichment,
        )

    _run_pipeline signature (from demo/agents.py):
        def run_pipeline(
            records: list[dict],
            agent_classes: list[Type[BaseAgent]],
            hero_local_id: str,
            *,
            cross_record_enrichment: dict | None = None,
        ) -> dict[str, Any]

    That's it — Step 1 + Step 2. Look at case_01.run_pipeline() for the exact pattern.
    """
    cross_enrichment = {
        "canonical_entity": "Auth Service / JWT Key Rotation",
        "resolved_entities": [
            {
                "name": "Alex Rivera",
                "email": "alex.rivera@meridian-tech.com",
                "team": "devops",
                "role": "closed incident — false resolution",
            },
            {
                "name": "James Kim",
                "email": "james.kim@meridian-tech.com",
                "team": "security",
                "role": "root cause author",
            },
        ],
        "conflict_count": 2,
        "unresolved_conflicts": [
            "DevOps close note (day -4): auth fixed — but 4 customer tickets show ongoing failure",
            "Key rotation log: primary updated — replica us-east-1b skipped",
        ],
        "causal_chain": [
            "JWT key rotation triggered by deployment INC-2026-0318",
            "Rotation script applied to primary auth node only",
            "Replica node us-east-1b still holds old ES256 key",
            "LDAP users authenticating against replica: token validation failure → login loop",
        ],
        "escalation_targets": [
            "james.kim@meridian-tech.com",
            "alex.rivera@meridian-tech.com",
            "oliver.smith@meridian-tech.com",
        ],
        "credibility_score": 0.85,
        "source_tier": "primary",
        "uncertainty_level": "low",
        "review_recommended": True,
        "conflict_resolution_strategy": "reopen_incident",
        "temporal_anchor": "2026-03-24T14:00:00Z",
        "sequence_position": 3,
    }
    return _run_pipeline(
        records=records,
        agent_classes=AGENT_PIPELINE,
        hero_local_id=HERO_LOCAL_ID,
        cross_record_enrichment=cross_enrichment,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3 of 4: format_comparison
# ─────────────────────────────────────────────────────────────────────────────
def format_comparison(result: dict) -> str:
    """Build the before/after markdown string for Case 3.

    Parameters
    ----------
    result : dict
        Returned by run_pipeline(). Use:
            enr  = result["enrichments"]["C3-SEC"]   # main enrichment dict
            hero = result["hero"]                     # C3-SEC record dict

    Returns
    -------
    str
        Markdown string combining:
        1. render_csr_card(3, CASE_TITLE, hero, enr, agent_outputs) — the briefing card
        2. "### Before vs After\\n\\n" + render_comparison_table(questions, raw, ninai)

        Both render_csr_card and render_comparison_table are already imported
        from demo.formatters at the top of this file.

    render_csr_card signature (from demo/formatters.py):
        render_csr_card(
            case_number: int,        # 3
            case_title: str,         # CASE_TITLE
            hero_record: dict,       # result["hero"]
            enrichment: dict,        # result["enrichments"]["C3-SEC"]
            agent_outputs: dict,     # result["agent_outputs"].get("C3-SEC", {})
        ) -> str

    render_comparison_table signature (from demo/formatters.py):
        render_comparison_table(
            questions: list[str],
            raw_answers: list[str],
            ninai_answers: list[str],
        ) -> str

    Implementation steps
    --------------------
    Step 1: Extract enr and hero from result (same as case_01.format_comparison)
    Step 2: Pull display values from enr:
        conflict_count = enr.get("conflict_count", 2)
        causal_root    = (enr.get("causal_chain") or ["unknown"])[-1]
        escalation     = enr.get("escalation_targets") or []
        tone           = enr.get("tone", "urgent").upper()
    Step 3: Build ninai_answers list using the values from step 2
    Step 4: Call render_csr_card() + render_comparison_table()
    Step 5: Return  csr_card_str + "\\n\\n### Before vs After\\n\\n" + table_str

    Use these exact questions and raw_answers:

    questions = [
        "Was the incident really resolved on Monday?",
        "Why are customers still reporting failures?",
        "What is the actual root cause?",
        "Who is responsible for the incomplete fix?",
        "Who needs to be contacted right now?",
    ]
    raw_answers = [
        "Yes — ticket INC-2026-0318 is marked Closed",
        "Unknown — these look like separate new tickets",
        "Unknown — each ticket just says login fails",
        "Unknown — the DevOps team closed the ticket",
        "Unknown — no escalation target visible",
    ]
    """
    enr = result["enrichments"].get(HERO_LOCAL_ID, {})
    hero = result["hero"]

    conflict_count = int(enr.get("conflict_count", 2) or 2)
    causal_root = (enr.get("causal_chain") or ["unknown"])[-1]
    escalation = enr.get("escalation_targets") or []
    escalation_str = ", ".join(escalation) if escalation else "No escalation targets identified"

    questions = [
        "Was the incident really resolved on Monday?",
        "Why are customers still reporting failures?",
        "What is the actual root cause?",
        "Who is responsible for the incomplete fix?",
        "Who needs to be contacted right now?",
    ]
    raw_answers = [
        "Yes - ticket INC-2026-0318 is marked Closed",
        "Unknown - these look like separate new tickets",
        "Unknown - each ticket just says login fails",
        "Unknown - the DevOps team closed the ticket",
        "Unknown - no escalation target visible",
    ]
    ninai_answers = [
        f"No - Ninai detected {conflict_count} contradictions against the close note",
        "Primary-only key rotation left replica us-east-1b on old key",
        causal_root,
        "DevOps ownership: rotation script skipped replica coverage",
        escalation_str,
    ]

    csr_card = render_csr_card(
        case_number=CASE_NUMBER,
        case_title=CASE_TITLE,
        hero_record=hero,
        enrichment=enr,
        agent_outputs=result["agent_outputs"].get(HERO_LOCAL_ID, {}),
    )
    table = render_comparison_table(questions, raw_answers, ninai_answers)
    return f"{csr_card}\n\n### Before vs After\n\n{table}"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4 of 4: get_cells
# ─────────────────────────────────────────────────────────────────────────────
def get_cells(org_data: dict) -> list[tuple[str, str]]:
    """Return the 6 notebook cells for Case 3.

    Parameters
    ----------
    org_data : dict
        Bootstrap org structure. Used only for display strings (team names, etc.).
        The placeholder dict in build_notebook.py is passed at build time;
        real UUIDs are passed at notebook runtime.

    Returns
    -------
    list[tuple[str, str]]
        Exactly 6 tuples: [("markdown", src), ("code", src), ...]
        Cell order:
            0: markdown  — section header with 3-source conflict table
            1: code      — build_records() call
            2: code      — ingest_records() call
            3: code      — run_pipeline() call + agent status printout
            4: code      — display(Markdown(format_comparison(result)))
            5: markdown  — key insight table + > quote

    Implementation steps
    --------------------
    Follow case_02.get_cells() exactly — it returns 6 tuples in the same order.

    Cell 0 (markdown) must include this 3-source conflict table:
        | Source | Claim | Credibility |
        |--------|-------|-------------|
        | DevOps close note (Monday) | "auth fixed, JWT keys rotated" | Lower — issue recurred |
        | Customer tickets (Tue–Thu) | "login still broken" | Higher — corroborated, repeated |
        | Security note (Wednesday) | "replica node NOT updated" | Highest — root cause found |

    Cell 1 (code) must contain:
        from demo.case_03_hidden_conflict import build_records, run_pipeline, format_comparison
        c3_records = build_records({"support": df_support, "incident": df_incident}, org_data)
        print(f"Case 3: {len(c3_records)} records built")
        for r in c3_records:
            print(f"  {r['id']}: {r['title'][:60]} (days_ago={r['days_ago']})")

    Cell 2 (code) must contain:
        from demo.agents import ingest_records
        c3_memory_ids = ingest_records(ninai_client, c3_records)
        print(f"Case 3 ingested: {len(c3_memory_ids)} records")

    Cell 3 (code) must contain:
        c3_result = run_pipeline(c3_records, org_data)
        print("Agents run on C3-SEC (security note — hero ticket):")
        for agent_name, outputs in c3_result["agent_outputs"].get("C3-SEC", {}).items():
            status = "ok" if "error" not in outputs else "ERROR"
            print(f"  {agent_name}: {status}")

    Cell 4 (code) must contain:
        from IPython.display import Markdown, display
        display(Markdown(format_comparison(c3_result)))

    Cell 5 (markdown) must contain the key insight table:
        | Without Ninai | With Ninai |
        |---|---|
        | INC-2026-0318 stays Closed | Incident reopened: conflict detected |
        | Customers keep calling Tue–Thu | Root cause surfaced on Wednesday |
        | 3 teams unaware of each other | Security + DevOps + Support L2 all paged |
        | Replica node runs old key forever | Replica updated within the hour |
        And: > **Key insight — Case 3:** ...
    """
    return [
        (
            "markdown",
            "\n".join([
                "---",
                "## Case 3: The Fix That Wasn't",
                "",
                "> **Complexity:** Medium | **Agents:** 5 | **Hero:** C3-SEC (security note)",
                "",
                "Three data sources disagree, and Ninai reconciles them in one briefing.",
                "",
                "| Source | Claim | Credibility |",
                "|--------|-------|-------------|",
                "| DevOps close note (Monday) | \"auth fixed, JWT keys rotated\" | Lower - issue recurred |",
                "| Customer tickets (Tue-Thu) | \"login still broken\" | Higher - corroborated, repeated |",
                "| Security note (Wednesday) | \"replica node NOT updated\" | Highest - root cause found |",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.case_03_hidden_conflict import build_records, run_pipeline, format_comparison",
                "c3_records = build_records({'support': df_support, 'incident': df_incident}, org_data)",
                "print(f'Case 3: {len(c3_records)} records built')",
                "for r in c3_records:",
                "    print(f\"  {r['id']}: {r['title'][:60]} (days_ago={r['days_ago']})\")",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.agents import ingest_records",
                "c3_memory_ids = ingest_records(ninai_client, c3_records)",
                "print(f'Case 3 ingested: {len(c3_memory_ids)} records')",
            ]),
        ),
        (
            "code",
            "\n".join([
                "c3_result = run_pipeline(c3_records, org_data)",
                "print('Agents run on C3-SEC (security note - hero ticket):')",
                "for agent_name, outputs in c3_result['agent_outputs'].get('C3-SEC', {}).items():",
                "    status = 'ok' if 'error' not in outputs else 'ERROR'",
                "    print(f'  {agent_name}: {status}')",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from IPython.display import Markdown, display",
                "display(Markdown(format_comparison(c3_result)))",
            ]),
        ),
        (
            "markdown",
            "\n".join([
                "### Key Insight - Case 3",
                "",
                "| Without Ninai | With Ninai |",
                "|---|---|",
                "| INC-2026-0318 stays Closed | Incident reopened: conflict detected |",
                "| Customers keep calling Tue-Thu | Root cause surfaced on Wednesday |",
                "| 3 teams unaware of each other | Security + DevOps + Support L2 all paged |",
                "| Replica node runs old key forever | Replica updated within the hour |",
                "",
                "> **Key insight - Case 3:** The ticket said \"fixed\", but cross-source memory proved otherwise.",
            ]),
        ),
    ]
