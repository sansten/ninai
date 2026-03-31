"""Case 4: The Monday Morning Crash — high complexity.

SCENARIO
--------
Monday 09:00 — the SRE on-call Tom Bradley opens the support queue.
There are 12 tickets. He starts reading.

By 09:15 there are 19. By 09:30 there are 31. By 11:00 there are 47.
Every ticket is a variation of the same thing: login failure, access denied,
authentication error. Three departments. Two office locations.

Tom is still reading ticket #3 when Ninai has already:
  - Detected the +380% anomaly vs the 30-day baseline
  - Identified it as company-wide (3 departments, org_tier=enterprise)
  - Matched playbook MAJOR-INCIDENT-RESPONSE-v3 at 91% confidence
  - Decomposed "Restore Auth Service" into 5 subtasks
  - Identified the blocking task: "Declare P1 and page infra on-call"
  - Written the SRE briefing

COMPLEXITY: HIGH
KEY INSIGHT: "47 individual complaints are one incident. Ninai declared it
             before a human read ticket #3."

AGENT PIPELINE
--------------
AnomalyDetectionAgent       — spike: 47 tickets in 2h vs baseline 12/day (+380%)
TemporalReasoningAgent      — velocity: doubling every ~30 minutes
OrgAttentionAgent           — company-wide scope, enterprise tier, escalate to exec layer
GoalDecompositionAgent      — "Restore Auth Service": 5 subtasks, blocking = "Declare P1"
PlaybookAgent               — matches MAJOR-INCIDENT-RESPONSE-v3 at 91% confidence
NarrativeSynthesisAgent     — writes the SRE incident briefing

RECORDS (~61 total)
-------------------
C4-HERO  — Synthetic spike summary (1 record), days_ago=0, scope=organization
            This is the hero — it represents the live picture of the spike.
            content: "SRE monitoring alert: 47 tickets in 2 hours (09:00-11:00 UTC).
                     All involve login/access failures. Departments: Engineering, Operations,
                     Finance. Baseline: 12/day. Extrapolated current rate: 280/day."

C4-S01..C4-S50  — Spike records (50 records), all days_ago=0
            Pulled from incident_event_log.csv filtered by category containing
            "access" or "software". Represent Monday morning's flood.
            Distributed across 3 SRE members (tom/aisha/luca) in rotation.

C4-B01..C4-B10  — Baseline records (10 records), days_ago from the list [7,10,14,18,22,25,28,30,35,40]
            Same category/subcategory as spike records but normal volume.
            Give AnomalyDetectionAgent the historical context to score the anomaly.

HERO RECORD: C4-HERO
  The spike summary is the hero because it is the record a CSR or SRE would look at
  to understand the situation. The briefing card surfaces off this record.

KAGGLE DATA USAGE
-----------------
Spike records (C4-S01..C4-S50):
    from df_incident, filter rows where:
        category.str.lower().str.contains("access|software", na=False, regex=True)
    Take first 50 rows. Map each to a record with days_ago=0.

Baseline records (C4-B01..C4-B10):
    Use the same filtered DataFrame but rows 50-60 (different rows from spike).
    Map each to days_ago from [7,10,14,18,22,25,28,30,35,40].

HOW TO IMPLEMENT
----------------
Follow case_02_team_storm.py exactly. The structure is identical.
The main difference: more records (61 vs 8) and different agents.

  1. build_records()      — hero + spike records + baseline records
  2. run_pipeline()       — calls _run_pipeline() with cross_enrichment
  3. format_comparison()  — CSR card + comparison table + anomaly chart
  4. get_cells()          — 7 cells (same as case_02 but +1 chart cell)

CROSS_ENRICHMENT for run_pipeline()
-------------------------------------
Inject this into C4-HERO before running agents:

    cross_enrichment = {
        "canonical_entity": "Meridian Auth Service",
        "resolved_entities": [
            {"name": "Tom Bradley",  "email": "tom.bradley@meridian-tech.com",  "role": "on-call SRE"},
            {"name": "Auth Service", "entity": "system", "status": "degraded"},
        ],
        "anomaly_detected":   True,
        "anomaly_score":      0.94,        # very high — 23x baseline
        "baseline_rate":      12.0,        # tickets/day over last 30 days
        "current_rate":       280.0,       # extrapolated from 2-hour window
        "spike_ratio":        23.3,        # 280/12
        "total_tickets_in_window": 47,
        "affected_departments": ["Engineering", "Operations", "Finance"],
        "temporal_anchor":    "2026-03-30T09:00:00Z",
        "sequence_position":  1,
        "trend_direction":    "spike",
        "goal_detected":      True,
        "subtasks": [
            "Declare P1 and page infra on-call",
            "Check auth service health dashboard",
            "Roll back last deployment if auth-related",
            "Enable incident bridge channel",
            "Post status-page update",
        ],
        "blocking_subtask":   "Declare P1 and page infra on-call",
        "completion_fraction": 0.0,
        "matched_playbook_id": "MAJOR-INCIDENT-RESPONSE-v3",
        "playbook_confidence": 0.91,
        "org_tier":           "enterprise",
        "org_attention_level": "critical",
        "uncertainty_level":  "low",
        "review_recommended": False,
        "credibility_score":  0.91,
        "source_tier":        "primary",
    }

COMPARISON QUESTIONS (use in format_comparison)
-------------------------------------------------
questions = [
    "What is this queue of 47 tickets?",
    "Is this normal volume for a Monday?",
    "Which departments are affected?",
    "What should the on-call do first?",
    "Is there a playbook for this situation?",
]
raw_answers = [
    "A busy Monday — read each one individually",
    "Unknown — no historical baseline visible",
    "Unknown — each ticket is from one person",
    "Unknown — start reading and triage manually",
    "No automatic matching — on-call must remember",
]
# Ninai answers: derive from enrichment dict keys listed in cross_enrichment above

ANOMALY CHART (call in format_comparison and in get_cells cell 6)
-----------------------------------------------------------------
from demo.formatters import render_anomaly_chart

fig = render_anomaly_chart(
    daily_counts    = [10, 9, 14, 11, 13, 12, 47],
    labels          = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Mon(now)"],
    baseline        = 12.0,
    anomaly_day_index = 6,
    title           = "Meridian Technologies — auth tickets this week",
)
plt.show()

GET_CELLS STRUCTURE (7 cells — same as case_02 + 1 chart cell)
--------------------------------------------------------------
Cell 0: markdown  — section header with impact table:
    | Metric | What SRE sees (raw) | What Ninai knows |
    |--------|---------------------|------------------|
    | Volume | "47 tickets in queue" | +380% above 30-day baseline |
    | Scope  | Individual tickets | Company-wide: 3 departments |
    | Status | Unknown | Auth Service degraded — P1 threshold crossed |
    | Action | Read each ticket | MAJOR-INCIDENT-RESPONSE-v3 triggered |

Cell 1: code — build_records() call (same pattern as case_02 cell 1)
Cell 2: code — ingest_records() call
Cell 3: code — run_pipeline() + agent status printout
Cell 4: code — display(Markdown(format_comparison(c4_result)))
Cell 5: code — anomaly chart:
    from demo.formatters import render_anomaly_chart
    import matplotlib.pyplot as plt
    fig = render_anomaly_chart([10,9,14,11,13,12,47],
                               ["Mon","Tue","Wed","Thu","Fri","Sat","Mon(now)"],
                               12.0, 6, "Meridian — auth tickets this week")
    plt.show()
Cell 6: markdown — key insight table:
    | Without Ninai | With Ninai |
    |---|---|
    | Tom reads ticket #3 at 09:15 | P1 declared at 09:01 by Ninai |
    | Each ticket handled separately | MAJOR-INCIDENT-RESPONSE-v3 triggered |
    | Root cause found after 2 hours | Playbook page 1 says: check auth dashboard |
    | 3 departments left in the dark | Exec layer paged, status page updated |

RETURN TYPE REFERENCE
----------------------
build_records()     -> list[dict]   — 61 records: [C4-HERO, C4-S01..S50, C4-B01..B10]
run_pipeline()      -> dict         — standard result dict, hero_id="C4-HERO"
format_comparison() -> str          — markdown string
get_cells()         -> list[tuple[str,str]]  — 7 tuples
"""

from __future__ import annotations

from typing import Any

from demo.data_prep import build_case4_records, load_incident
from demo.agents import run_pipeline as _run_pipeline, ingest_records
from demo.formatters import render_csr_card, render_comparison_table, render_anomaly_chart

# Agent imports — 6 agents for this case
from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.temporal_reasoning_agent import TemporalReasoningAgent
from app.agents.org_attention_agent import OrgAttentionAgent
from app.agents.goal_decomposition_agent import GoalDecompositionAgent
from app.agents.playbook_agent import PlaybookAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent

# ── Module constants (do not change) ─────────────────────────────────────────
CASE_NUMBER    = 4
CASE_TITLE     = "The Monday Morning Crash"
HERO_LOCAL_ID  = "C4-HERO"

AGENT_PIPELINE = [
    AnomalyDetectionAgent,
    TemporalReasoningAgent,
    OrgAttentionAgent,
    GoalDecompositionAgent,
    PlaybookAgent,
    NarrativeSynthesisAgent,
]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 of 4: build_records
# ─────────────────────────────────────────────────────────────────────────────
def build_records(data: dict, org_data: dict) -> list[dict]:
    """Construct C4-HERO + 50 spike records + 10 baseline records.

    Parameters — same shape as all other cases:
        data["support"]  : pd.DataFrame  (not used in this case)
        data["incident"] : pd.DataFrame  (incident_event_log.csv, 10K rows)
        org_data         : bootstrap dict with org_id, teams, users

    Returns
    -------
    list[dict]  — 61 records: [C4-HERO, C4-S01, ..., C4-S50, C4-B01, ..., C4-B10]

    Implementation steps
    --------------------
    Step 1: Extract org IDs needed:
        org_id   = org_data["org_id"]
        sre_tid  = org_data["teams"]["sre"]["team_id"]
        tom_uid  = org_data["users"]["tom.bradley@meridian-tech.com"]["user_id"]
        aisha_uid= org_data["users"]["aisha.hassan@meridian-tech.com"]["user_id"]
        luca_uid = org_data["users"]["luca.ferrari@meridian-tech.com"]["user_id"]

    Step 2: Build C4-HERO (1 synthetic record):
        hero = {
            "id": "C4-HERO", "days_ago": 0,
            "title": "LIVE: Auth/access ticket spike — 47 tickets in 2 hours",
            "content": (
                "SRE monitoring alert: ticket intake has reached 47 tickets in 2 hours "
                "(09:00-11:00 UTC Monday). All involve login failures, access denied, "
                "or authentication errors. Departments: Engineering, Operations, Finance. "
                "Baseline: 12 tickets/day. Extrapolated rate: 280/day. On-call: Tom Bradley."
            ),
            "source": "system", "role": "engineer", "scope": "organization",
            "tags": ["spike", "auth", "company-wide", "monday-crash", "case-4", "sre"],
            "meta": {
                "org_id": org_id, "team_id": sre_tid, "team": "sre",
                "user_id": tom_uid, "ticket_id": "ALERT-2026-0330",
                "priority": "critical", "category": "sre/incident", "case": 4,
            }
        }

    Step 3: Filter incident DataFrame for spike records:
        _df = data["incident"]
        if "category" in _df.columns:
            spike_rows = _df[
                _df["category"].str.lower().str.contains("access|software", na=False, regex=True)
            ].head(50).reset_index(drop=True)
        else:
            spike_rows = _df.head(50).reset_index(drop=True)

    Step 4: Build 50 spike records (C4-S01 .. C4-S50):
        SRE members in rotation: [(tom_uid, "tom.bradley"), (aisha_uid, "aisha.hassan"), (luca_uid, "luca.ferrari")]
        For each row i in spike_rows:
            uid, uname = sre_members[i % 3]
            record = {
                "id": f"C4-S{i+1:02d}", "days_ago": 0,
                "title": f"Access failure — {row.get('category','auth')} (ticket {i+1} of 47)",
                "content": (
                    f"[Kaggle ref: {row.get('number','N/A')}, group: {row.get('assignment_group','N/A')}] "
                    f"Reporter: {uname} (SRE on-call). "
                    f"Issue: {row.get('category','access')}/{row.get('subcategory','login')} failure. "
                    f"Priority: {row.get('priority','high')}. Part of Monday morning spike."
                ),
                "source": row.get("contact_type", "portal") or "portal",
                "role": "engineer", "scope": "organization",
                "tags": ["auth", "access", "spike", "monday-crash", "case-4"],
                "meta": {
                    "org_id": org_id, "team_id": sre_tid, "team": "sre",
                    "user_id": uid, "ticket_id": f"INC-{row.get('number','0000')}",
                    "priority": row.get("priority", "high") or "high",
                    "category": f"sre/{row.get('category','access')}",
                    "case": 4,
                }
            }

    Step 5: Build 10 baseline records (C4-B01 .. C4-B10):
        baseline_rows = spike_rows.tail(10)  # rows 40-49 from the filtered DataFrame
        baseline_days = [7, 10, 14, 18, 22, 25, 28, 30, 35, 40]
        For each row i, create a record with days_ago=baseline_days[i], same structure as spike
        but with id="C4-B{i+1:02d}" and tags=["baseline", "auth", "case-4"]

    Step 6: return [hero] + spike_records + baseline_records
    """
    # TODO: implement following steps 1-6 above
    # Reference: build_case2_records() in data_prep.py for the Kaggle row mapping pattern
    return build_case4_records(data["incident"], org_data)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 of 4: run_pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(records: list[dict], org_data: dict) -> dict[str, Any]:
    """Run the Case 4 agent pipeline on all 61 records.

    Parameters / Returns — same shape as case_01.run_pipeline() and case_02.run_pipeline().

    Implementation: 2 steps only.

    Step 1: Define cross_enrichment (copy exactly from module docstring above).
    Step 2: return _run_pipeline(
                records=records,
                agent_classes=AGENT_PIPELINE,
                hero_local_id=HERO_LOCAL_ID,
                cross_record_enrichment=cross_enrichment,
            )

    That is the entire implementation. Look at case_01.run_pipeline() lines 70-100
    for the identical pattern.
    """
    cross_enrichment = {
        "canonical_entity": "Meridian Auth Service",
        "resolved_entities": [
            {"name": "Tom Bradley",  "email": "tom.bradley@meridian-tech.com",  "role": "on-call SRE"},
            {"name": "Auth Service", "entity": "system", "status": "degraded"},
        ],
        "anomaly_detected":    True,
        "anomaly_score":       0.94,
        "baseline_rate":       12.0,
        "current_rate":        280.0,
        "spike_ratio":         23.3,
        "total_tickets_in_window": 47,
        "affected_departments": ["Engineering", "Operations", "Finance"],
        "temporal_anchor":     "2026-03-30T09:00:00Z",
        "sequence_position":   1,
        "trend_direction":     "spike",
        "goal_detected":       True,
        "subtasks": [
            "Declare P1 and page infra on-call",
            "Check auth service health dashboard",
            "Roll back last deployment if auth-related",
            "Enable incident bridge channel",
            "Post status-page update",
        ],
        "blocking_subtask":    "Declare P1 and page infra on-call",
        "completion_fraction": 0.0,
        "matched_playbook_id": "MAJOR-INCIDENT-RESPONSE-v3",
        "playbook_confidence": 0.91,
        "org_tier":            "enterprise",
        "org_attention_level": "critical",
        "uncertainty_level":   "low",
        "review_recommended":  False,
        "credibility_score":   0.91,
        "source_tier":         "primary",
        "tone":                "urgent",
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
    """Build the before/after markdown string for Case 4.

    Parameters / Returns — same shape as case_01.format_comparison().

    Implementation steps (same 5 steps as case_01, with Case 4 specific values)
    --------------------
    Step 1: enr  = result["enrichments"].get("C4-HERO", {})
             hero = result["hero"]

    Step 2: Extract display values from enr:
        spike_pct = int((enr.get("spike_ratio", 23.3) - 1) * 100)   # 2230%? use baseline vs current
        spike_pct = int((enr.get("current_rate", 280) / enr.get("baseline_rate", 12) - 1) * 100)
        departments = ", ".join(enr.get("affected_departments") or ["Engineering", "Operations", "Finance"])
        playbook    = enr.get("matched_playbook_id", "MAJOR-INCIDENT-RESPONSE-v3")
        pb_conf     = int(float(enr.get("playbook_confidence", 0.91)) * 100)
        blocking    = enr.get("blocking_subtask", "Declare P1 and page infra on-call")
        tone        = enr.get("tone", "urgent").upper()

    Step 3: Build ninai_answers using the values from step 2:
        ninai_answers = [
            f"+{spike_pct}% above 30-day baseline — anomaly declared (score: {enr.get('anomaly_score', 0.94):.2f})",
            f"No — {spike_pct}% above 30-day baseline of {enr.get('baseline_rate', 12):.0f}/day",
            departments,
            f"'{blocking}' — first step of matched playbook",
            f"{playbook} ({pb_conf}% confidence) — auto-matched from incident pattern",
        ]

    Step 4: csr_card_str = render_csr_card(CASE_NUMBER, CASE_TITLE, hero, enr,
                                            result["agent_outputs"].get("C4-HERO", {}))

    Step 5: table_str = render_comparison_table(questions, raw_answers, ninai_answers)
            return csr_card_str + "\\n\\n### Before vs After\\n\\n" + table_str

    Use these exact questions and raw_answers:

    questions = [
        "What is this queue of 47 tickets?",
        "Is this normal volume for a Monday?",
        "Which departments are affected?",
        "What should the on-call do first?",
        "Is there a playbook for this situation?",
    ]
    raw_answers = [
        "A busy Monday — read each one individually",
        "Unknown — no historical baseline visible",
        "Unknown — each ticket is from one person",
        "Unknown — start reading and triage manually",
        "No automatic matching — on-call must remember",
    ]
    """
    enr = result["enrichments"].get(HERO_LOCAL_ID, {})
    hero = result["hero"]

    baseline_rate = float(enr.get("baseline_rate", 12) or 12)
    current_rate = float(enr.get("current_rate", 280) or 280)
    spike_pct = int((current_rate / baseline_rate - 1.0) * 100)
    departments = ", ".join(enr.get("affected_departments") or ["Engineering", "Operations", "Finance"])
    playbook = enr.get("matched_playbook_id", "MAJOR-INCIDENT-RESPONSE-v3")
    pb_conf = int(float(enr.get("playbook_confidence", 0.91)) * 100)
    blocking = enr.get("blocking_subtask", "Declare P1 and page infra on-call")
    anomaly_score = float(enr.get("anomaly_score", 0.94))

    questions = [
        "What is this queue of 47 tickets?",
        "Is this normal volume for a Monday?",
        "Which departments are affected?",
        "What should the on-call do first?",
        "Is there a playbook for this situation?",
    ]
    raw_answers = [
        "A busy Monday - read each one individually",
        "Unknown - no historical baseline visible",
        "Unknown - each ticket is from one person",
        "Unknown - start reading and triage manually",
        "No automatic matching - on-call must remember",
    ]
    ninai_answers = [
        f"+{spike_pct}% above baseline - anomaly declared (score: {anomaly_score:.2f})",
        f"No - {spike_pct}% above 30-day baseline of {baseline_rate:.0f}/day",
        departments,
        f"{blocking} - first step of matched playbook",
        f"{playbook} ({pb_conf}% confidence) auto-matched from pattern",
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
    """Return 7 notebook cells for Case 4 (same as case_02 + 1 chart cell).

    Returns
    -------
    list[tuple[str, str]]  — 7 tuples in this order:
        0: ("markdown", section_header)
        1: ("code",     build_records_cell)
        2: ("code",     ingest_records_cell)
        3: ("code",     run_pipeline_cell)
        4: ("code",     display_card_cell)
        5: ("code",     anomaly_chart_cell)   ← extra cell vs case_02
        6: ("markdown", key_insight_cell)

    Implementation steps
    --------------------
    Follow case_02.get_cells() exactly for cells 0-4 and 6.
    Add cell 5 (anomaly chart) between cells 4 and 6.

    Cell 0 content (markdown section header) must include:
        ## Case 4 — The Monday Morning Crash
        > Complexity: High | Agents: 6 | Team: SRE on-call
        And this table:
        | Metric | Raw queue view | Ninai |
        |--------|----------------|-------|
        | Volume | "47 tickets in queue" | +380% above 30-day baseline |
        | Scope  | Individual tickets | Company-wide: 3 departments |
        | Status | Unknown | Auth Service degraded, P1 threshold crossed |
        | Action | Read each ticket | MAJOR-INCIDENT-RESPONSE-v3 triggered |

    Cell 1 content (code) must contain:
        from demo.case_04_company_crash import build_records, run_pipeline, format_comparison
        df_incident = load_incident(nrows=5000)
        c4_records = build_records({"support": df_support, "incident": df_incident}, org_data)
        print(f"Case 4: {len(c4_records)} records (hero + spike + baseline)")

    Cell 2 content (code) must contain:
        from demo.agents import ingest_records
        c4_memory_ids = ingest_records(ninai_client, c4_records)
        print(f"Case 4 ingested: {len(c4_memory_ids)} records")

    Cell 3 content (code) must contain:
        c4_result = run_pipeline(c4_records, org_data)
        print("Agents run on C4-HERO:")
        for agent_name, outputs in c4_result["agent_outputs"].get("C4-HERO", {}).items():
            print(f"  {agent_name}: {'ok' if 'error' not in outputs else 'ERROR'}")

    Cell 4 content (code) must contain:
        from IPython.display import Markdown, display
        display(Markdown(format_comparison(c4_result)))

    Cell 5 content (code) must contain:
        from demo.formatters import render_anomaly_chart
        import matplotlib.pyplot as plt
        fig = render_anomaly_chart(
            daily_counts=[10, 9, 14, 11, 13, 12, 47],
            labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Mon(now)"],
            baseline=12.0,
            anomaly_day_index=6,
            title="Meridian Technologies — auth tickets this week",
        )
        plt.show()

    Cell 6 content (markdown key insight table) must contain:
        | Without Ninai | With Ninai |
        |---|---|
        | Tom reads ticket #3 at 09:15 | P1 declared at 09:01 by Ninai |
        | Each ticket handled separately | MAJOR-INCIDENT-RESPONSE-v3 triggered |
        | Root cause found after 2 hours | Playbook step 1: check auth dashboard |
        | 3 departments left waiting | Exec layer paged, status page updated |
        And: > **Key insight — Case 4:** ...
    """
    return [
        (
            "markdown",
            "\n".join([
                "---",
                "## Case 4: The Monday Morning Crash",
                "",
                "> **Complexity:** High | **Agents:** 6 | **Team:** SRE on-call",
                "",
                "| Metric | Raw queue view | Ninai |",
                "|--------|----------------|-------|",
                "| Volume | \"47 tickets in queue\" | +380% above 30-day baseline |",
                "| Scope  | Individual tickets | Company-wide: 3 departments |",
                "| Status | Unknown | Auth Service degraded, P1 threshold crossed |",
                "| Action | Read each ticket | MAJOR-INCIDENT-RESPONSE-v3 triggered |",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.case_04_company_crash import build_records, run_pipeline, format_comparison",
                "from demo.data_prep import load_incident",
                "df_incident = load_incident(nrows=5000)",
                "c4_records = build_records({'support': df_support, 'incident': df_incident}, org_data)",
                "print(f'Case 4: {len(c4_records)} records (hero + spike + baseline)')",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.agents import ingest_records",
                "c4_memory_ids = ingest_records(ninai_client, c4_records)",
                "print(f'Case 4 ingested: {len(c4_memory_ids)} records')",
            ]),
        ),
        (
            "code",
            "\n".join([
                "c4_result = run_pipeline(c4_records, org_data)",
                "print('Agents run on C4-HERO:')",
                "for agent_name, outputs in c4_result['agent_outputs'].get('C4-HERO', {}).items():",
                "    print(f\"  {agent_name}: {'ok' if 'error' not in outputs else 'ERROR'}\")",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from IPython.display import Markdown, display",
                "display(Markdown(format_comparison(c4_result)))",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.formatters import render_anomaly_chart",
                "import matplotlib.pyplot as plt",
                "fig = render_anomaly_chart(",
                "    daily_counts=[10, 9, 14, 11, 13, 12, 47],",
                "    labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Mon(now)'],",
                "    baseline=12.0,",
                "    anomaly_day_index=6,",
                "    title='Meridian Technologies - auth tickets this week',",
                ")",
                "plt.show()",
            ]),
        ),
        (
            "markdown",
            "\n".join([
                "### Key Insight - Case 4",
                "",
                "| Without Ninai | With Ninai |",
                "|---|---|",
                "| Tom reads ticket #3 at 09:15 | P1 declared at 09:01 by Ninai |",
                "| Each ticket handled separately | MAJOR-INCIDENT-RESPONSE-v3 triggered |",
                "| Root cause found after 2 hours | Playbook step 1: check auth dashboard |",
                "| 3 departments left waiting | Exec layer paged, status page updated |",
                "",
                "> **Key insight - Case 4:** Ninai converts a flood of tickets into a single incident response within minutes.",
            ]),
        ),
    ]
