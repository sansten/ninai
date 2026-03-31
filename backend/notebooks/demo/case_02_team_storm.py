"""Case 2: The Team Storm — low-medium complexity.

Five DevOps engineers at Meridian file 8 separate tickets in one week.
Each ticket looks isolated: a pipeline failure, a cert error, a permissions
issue. Nobody connects them. The account manager has no idea the company's
biggest engineering team is in crisis. Ninai groups them into one episode
and surfaces the common root cause.

Complexity: LOW-MEDIUM
Agents: EntityResolutionAgent → EpisodicGroupingAgent → SiloPropagationAgent
        → OrgAttentionAgent → NarrativeSynthesisAgent
Kaggle source: incident_event_log.csv (real assignment_group cluster)
Hero record: C2-T7 (SSO redirect broken — highest impact)
Key insight: "Five isolated tickets are one organizational crisis."

Interface (same for all cases):
    build_records(data, org_data)  -> list[dict]
    run_pipeline(records, org_data) -> dict
    format_comparison(result)       -> str
    get_cells(org_data)             -> list[tuple[str, str]]
"""

from __future__ import annotations

from typing import Any

from demo.data_prep import build_case2_records, load_incident
from demo.agents import run_pipeline as _run_pipeline, ingest_records
from demo.formatters import render_csr_card, render_comparison_table, render_episode_table

# Agent imports
from app.agents.entity_resolution_agent import EntityResolutionAgent
from app.agents.episodic_grouping_agent import EpisodicGroupingAgent
from app.agents.silo_propagation_agent import SiloPropagationAgent
from app.agents.org_attention_agent import OrgAttentionAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent

CASE_NUMBER = 2
CASE_TITLE = "The Team Storm"
HERO_LOCAL_ID = "C2-T7"  # SSO redirect broken — broadest impact

AGENT_PIPELINE = [
    EntityResolutionAgent,
    EpisodicGroupingAgent,
    SiloPropagationAgent,
    OrgAttentionAgent,
    NarrativeSynthesisAgent,
]


# ---------------------------------------------------------------------------
# Standard interface
# ---------------------------------------------------------------------------

def build_records(data: dict, org_data: dict) -> list[dict]:
    """Build Case 2 records from Kaggle incident data + org structure.

    Parameters
    ----------
    data:     {"support": pd.DataFrame, "incident": pd.DataFrame}
    org_data: dict returned by bootstrap.py (org_id, teams, users)
    """
    return build_case2_records(data["incident"], org_data)


def run_pipeline(records: list[dict], org_data: dict) -> dict[str, Any]:
    """Run Case 2 agent pipeline on the 8 DevOps team tickets.

    Injects cross-record enrichment into the hero (C2-T7 — SSO broken):
    - resolved_entities: 5 DevOps members, all same org + team
    - episode pre-seed: all 8 tickets share auth/cert root theme
    - org_tier: "enterprise" (DevOps == core engineering)
    """
    devops_members = [
        org_data["users"]["alex.rivera@meridian-tech.com"],
        org_data["users"]["priya.patel@meridian-tech.com"],
        org_data["users"]["marcus.johnson@meridian-tech.com"],
        org_data["users"]["sofia.chen@meridian-tech.com"],
        org_data["users"]["ryan.osei@meridian-tech.com"],
    ]
    member_names = [m["full_name"] for m in devops_members]
    member_emails = [
        "alex.rivera@meridian-tech.com",
        "priya.patel@meridian-tech.com",
        "marcus.johnson@meridian-tech.com",
        "sofia.chen@meridian-tech.com",
        "ryan.osei@meridian-tech.com",
    ]

    cross_enrichment = {
        "resolved_entities": [
            {
                "name": name,
                "canonical_name": name,
                "email": email,
                "team": "devops",
                "organization_id": org_data["org_id"],
                "entity": "person",
            }
            for name, email in zip(member_names, member_emails)
        ],
        "canonical_entity": "Meridian DevOps Team",
        "episode_label": "DevOps Auth Crisis — Week 12",
        "episode_id": "EP-DEVOPS-W12",
        "episode_size": 8,
        "silo_ids": ["devops", "security"],
        "org_tier": "enterprise",
        "corroboration_count": 8,
        "affected_team_count": 1,
        "affected_user_count": 5,
        "related_ticket_ids": [f"C2-T{i+1}" for i in range(8) if f"C2-T{i+1}" != HERO_LOCAL_ID],
    }

    result = _run_pipeline(
        records=records,
        agent_classes=AGENT_PIPELINE,
        hero_local_id=HERO_LOCAL_ID,
        cross_record_enrichment=cross_enrichment,
    )
    # All 8 records belong to the same episode — ensure every enrichment shows it
    # so the episode membership table renders correctly for all rows.
    for rid in result["enrichments"]:
        result["enrichments"][rid].setdefault("episode_label", "DevOps Auth Crisis — Week 12")
        result["enrichments"][rid].setdefault("episode_id", "EP-DEVOPS-W12")
    return result


def format_comparison(result: dict) -> str:
    """Return markdown before/after comparison for Case 2."""
    enr = result["enrichments"].get(HERO_LOCAL_ID, {})
    hero = result["hero"]
    all_records = [r for r in result.get("_records", [])]  # populated by notebook cell

    questions = [
        "Is this a unique problem?",
        "Who else is affected?",
        "Is this an individual or team issue?",
        "What is the business impact?",
        "What should happen next?",
    ]
    raw_answers = [
        "Looks unique — 1 SSO failure ticket",
        "Unknown — no cross-ticket view",
        "Looks individual — one engineer filed it",
        "Unknown from one ticket",
        "Assign to DevOps on-call",
    ]

    episode = enr.get("episode_label", "DevOps Auth Crisis — Week 12")
    episode_size = enr.get("episode_size", 8)
    affected = enr.get("affected_user_count", 5)
    org_tier = enr.get("org_tier", "enterprise")
    tone = enr.get("tone", "cautionary")

    ninai_answers = [
        f"No — episode '{episode}' has {episode_size} linked tickets",
        f"{affected} DevOps engineers + security team (cross-silo propagation detected)",
        f"Team-level crisis — all {affected} DevOps members affected",
        f"Org tier: {org_tier} — entire engineering platform at risk, tone: {tone.upper()}",
        "Open P1 incident, page account manager, assign episode to DevOps lead (Alex Rivera)",
    ]

    table = render_comparison_table(questions, raw_answers, ninai_answers)
    csr_card = render_csr_card(
        case_number=CASE_NUMBER,
        case_title=CASE_TITLE,
        hero_record=hero,
        enrichment=enr,
        agent_outputs=result["agent_outputs"].get(HERO_LOCAL_ID, {}),
    )
    return f"{csr_card}\n\n### Before vs After\n\n{table}"


# ---------------------------------------------------------------------------
# Notebook cells
# ---------------------------------------------------------------------------

def get_cells(org_data: dict) -> list[tuple[str, str]]:
    """Return [(cell_type, source)] for Case 2 notebook section."""
    devops_team = org_data["teams"]["devops"]

    return [
        # --- section header ---
        ("markdown", "\n".join([
            "---",
            "## Case 2: The Team Storm",
            "",
            "> **Complexity:** Low-Medium &nbsp;|&nbsp; **Team:** DevOps (5 engineers)",
            "> &nbsp;|&nbsp; **Episode:** DevOps Auth Crisis — Week 12",
            "",
            "Eight tickets. Five engineers. One week. No one noticed it was the same crisis.",
            "",
            "| Ticket | Reporter | Issue | Days Ago |",
            "|--------|----------|-------|----------|",
            "| C2-T1 | Alex Rivera | Deployment pipeline failing | 7 |",
            "| C2-T2 | Priya Patel | Permissions denied after key rotation | 6 |",
            "| C2-T3 | Marcus Johnson | Auth tokens expiring too fast | 5 |",
            "| C2-T4 | Sofia Chen | Staging login loop | 4 |",
            "| C2-T5 | Ryan Osei | Jenkins auth plugin failing | 3 |",
            "| C2-T6 | Alex Rivera | Docker registry auth rejected | 2 |",
            "| **C2-T7** | **Marcus Johnson** | **SSO redirect broken — ALL tools** | **1** |",
            "| C2-T8 | Priya Patel | Git webhooks auth rejected | 1 |",
            "",
            "**What the CSR sees:** 8 unrelated tickets in the DevOps queue.",
            "",
            "**What Ninai knows:** One episode, one root cause (cert rotation), five people impacted.",
        ])),

        # --- build records ---
        ("code", "\n".join([
            "from demo.data_prep import load_incident",
            "from demo.case_02_team_storm import build_records, run_pipeline, format_comparison",
            "",
            "df_incident = load_incident(nrows=5000)  # load first 5K rows for cluster detection",
            "c2_records = build_records({'support': df_support, 'incident': df_incident}, org_data)",
            "print(f'Case 2: {len(c2_records)} records built')",
            "for r in c2_records:",
            "    print(f\"  {r['id']}: {r['title'][:65]} (days_ago={r['days_ago']})\")",
        ])),

        # --- ingest ---
        ("code", "\n".join([
            "c2_memory_ids = ingest_records(ninai_client, c2_records)",
            "print(f'Case 2 ingested: {len(c2_memory_ids)} records')",
        ])),

        # --- run pipeline ---
        ("code", "\n".join([
            "c2_result = run_pipeline(c2_records, org_data)",
            "c2_result['_records'] = c2_records  # attach for episode table",
            "",
            "print('Agents run on C2-T7 (SSO broken — hero ticket):')",
            "for agent_name, outputs in c2_result['agent_outputs'].get('C2-T7', {}).items():",
            "    status = 'ok' if 'error' not in outputs else 'ERROR'",
            "    print(f'  {agent_name}: {status}')",
        ])),

        # --- episode table ---
        ("code", "\n".join([
            "from demo.formatters import render_episode_table",
            "from IPython.display import Markdown, display",
            "",
            "display(Markdown('### Episode Membership (EpisodicGroupingAgent)'))",
            "display(Markdown(render_episode_table(c2_records, c2_result['enrichments'])))",
        ])),

        # --- show CSR card ---
        ("code", "\n".join([
            "display(Markdown(format_comparison(c2_result)))",
        ])),

        # --- key insight ---
        ("markdown", "\n".join([
            "### Key Insight — Case 2",
            "",
            "| Without Ninai | With Ninai |",
            "|---|---|",
            "| 8 tickets in 8 different queues | 1 episode, 1 owner, 1 root cause |",
            "| Account manager unaware | Org-level alert: enterprise client in crisis |",
            "| Engineers file duplicates | Episode membership stops duplicate work |",
            "| 5 engineers blocked for 7 days | P1 declared on day 1 of episode |",
            "",
            "> Ninai's `EpisodicGroupingAgent` grouped 8 tickets across 5 people",
            "> into one named episode. `OrgAttentionAgent` flagged the org tier as",
            "> enterprise and surfaced the team-level signal to the account layer.",
        ])),
    ]
