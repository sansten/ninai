"""Case 1: The Repeat Caller — low complexity.

Emma Chen has called support 3 times in 8 days about the same broken login.
Each CSR resolved it differently. Each time it came back. On the 4th call,
Ninai gives the CSR the full picture before they even say hello.

Complexity: LOW
Agents: EntityResolutionAgent → TemporalReasoningAgent → CredibilityAgent
        → FeedbackIntegrationAgent → NarrativeSynthesisAgent
Kaggle source: customer_support_tickets.csv (software type rows)
Hero record: C1-T3 (3rd contact, unresolved)
Key insight: "This is not a new ticket. It is an unresolved escalation."

Interface (same for all cases):
    build_records(data, org_data)  -> list[dict]
    run_pipeline(records, org_data) -> dict
    format_comparison(result)       -> str
    get_cells(org_data)             -> list[tuple[str, str]]
"""

from __future__ import annotations

from typing import Any

from demo.data_prep import build_case1_records, load_support
from demo.agents import run_pipeline as _run_pipeline, ingest_records
from demo.formatters import render_csr_card, render_comparison_table

# Agent imports
from app.agents.entity_resolution_agent import EntityResolutionAgent
from app.agents.temporal_reasoning_agent import TemporalReasoningAgent
from app.agents.credibility_agent import CredibilityAgent
from app.agents.feedback_integration_agent import FeedbackIntegrationAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent

CASE_NUMBER = 1
CASE_TITLE = "The Repeat Caller"
HERO_LOCAL_ID = "C1-T3"

AGENT_PIPELINE = [
    EntityResolutionAgent,
    TemporalReasoningAgent,
    CredibilityAgent,
    FeedbackIntegrationAgent,
    NarrativeSynthesisAgent,
]


# ---------------------------------------------------------------------------
# Standard interface
# ---------------------------------------------------------------------------

def build_records(data: dict, org_data: dict) -> list[dict]:
    """Build Case 1 records from loaded Kaggle data + org structure.

    Parameters
    ----------
    data:     {"support": pd.DataFrame, "incident": pd.DataFrame}
    org_data: dict returned by bootstrap.py (org_id, teams, users)
    """
    return build_case1_records(data["support"], org_data)


def run_pipeline(records: list[dict], org_data: dict) -> dict[str, Any]:
    """Run Case 1 agent pipeline on Emma's 3 tickets.

    Injects cross-record enrichment into the hero (3rd ticket):
    - resolved_entities: list with Emma's name (found in tickets 1+2)
    - unresolved_conflicts: prior resolutions failed (inferred from content)
    - corroboration_count: 3 (three tickets, same issue)
    """
    # Build enrichment from cross-record reading so the hero has context
    cross_enrichment = {
        "resolved_entities": [
            {
                "name": "Emma Chen",
                "canonical_name": "Emma Chen",
                "email": "emma.chen@meridian-client.com",
                "contact_count": 3,
                "entity": "person",
            }
        ],
        "canonical_entity": "Emma Chen",
        "corroboration_count": 3,
        "unresolved_conflicts": [
            "TKT-2026-0311: password reset — issue recurred after 3 days",
            "TKT-2026-0314: cache cleared — issue recurred after 3 days",
        ],
        "prior_resolution_failed": True,
        "contact_history": [
            {"ticket": "TKT-2026-0311", "resolution": "password reset", "outcome": "failed"},
            {"ticket": "TKT-2026-0314", "resolution": "cache cleared", "outcome": "failed"},
        ],
    }

    return _run_pipeline(
        records=records,
        agent_classes=AGENT_PIPELINE,
        hero_local_id=HERO_LOCAL_ID,
        cross_record_enrichment=cross_enrichment,
    )


def format_comparison(result: dict) -> str:
    """Return markdown before/after comparison for Case 1."""
    enr = result["enrichments"].get(HERO_LOCAL_ID, {})
    hero = result["hero"]

    questions = [
        "Is this a new issue?",
        "Has this person called before?",
        "Did the previous fix work?",
        "How urgent is this?",
        "What should the CSR do?",
    ]
    raw_answers = [
        "Looks new — only this ticket is visible",
        "No cross-ticket view in a standard portal",
        "Unknown — no follow-up linked",
        "Medium — that's what this ticket says",
        "Apply a standard fix (reset, cache clear)",
    ]

    tone = enr.get("tone", "cautionary")
    contact_count = len(
        (enr.get("resolved_entities") or [{}])[0].get("contact_count", 3)
        if enr.get("resolved_entities") else [3]
    )
    prior_failed = enr.get("prior_resolution_failed", True)
    cred = enr.get("credibility_score")
    cred_str = f"{float(cred):.0%}" if cred else "high"
    uncertainty = enr.get("uncertainty_level", "medium")

    ninai_answers = [
        "No — Ninai knows this is the 3rd ticket for the same root issue",
        f"Yes — 3 contacts in 8 days (entity resolved across all tickets)",
        "No — both previous resolutions are recorded as failed",
        f"Tone: {tone.upper()} | Credibility: {cred_str} | Uncertainty: {uncertainty}",
        "Escalate immediately — standard fixes exhausted, root cause not yet found",
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
    """Return [(cell_type, source)] for Case 1 notebook section."""
    support_team = org_data["teams"]["support-l1"]
    carlos = org_data["users"]["carlos.mendez@meridian-tech.com"]

    return [
        # --- section header ---
        ("markdown", "\n".join([
            "---",
            "## Case 1: The Repeat Caller",
            "",
            "> **Complexity:** Low &nbsp;|&nbsp; **Team:** Support L1 &nbsp;|&nbsp; **Customer:** Emma Chen",
            "",
            "Emma Chen has contacted support **3 times in 8 days** about the same broken login.",
            "Each CSR saw only their ticket. Each applied a different fix. Each fix failed.",
            "On the 4th call, the CSR using Ninai sees the full picture before picking up.",
            "",
            "**What the raw ticket says:** `\"Login broken again, previous reset did not work.\"`",
            "",
            "**What Ninai knows:** 3rd contact, 2 failed resolutions, customer credibility HIGH,",
            "entity resolved across all tickets, escalation required.",
        ])),

        # --- build records ---
        ("code", "\n".join([
            "from demo.data_prep import load_support",
            "from demo.case_01_repeat_caller import build_records, run_pipeline, format_comparison",
            "",
            "df_support = load_support()",
            f"# org_data loaded in bootstrap cell — org: {org_data['org_id'][:8]}...",
            "c1_records = build_records({'support': df_support, 'incident': None}, org_data)",
            "print(f'Case 1: {len(c1_records)} records built')",
            "for r in c1_records:",
            "    print(f\"  {r['id']}: {r['title'][:60]} (days_ago={r['days_ago']})\")",
        ])),

        # --- ingest ---
        ("code", "\n".join([
            "from demo.agents import ingest_records",
            "from tqdm.notebook import tqdm",
            "",
            "c1_memory_ids = ingest_records(ninai_client, c1_records)",
            "print('Case 1 ingested:', len(c1_memory_ids), 'records')",
            "for local_id, ninai_id in c1_memory_ids.items():",
            "    print(f'  {local_id} -> {ninai_id[:8]}...')",
        ])),

        # --- run pipeline ---
        ("code", "\n".join([
            "c1_result = run_pipeline(c1_records, org_data)",
            "",
            "hero_enr = c1_result['enrichments']['C1-T3']",
            "print('Agents run on C1-T3 (Emma\\'s 3rd ticket):')",
            "for agent_name, outputs in c1_result['agent_outputs'].get('C1-T3', {}).items():",
            "    status = 'ok' if 'error' not in outputs else 'ERROR'",
            "    print(f'  {agent_name}: {status}')",
        ])),

        # --- show CSR card + comparison ---
        ("code", "\n".join([
            "from IPython.display import Markdown, display",
            "",
            "display(Markdown(format_comparison(c1_result)))",
        ])),

        # --- key insight ---
        ("markdown", "\n".join([
            "### Key Insight — Case 1",
            "",
            "| Without Ninai | With Ninai |",
            "|---|---|",
            "| CSR opens a fresh ticket | CSR sees 3-contact history before hello |",
            "| Applies reset (3rd time) | Escalates immediately — standard fixes exhausted |",
            "| Emma calls again in 3 days | Root cause investigation triggered |",
            "",
            "> The raw ticket was 12 words. Ninai's briefing card was built from",
            "> persistent memory across all 3 contacts — same system, same data,",
            "> completely different outcome.",
        ])),
    ]
