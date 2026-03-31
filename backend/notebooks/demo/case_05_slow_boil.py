"""Case 5: The Slow Boil — very high complexity / predictive.

Over 5 days a specific subcategory of tickets grows: 2, 4, 7, 12, 19.
No single day looks alarming to the human on-call. Day 3 has 7 tickets —
above average but not a crisis. Day 7 is projected to hit 89 — which breaches
the SLA threshold of 50/day. Ninai's TemporalReasoningAgent detects the
exponential trend on day 3, PredictiveMonitorAgent forecasts the breach date,
and AutonomousGoalGenerationAgent generates a proactive investigation goal
— giving the team 4 days to act before it becomes a crisis.

This is Ninai's most advanced capability: acting on a signal that no human
would have noticed in time.

Complexity: VERY HIGH / PREDICTIVE
Agents: TemporalReasoningAgent → PredictiveMonitorAgent → AnomalyDetectionAgent
        → AutonomousGoalGenerationAgent → MetaCognitivePlanningAgent
        → NarrativeSynthesisAgent
Kaggle source: incident_event_log.csv — single subcategory, 5-day distribution
Hero record: C5-DAY3 (the inflection point — Ninai warns here)
Key insight: "Ninai warned you on day 3. You had 4 days to fix it.
             The on-call would have paged you on day 7 when it was too late."

Interface (same for all cases):
    build_records(data, org_data)  -> list[dict]
    run_pipeline(records, org_data) -> dict
    format_comparison(result)       -> str
    get_cells(org_data)             -> list[tuple[str, str]]

Copilot implementation guide
-----------------------------
build_records():
    Call build_case5_records(data["incident"], org_data) from data_prep.py.
    That function returns 44 records distributed as:
        Day 1 (days_ago=5): 2 records  — quiet, nothing alarming
        Day 2 (days_ago=4): 4 records  — slightly elevated
        Day 3 (days_ago=3): 7 records  — inflection point (hero records here)
        Day 4 (days_ago=2): 12 records — accelerating
        Day 5 (days_ago=1): 19 records — nearly at SLA limit

    Additionally, create 1 synthetic hero summary record:
        local_id = "C5-DAY3"
        title = "Day 3 trend alert: storage/OS tickets accelerating"
        content = (
            "Ninai TemporalReasoningAgent detected exponential growth in "
            "storage/OS incident tickets: 2 (day 1) → 4 (day 2) → 7 (day 3). "
            "Velocity is doubling every ~1.5 days. "
            "Projected day 7 volume: 89 tickets. "
            "SLA threshold: 50 tickets/day. "
            "Projected breach: day 7 (in 4 days). "
            "This alert was generated proactively by Ninai — no human escalated it."
        )
        tags = ["trend", "storage", "predictive", "slow-boil", "case-5", "proactive"]
        scope = "organization", team_slug = "sre", days_ago = 3

run_pipeline():
    Hero: "C5-DAY3"
    cross_record_enrichment = {
        "canonical_entity": "Storage/OS Subsystem",
        "temporal_anchor": "2026-03-27T00:00:00Z",  # day 3
        "sequence_position": 3,
        "trend_direction": "exponential_growth",
        "trend_velocity": 1.9,                        # doubling rate per 1.5 days
        "daily_counts": [2, 4, 7, 12, 19],
        "day_labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        "predicted_day7_count": 89,
        "sla_threshold": 50,
        "breach_day": 7,
        "days_until_breach": 4,
        "anomaly_detected": False,           # not yet anomalous on day 3
        "trend_anomaly": True,               # but trend is anomalous
        "goal_detected": True,
        "subtasks": [
            "Investigate storage subsystem health metrics",
            "Check recent OS patches applied this week",
            "Review disk usage across affected nodes",
            "Alert storage vendor if hardware degradation suspected",
            "Set up daily trend monitoring for this subcategory",
        ],
        "blocking_subtask": "Investigate storage subsystem health metrics",
        "completion_fraction": 0.0,
        "uncertainty_level": "medium",
        "review_recommended": True,
        "confidence": 0.78,
    }
    AGENT_PIPELINE = [
        TemporalReasoningAgent,
        PredictiveMonitorAgent,
        AnomalyDetectionAgent,
        AutonomousGoalGenerationAgent,
        MetaCognitivePlanningAgent,
        NarrativeSynthesisAgent,
    ]

format_comparison():
    Questions (6):
        "Is this trend concerning?"
        "When will it become a crisis?"
        "What is causing the growth?"
        "Was any alert generated?"
        "What should the team do today?"
        "How much time do we have?"
    Raw answers: all "Unknown — looked normal each day individually"
    Ninai answers: derived from enrichment keys above

    Also call render_trend_chart() from formatters.py:
        daily_counts = [2, 4, 7, 12, 19]
        labels = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
        sla_threshold = 50
        title = "Storage/OS tickets — slow-boil detection (Days 1-5)"

get_cells():
    Follow the exact same 6-cell pattern as case_01 and case_02.
    Add a 7th cell (code) that calls render_trend_chart() and plt.show().
    Section header should include a dramatic timeline table:
      | Day | Tickets | What the on-call sees | What Ninai sees |
      | 1   | 2       | Normal day            | Baseline established |
      | 2   | 4       | Slightly busy         | Growth detected (50% up) |
      | **3** | **7** | **A bit elevated**  | **⚠️ Trend alert — 4 days to SLA breach** |
      | 4   | 12      | Getting busier        | Ninai goal active: investigation started |
      | 5   | 19      | Concerning            | Ninai goal: 1/5 subtasks done |
      | 7   | 89 (projected) | 🚨 SLA BREACH | Ninai warned 4 days ago |
"""

from __future__ import annotations

from typing import Any

from demo.data_prep import build_case5_records, load_incident
from demo.agents import run_pipeline as _run_pipeline, ingest_records
from demo.formatters import (
    render_csr_card,
    render_comparison_table,
    render_trend_chart,
)

# Agent imports
from app.agents.temporal_reasoning_agent import TemporalReasoningAgent
from app.agents.predictive_monitor_agent import PredictiveMonitorAgent
from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.autonomous_goal_generation_agent import AutonomousGoalGenerationAgent
from app.agents.meta_cognitive_planning_agent import MetaCognitivePlanningAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent

CASE_NUMBER = 5
CASE_TITLE = "The Slow Boil"
HERO_LOCAL_ID = "C5-DAY3"

AGENT_PIPELINE = [
    TemporalReasoningAgent,
    PredictiveMonitorAgent,
    AnomalyDetectionAgent,
    AutonomousGoalGenerationAgent,
    MetaCognitivePlanningAgent,
    NarrativeSynthesisAgent,
]


# ---------------------------------------------------------------------------
# Standard interface — implement with Copilot following case_01/case_02
# ---------------------------------------------------------------------------

def build_records(data: dict, org_data: dict) -> list[dict]:
    """Build Case 5 records: 44 distributed + 1 day-3 hero summary.

    Copilot: call build_case5_records(data["incident"], org_data) from
    data_prep.py, prepend the C5-DAY3 synthetic hero record described above,
    and return the combined list.
    """
    base_records = build_case5_records(data["incident"], org_data)

    hero = {
        "id": HERO_LOCAL_ID,
        "title": "Day 3 trend alert: storage/OS tickets accelerating",
        "content": (
            "Ninai TemporalReasoningAgent detected growth in storage/OS incident tickets: "
            "2 (day 1) -> 4 (day 2) -> 7 (day 3). Velocity is doubling about every 1.5 days. "
            "Projected day 7 volume: 89 tickets. SLA threshold: 50 tickets/day. "
            "Projected breach in 4 days. This alert is proactive: no human escalation yet."
        ),
        "memory_type": "semantic",
        "scope": "organization",
        "source_type": "system",
        "author_role": "engineer",
        "tags": ["trend", "storage", "predictive", "slow-boil", "case-5", "proactive"],
        "days_ago": 3,
        "extra_metadata": {
            "org_id": org_data["org_id"],
            "team_id": org_data["teams"]["sre"]["team_id"],
            "team_slug": "sre",
            "user_email": "tom.bradley@meridian-tech.com",
            "user_id": org_data["users"]["tom.bradley@meridian-tech.com"]["user_id"],
            "ticket_id": "TREND-2026-0327",
            "priority": "high",
            "category": "predictive/storage-os",
            "resolution": None,
            "case": 5,
            "created_at": "2026-03-27T00:00:00Z",
        },
    }

    return [hero] + base_records


def run_pipeline(records: list[dict], org_data: dict) -> dict[str, Any]:
    """Run Case 5 agent pipeline on the trend records.

    Copilot: build cross_enrichment as described above, call _run_pipeline().
    """
    cross_enrichment = {
        "canonical_entity": "Storage/OS Subsystem",
        "temporal_anchor": "2026-03-27T00:00:00Z",
        "sequence_position": 3,
        "trend_direction": "exponential_growth",
        "trend_velocity": 1.9,
        "daily_counts": [2, 4, 7, 12, 19],
        "day_labels": ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        "predicted_day7_count": 89,
        "sla_threshold": 50,
        "breach_day": 7,
        "days_until_breach": 4,
        "anomaly_detected": False,
        "trend_anomaly": True,
        "goal_detected": True,
        "subtasks": [
            "Investigate storage subsystem health metrics",
            "Check recent OS patches applied this week",
            "Review disk usage across affected nodes",
            "Alert storage vendor if hardware degradation suspected",
            "Set up daily trend monitoring for this subcategory",
        ],
        "blocking_subtask": "Investigate storage subsystem health metrics",
        "completion_fraction": 0.0,
        "uncertainty_level": "medium",
        "review_recommended": True,
        "confidence": 0.78,
    }
    return _run_pipeline(
        records=records,
        agent_classes=AGENT_PIPELINE,
        hero_local_id=HERO_LOCAL_ID,
        cross_record_enrichment=cross_enrichment,
    )


def format_comparison(result: dict) -> str:
    """Return markdown + trend chart for Case 5.

    Copilot: follow format_comparison in case_02.py. Also call
    render_trend_chart() with the values in the docstring above.
    """
    enr = result["enrichments"].get(HERO_LOCAL_ID, {})
    hero = result["hero"]

    pred_day7 = int(enr.get("predicted_day7_count", 89) or 89)
    sla = int(enr.get("sla_threshold", 50) or 50)
    days_until = int(enr.get("days_until_breach", 4) or 4)
    blocking = enr.get("blocking_subtask", "Investigate storage subsystem health metrics")
    confidence = float(enr.get("confidence", 0.78) or 0.78)

    questions = [
        "Is this trend concerning?",
        "When will it become a crisis?",
        "What is causing the growth?",
        "Was any alert generated?",
        "What should the team do today?",
        "How much time do we have?",
    ]
    raw_answers = [
        "Unknown - looked normal each day individually",
        "Unknown - no forecast in the queue",
        "Unknown - tickets look unrelated day to day",
        "No - no proactive alert from raw tickets",
        "Unknown - continue routine triage",
        "Unknown - risk only visible after SLA breach",
    ]
    ninai_answers = [
        "Yes - exponential trend detected on day 3",
        f"Projected day 7 reaches {pred_day7}, breaching SLA {sla}",
        "Storage/OS subsystem degradation with accelerating recurrence",
        "Yes - proactive trend alert generated at day 3",
        f"Start goal execution now: {blocking}",
        f"About {days_until} days to mitigate before breach",
    ]

    csr_card = render_csr_card(
        case_number=CASE_NUMBER,
        case_title=CASE_TITLE,
        hero_record=hero,
        enrichment=enr,
        agent_outputs=result["agent_outputs"].get(HERO_LOCAL_ID, {}),
    )
    table = render_comparison_table(questions, raw_answers, ninai_answers)
    trend_note = (
        f"\n\nTrend signal confidence: **{int(confidence * 100)}%**. "
        f"Projected day 7 load: **{pred_day7}** vs SLA **{sla}**."
    )
    return f"{csr_card}\n\n### Before vs After\n\n{table}{trend_note}"


def get_cells(org_data: dict) -> list[tuple[str, str]]:
    """Return [(cell_type, source)] for Case 5 notebook section.

    Copilot: follow get_cells in case_02.py (6 cells + 1 chart cell).
    Include the dramatic timeline table from the module docstring.
    """
    return [
        (
            "markdown",
            "\n".join([
                "---",
                "## Case 5: The Slow Boil",
                "",
                "> **Complexity:** Very High / Predictive | **Agents:** 6 | **Hero:** C5-DAY3",
                "",
                "| Day | Tickets | What the on-call sees | What Ninai sees |",
                "|-----|---------|------------------------|------------------|",
                "| 1 | 2 | Normal day | Baseline established |",
                "| 2 | 4 | Slightly busy | Growth detected (+100%) |",
                "| **3** | **7** | **A bit elevated** | **Trend alert - 4 days to SLA breach** |",
                "| 4 | 12 | Getting busier | Goal active: investigation started |",
                "| 5 | 19 | Concerning | Goal progress tracked with blockers |",
                "| 7 (projected) | 89 | SLA breach | Ninai warned 4 days earlier |",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.case_05_slow_boil import build_records, run_pipeline, format_comparison",
                "from demo.data_prep import load_incident",
                "df_incident = load_incident(nrows=5000)",
                "c5_records = build_records({'support': df_support, 'incident': df_incident}, org_data)",
                "print(f'Case 5: {len(c5_records)} records built')",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.agents import ingest_records",
                "c5_memory_ids = ingest_records(ninai_client, c5_records)",
                "print(f'Case 5 ingested: {len(c5_memory_ids)} records')",
            ]),
        ),
        (
            "code",
            "\n".join([
                "c5_result = run_pipeline(c5_records, org_data)",
                "print('Agents run on C5-DAY3:')",
                "for agent_name, outputs in c5_result['agent_outputs'].get('C5-DAY3', {}).items():",
                "    print(f\"  {agent_name}: {'ok' if 'error' not in outputs else 'ERROR'}\")",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from IPython.display import Markdown, display",
                "display(Markdown(format_comparison(c5_result)))",
            ]),
        ),
        (
            "code",
            "\n".join([
                "from demo.formatters import render_trend_chart",
                "import matplotlib.pyplot as plt",
                "fig = render_trend_chart(",
                "    daily_counts=[2, 4, 7, 12, 19],",
                "    labels=['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],",
                "    sla_threshold=50,",
                "    title='Storage/OS tickets - slow-boil detection (Days 1-5)',",
                ")",
                "plt.show()",
            ]),
        ),
        (
            "markdown",
            "\n".join([
                "### Key Insight - Case 5",
                "",
                "| Without Ninai | With Ninai |",
                "|---|---|",
                "| No alert until SLA breach | Day 3 trend alert with 4-day lead time |",
                "| Daily tickets reviewed in isolation | Cross-day trajectory tracked as one signal |",
                "| Reactive firefighting on day 7 | Proactive goal execution from day 3 |",
                "| Root cause search starts late | Blocking subtask identified immediately |",
                "",
                "> **Key insight - Case 5:** Predictive memory converts weak daily signals into early action.",
            ]),
        ),
    ]
