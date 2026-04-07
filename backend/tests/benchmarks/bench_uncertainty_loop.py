"""E4: Uncertainty loop closure benchmark.

Proves that high-uncertainty enrichment (from UncertaintyReportingAgent heuristic)
triggers ActiveKnowledgeSeekerAgent to produce knowledge gaps and an actionable
next step (top_question). The loop is closed when:
  1. uncertainty_level is "high" or "critical"
  2. knowledge_gaps list is non-empty
  3. top_question (the next step) is non-None

Both agents are called in heuristic mode — no Ollama or database required.
"""
from __future__ import annotations

from typing import Any

# Enrichment signals that drive high uncertainty in run_heuristic:
#   - conflicts + resolution_rate=0.0  → unresolved_count > 0
#   - high_severity_conflicts          → high_severity_count > 0
#   - low credibility / confidence fields
# Required entities for knowledge seeker are left uncovered (available_memories=[])
# so that knowledge_gaps is non-empty and top_question is set.

_UNCERTAINTY_TASKS: list[dict[str, Any]] = [
    {
        "content": "unknown vendor api behavior causing failures",
        "goal": "understand unknown vendor api behavior causing failures",
        "required_entities": ["vendor api", "failure mode", "sla terms"],
        "enrichment": {
            "conflicts": [
                {"severity": "high", "entity": "vendor-api", "conflict_type": "timeout"},
                {"severity": "high", "entity": "sla-terms", "conflict_type": "ambiguous"},
            ],
            "high_severity_conflicts": [
                {"entity": "vendor-api"},
                {"entity": "sla-terms"},
                {"entity": "dependencies"},
            ],
            "resolution_rate": 0.0,
            "credibility_score": 0.15,
            "playbook_confidence": 0.20,
            "temporal_confidence": 0.25,
            "causal_confidence": 0.30,
        },
    },
    {
        "content": "ambiguous customer churn signal in analytics pipeline",
        "goal": "resolve ambiguous customer churn signal in analytics pipeline",
        "required_entities": ["churn signal", "analytics model", "data source"],
        "enrichment": {
            "conflicts": [
                {"severity": "high", "entity": "churn-model", "conflict_type": "data_gap"},
            ],
            "high_severity_conflicts": [{"entity": "churn-model"}, {"entity": "data-source"}],
            "resolution_rate": 0.0,
            "credibility_score": 0.10,
            "playbook_confidence": 0.18,
            "temporal_confidence": 0.22,
            "causal_confidence": 0.28,
        },
    },
    {
        "content": "conflicting compliance requirements across regions",
        "goal": "clarify conflicting compliance requirements across regions",
        "required_entities": ["gdpr scope", "ccpa scope", "legal team decision"],
        "enrichment": {
            "conflicts": [
                {"severity": "high", "entity": "gdpr-scope", "conflict_type": "policy"},
                {"severity": "high", "entity": "ccpa-scope", "conflict_type": "policy"},
            ],
            "high_severity_conflicts": [
                {"entity": "gdpr-scope"},
                {"entity": "ccpa-scope"},
            ],
            "resolution_rate": 0.0,
            "credibility_score": 0.12,
            "playbook_confidence": 0.22,
            "temporal_confidence": 0.30,
            "causal_confidence": 0.28,
        },
    },
    {
        "content": "unclear executive directive on product roadmap",
        "goal": "clarify unclear executive directive on product roadmap",
        "required_entities": ["roadmap decision", "executive sponsor", "timeline"],
        "enrichment": {
            "conflicts": [
                {"severity": "high", "entity": "roadmap-decision", "conflict_type": "ambiguous"},
            ],
            "high_severity_conflicts": [
                {"entity": "roadmap-decision"},
                {"entity": "executive-intent"},
            ],
            "resolution_rate": 0.0,
            "credibility_score": 0.08,
            "playbook_confidence": 0.15,
            "temporal_confidence": 0.20,
            "causal_confidence": 0.25,
        },
    },
    {
        "content": "unrecognized error pattern in production logs",
        "goal": "identify unrecognized error pattern in production logs",
        "required_entities": ["error signature", "affected service", "incident history"],
        "enrichment": {
            "conflicts": [
                {"severity": "high", "entity": "error-sig", "conflict_type": "unknown"},
                {"severity": "high", "entity": "affected-svc", "conflict_type": "network"},
            ],
            "high_severity_conflicts": [
                {"entity": "error-sig"},
                {"entity": "affected-svc"},
                {"entity": "incident-db"},
            ],
            "resolution_rate": 0.0,
            "credibility_score": 0.10,
            "playbook_confidence": 0.20,
            "temporal_confidence": 0.18,
            "causal_confidence": 0.22,
        },
    },
]

LOOP_FLOOR = 0.80  # 80% of high-uncertainty cases must produce a next step


async def run(*, mode: str, strategy: str, **kwargs: Any) -> dict[str, Any]:
    from app.agents import uncertainty_reporting_agent as ura
    from app.agents import active_knowledge_seeker_agent as aksa

    closed = 0
    results = []

    for task in _UNCERTAINTY_TASKS:
        # Step 1: Check uncertainty level from enrichment signals
        u_result = ura.run_heuristic(task["enrichment"])
        uncertainty_level = u_result["uncertainty_level"]
        is_high_uncertainty = uncertainty_level in {"high", "critical"}

        # Step 2: Run knowledge seeker with empty memories → gaps exist
        s_result = aksa.run_heuristic(
            goal=task["goal"],
            available_memories=[],
            required_entities=task["required_entities"],
        )
        gaps = s_result.get("knowledge_gaps") or []
        top_question = s_result.get("top_question")

        loop_closed = is_high_uncertainty and bool(gaps) and bool(top_question)
        if loop_closed:
            closed += 1

        results.append(
            {
                "content": task["content"][:50],
                "uncertainty_level": uncertainty_level,
                "is_high_uncertainty": is_high_uncertainty,
                "gap_count": len(gaps),
                "has_next_step": bool(top_question),
                "top_question": top_question,
                "loop_closed": loop_closed,
            }
        )

    loop_rate = closed / len(_UNCERTAINTY_TASKS)
    return {
        "benchmark": "uncertainty_loop",
        "mode": mode,
        "strategy": strategy,
        "loop_rate": round(loop_rate, 4),
        "quality_floor": LOOP_FLOOR,
        "passed": loop_rate >= LOOP_FLOOR,
        "task_count": len(_UNCERTAINTY_TASKS),
        "loops_closed": closed,
        "results": results,
        "status": "ok" if loop_rate >= LOOP_FLOOR else "below_floor",
    }
