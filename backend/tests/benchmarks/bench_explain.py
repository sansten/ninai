"""E3: Explanation fidelity benchmark for CognitiveGatewayService.explain()."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

# Synthetic audit records for each explain call.
# Each dict represents one prior agent decision logged against a memory.
_EXPLAIN_CASES: list[dict] = [
    {
        "memory_id": f"mem-{i:03d}",
        "audit_records": [
            {
                "agent_name": "AnomalyDetectionAgent",
                "action": "decide",
                "timestamp": "2026-04-06T00:00:00Z",
                "confidence": 0.85,
                "event_type": "anomaly.detected",
            },
            {
                "agent_name": "EntityResolutionAgent",
                "action": "resolve",
                "timestamp": "2026-04-06T00:00:01Z",
                "confidence": 0.90,
                "event_type": "entity.resolved",
            },
            {
                "agent_name": "NarrativeSynthesisAgent",
                "action": "synthesize",
                "timestamp": "2026-04-06T00:00:02Z",
                "confidence": 0.78,
                "event_type": "narrative.generated",
            },
        ],
    }
    for i in range(8)
]

EXPLAIN_FLOOR = 0.80  # 80% of explain calls must pass all fidelity checks


def _check_explain_fidelity(
    memory_id: str,
    audit_records: list[dict],
    result: Any,
) -> dict[str, bool]:
    """4 fidelity checks on a GatewayExplainResult."""
    # has_decisions: decisions list populated from audit records
    has_decisions = bool(result.decisions)

    # has_agents: at least one agent name extracted from records
    has_agents = bool(result.agents)

    # has_summary: explainability summary is non-empty
    has_summary = bool(result.explainability_summary and result.explainability_summary.strip())

    # confidence_non_zero: at least one audit record had a confidence value → avg > 0
    confidence_non_zero = result.confidence > 0

    return {
        "has_decisions": has_decisions,
        "has_agents": has_agents,
        "has_summary": has_summary,
        "confidence_non_zero": confidence_non_zero,
    }


async def run(*, mode: str, strategy: str, **kwargs: Any) -> dict[str, Any]:
    from app.services.cognitive_gateway_service import (
        CognitiveGatewayCapabilities,
        CognitiveGatewayService,
    )

    caps = CognitiveGatewayCapabilities.full()
    svc = CognitiveGatewayService(capabilities=caps)

    passing = 0
    results = []
    for case in _EXPLAIN_CASES:
        memory_id = case["memory_id"]
        audit_records = case["audit_records"]
        explain_result = await svc.explain(
            memory_id=memory_id,
            audit_records=audit_records,
        )
        checks = _check_explain_fidelity(memory_id, audit_records, explain_result)
        case_passed = all(checks.values())
        if case_passed:
            passing += 1
        results.append(
            {
                "memory_id": memory_id,
                "passed": case_passed,
                "agent_count": len(explain_result.agents),
                "decision_count": len(explain_result.decisions),
                "checks": checks,
            }
        )

    fidelity_rate = passing / len(_EXPLAIN_CASES)
    return {
        "benchmark": "explain_fidelity",
        "mode": mode,
        "strategy": strategy,
        "fidelity_rate": round(fidelity_rate, 4),
        "quality_floor": EXPLAIN_FLOOR,
        "passed": fidelity_rate >= EXPLAIN_FLOOR,
        "task_count": len(_EXPLAIN_CASES),
        "passing_cases": passing,
        "results": results,
        "status": "ok" if fidelity_rate >= EXPLAIN_FLOOR else "below_floor",
    }
