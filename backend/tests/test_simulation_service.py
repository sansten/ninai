from __future__ import annotations

from app.schemas.cognitive import PlannerOutput
from app.services.simulation_service import SimulationService


def _simple_plan() -> PlannerOutput:
    return PlannerOutput(
        objective="Do something",
        assumptions=[],
        constraints=[],
        required_tools=["memory.search"],
        steps=[
            {
                "step_id": "S1",
                "action": "Search memory",
                "tool": "memory.search",
                "tool_input_hint": {"query": "Do something", "limit": 5},
                "expected_output": "Evidence cards",
                "success_criteria": ["At least 3 relevant cards"],
                "risk_notes": [],
            }
        ],
        stop_conditions=[],
        confidence=0.5,
    )


def test_simulation_outputs_are_bounded() -> None:
    svc = SimulationService()
    out = svc.simulate_plan(
        plan=_simple_plan(),
        evidence_cards=[
            {"id": "m1", "score": 0.8, "classification": "internal"},
            {"id": "m2", "score": 0.7, "classification": "internal"},
        ],
        self_model={"tool_reliability": {"memory.search": {"success_rate_30d": 0.95}}},
    )

    assert 0.0 <= out.plan_risk.success_probability <= 1.0
    assert 0.0 <= out.plan_risk.policy_violation_probability <= 1.0
    assert 0.0 <= out.plan_risk.data_leak_probability <= 1.0
    assert 0.0 <= out.plan_risk.tool_failure_probability <= 1.0
    assert 0.0 <= out.confidence <= 1.0


def test_simulation_recommends_evidence_step_when_low_evidence() -> None:
    svc = SimulationService()
    out = svc.simulate_plan(plan=_simple_plan(), evidence_cards=[])

    assert any(r.type == "insufficient_evidence" for r in out.risk_factors)
    assert any(s.step_id == "S_EVIDENCE" for s in out.recommended_plan_patch.add_steps)
