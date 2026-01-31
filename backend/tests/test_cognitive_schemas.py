from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.cognitive import (
    CriticOutput,
    CriticIssue,
    EvaluationQualityMetrics,
    EvaluationReportPayload,
    ExecutorOutput,
    ExecutorStepResult,
    PlannerOutput,
    PlannerStep,
)


def test_planner_output_schema_validates() -> None:
    out = PlannerOutput(
        objective="Do thing",
        assumptions=["a"],
        constraints=["c"],
        required_tools=["memory.search"],
        steps=[
            PlannerStep(
                step_id="S1",
                action="Search",
                tool="memory.search",
                tool_input_hint={"q": "x"},
                expected_output="results",
                success_criteria=["has_results"],
                risk_notes=["none"],
            )
        ],
        stop_conditions=["done"],
        confidence=0.7,
    )
    assert out.confidence == 0.7


def test_executor_output_schema_validates() -> None:
    out = ExecutorOutput(
        step_results=[
            ExecutorStepResult(
                step_id="S1",
                status="success",
                tool_name="memory.search",
                tool_call_id=None,
                summary="ok",
                artifacts={},
            )
        ],
        overall_status="success",
        errors=[],
    )
    assert out.overall_status == "success"


def test_critic_output_schema_validates() -> None:
    out = CriticOutput(
        evaluation="needs_evidence",
        strengths=["structured"],
        issues=[
            CriticIssue(
                type="missing_evidence",
                description="Need citations",
                affected_steps=["S1"],
                recommended_fix="Retrieve evidence",
            )
        ],
        followup_questions=["Which org?"],
        confidence=0.4,
    )
    assert out.evaluation == "needs_evidence"


def test_evaluation_report_payload_validation() -> None:
    payload = EvaluationReportPayload(
        final_decision="contested",
        reason="needs more evidence",
        evidence_memory_ids=[],
        tool_calls=[],
        iteration_count=1,
        quality_metrics=EvaluationQualityMetrics(avg_confidence=0.5, policy_denials=1, tool_failures=0),
    )
    assert payload.final_decision == "contested"


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        PlannerOutput(
            objective="x",
            steps=[
                PlannerStep(
                    step_id="S1",
                    action="a",
                    expected_output="e",
                    success_criteria=["ok"],
                )
            ],
            confidence=1.5,
        )
