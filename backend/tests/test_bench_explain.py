"""Tests for Gate E3 — Explanation fidelity benchmark (bench_explain.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.benchmarks import bench_explain
from tests.benchmarks.bench_explain import EXPLAIN_FLOOR, _EXPLAIN_CASES, _check_explain_fidelity


# ---------------------------------------------------------------------------
# Structural / constant tests
# ---------------------------------------------------------------------------


def test_explain_floor_constant_is_0_80():
    assert EXPLAIN_FLOOR == 0.80


def test_explain_cases_count():
    assert len(_EXPLAIN_CASES) == 8


def test_each_case_has_memory_id_and_audit_records():
    for case in _EXPLAIN_CASES:
        assert "memory_id" in case
        assert "audit_records" in case
        assert isinstance(case["audit_records"], list)


def test_each_audit_record_has_agent_name_and_confidence():
    for case in _EXPLAIN_CASES:
        for record in case["audit_records"]:
            assert "agent_name" in record
            assert "confidence" in record


def test_audit_records_have_three_entries():
    for case in _EXPLAIN_CASES:
        assert len(case["audit_records"]) == 3


def test_all_memory_ids_are_unique():
    ids = [c["memory_id"] for c in _EXPLAIN_CASES]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# _check_explain_fidelity unit tests
# ---------------------------------------------------------------------------


def _make_fake_result(*, decisions=None, agents=None, summary="a summary", confidence=0.8):
    """Build a minimal object mirroring GatewayExplainResult."""
    from dataclasses import dataclass

    @dataclass
    class _Result:
        decisions: list
        agents: list
        explainability_summary: str
        confidence: float

    return _Result(
        decisions=decisions if decisions is not None else [{"action": "decide"}],
        agents=agents if agents is not None else ["AgentA"],
        explainability_summary=summary,
        confidence=confidence,
    )


def test_fidelity_check_has_decisions_true():
    result = _make_fake_result(decisions=[{"action": "decide"}])
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["has_decisions"] is True


def test_fidelity_check_has_decisions_false_on_empty():
    result = _make_fake_result(decisions=[])
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["has_decisions"] is False


def test_fidelity_check_has_agents_true():
    result = _make_fake_result(agents=["AgentX"])
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["has_agents"] is True


def test_fidelity_check_has_agents_false_on_empty():
    result = _make_fake_result(agents=[])
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["has_agents"] is False


def test_fidelity_check_has_summary_false_on_empty():
    result = _make_fake_result(summary="")
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["has_summary"] is False


def test_fidelity_check_confidence_non_zero_true():
    result = _make_fake_result(confidence=0.75)
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["confidence_non_zero"] is True


def test_fidelity_check_confidence_non_zero_false():
    result = _make_fake_result(confidence=0.0)
    checks = _check_explain_fidelity("mem-000", [], result)
    assert checks["confidence_non_zero"] is False


# ---------------------------------------------------------------------------
# Async run tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_required_keys():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    for key in ("benchmark", "mode", "strategy", "fidelity_rate", "quality_floor",
                "passed", "task_count", "passing_cases", "results", "status"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_benchmark_key_is_explain_fidelity():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    assert result["benchmark"] == "explain_fidelity"


@pytest.mark.asyncio
async def test_mode_propagated():
    result = await bench_explain.run(mode="eval", strategy="heuristic")
    assert result["mode"] == "eval"


@pytest.mark.asyncio
async def test_strategy_propagated():
    result = await bench_explain.run(mode="heuristic", strategy="bench")
    assert result["strategy"] == "bench"


@pytest.mark.asyncio
async def test_results_list_length_matches_cases():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    assert len(result["results"]) == len(_EXPLAIN_CASES)


@pytest.mark.asyncio
async def test_each_result_has_required_fields():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        for key in ("memory_id", "passed", "agent_count", "decision_count", "checks"):
            assert key in r, f"result entry missing key: {key}"


@pytest.mark.asyncio
async def test_each_result_checks_has_four_keys():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        for k in ("has_decisions", "has_agents", "has_summary", "confidence_non_zero"):
            assert k in r["checks"]


@pytest.mark.asyncio
async def test_heuristic_mode_passes_fidelity_floor():
    """Core E3 gate: heuristic explain() must achieve >= 80% fidelity."""
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    failing = [r for r in result["results"] if not r["passed"]]
    assert result["passed"] is True, (
        f"E3 FAIL: fidelity_rate={result['fidelity_rate']}, floor={EXPLAIN_FLOOR}. "
        f"Failing: {failing}"
    )


@pytest.mark.asyncio
async def test_fidelity_rate_calculation():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    expected = result["passing_cases"] / result["task_count"]
    assert abs(result["fidelity_rate"] - round(expected, 4)) < 1e-6


@pytest.mark.asyncio
async def test_status_ok_when_passed():
    result = await bench_explain.run(mode="heuristic", strategy="heuristic")
    if result["passed"]:
        assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_all_integration():
    from tests.benchmarks import run_all
    bench_names = [fn.__module__.split(".")[-1] for fn in run_all.BENCHMARKS]
    assert "bench_explain" in bench_names
