"""Tests for Gate E2 — Plan quality benchmark (bench_plan.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.benchmarks import bench_plan
from tests.benchmarks.bench_plan import PLAN_TASKS, STEP_QUALITY_FLOOR


# ---------------------------------------------------------------------------
# Structural / constant tests
# ---------------------------------------------------------------------------


def test_quality_floor_constant_is_0_80():
    assert STEP_QUALITY_FLOOR == 0.80


def test_task_count():
    assert len(PLAN_TASKS) == 8


def test_all_tasks_have_two_elements():
    for task in PLAN_TASKS:
        assert len(task) == 2, f"plan task wrong length: {task}"


def test_all_tasks_have_string_goal():
    for goal, _ in PLAN_TASKS:
        assert isinstance(goal, str) and goal


def test_all_tasks_have_dict_context():
    for _, ctx in PLAN_TASKS:
        assert isinstance(ctx, dict)


def test_check_plan_quality_has_steps_true():
    checks = bench_plan._check_plan_quality(
        "some goal", [{"step_id": "s1", "action": "retrieve context", "tool": None}]
    )
    assert checks["has_steps"] is True


def test_check_plan_quality_has_steps_false():
    checks = bench_plan._check_plan_quality("some goal", [])
    assert checks["has_steps"] is False


def test_check_plan_quality_no_duplicate_steps_true():
    steps = [
        {"step_id": "s1", "action": "Retrieve data"},
        {"step_id": "s2", "action": "Validate output"},
    ]
    checks = bench_plan._check_plan_quality("goal", steps)
    assert checks["no_duplicate_steps"] is True


def test_check_plan_quality_no_duplicate_steps_false():
    steps = [
        {"step_id": "s1", "action": "Retrieve data"},
        {"step_id": "s2", "action": "retrieve data"},  # duplicate (case-insensitive)
    ]
    checks = bench_plan._check_plan_quality("goal", steps)
    assert checks["no_duplicate_steps"] is False


def test_check_plan_quality_actionable_true():
    steps = [
        {"action": "Retrieve context for goal"},
        {"action": "Decompose goal into subtasks"},
        {"action": "Execute subtasks"},
        {"action": "Validate completion"},
    ]
    checks = bench_plan._check_plan_quality("goal", steps)
    assert checks["actionable"] is True


def test_check_plan_quality_has_goal_true():
    checks = bench_plan._check_plan_quality("deploy service", [{"action": "retrieve"}])
    assert checks["has_goal"] is True


def test_check_plan_quality_has_goal_false_on_empty():
    checks = bench_plan._check_plan_quality("", [{"action": "retrieve"}])
    assert checks["has_goal"] is False


# ---------------------------------------------------------------------------
# Async run tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_required_keys():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    for key in ("benchmark", "mode", "strategy", "pass_rate", "quality_floor",
                "passed", "task_count", "passing_plans", "results", "status"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_benchmark_key_is_plan_quality():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    assert result["benchmark"] == "plan_quality"


@pytest.mark.asyncio
async def test_mode_propagated_to_output():
    result = await bench_plan.run(mode="eval", strategy="heuristic")
    assert result["mode"] == "eval"


@pytest.mark.asyncio
async def test_strategy_propagated_to_output():
    result = await bench_plan.run(mode="heuristic", strategy="bench")
    assert result["strategy"] == "bench"


@pytest.mark.asyncio
async def test_task_count_in_output():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    assert result["task_count"] == len(PLAN_TASKS)


@pytest.mark.asyncio
async def test_results_list_has_entry_per_task():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    assert len(result["results"]) == len(PLAN_TASKS)


@pytest.mark.asyncio
async def test_each_result_has_required_fields():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        for key in ("goal", "passed", "step_count", "checks"):
            assert key in r, f"result entry missing key: {key}"


@pytest.mark.asyncio
async def test_each_result_checks_has_four_keys():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        for k in ("has_steps", "no_duplicate_steps", "actionable", "has_goal"):
            assert k in r["checks"], f"checks dict missing: {k}"


@pytest.mark.asyncio
async def test_heuristic_mode_passes_quality_floor():
    """Core E2 gate: heuristic plan() must achieve >= 80% plan quality."""
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    failing = [r for r in result["results"] if not r["passed"]]
    assert result["passed"] is True, (
        f"E2 FAIL: pass_rate={result['pass_rate']}, floor={STEP_QUALITY_FLOOR}. "
        f"Failing plans: {failing}"
    )


@pytest.mark.asyncio
async def test_pass_rate_calculation():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    expected = result["passing_plans"] / result["task_count"]
    assert abs(result["pass_rate"] - round(expected, 4)) < 1e-6


@pytest.mark.asyncio
async def test_status_ok_when_passed():
    result = await bench_plan.run(mode="heuristic", strategy="heuristic")
    if result["passed"]:
        assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_all_integration():
    from tests.benchmarks import run_all
    bench_names = [fn.__module__.split(".")[-1] for fn in run_all.BENCHMARKS]
    assert "bench_plan" in bench_names
