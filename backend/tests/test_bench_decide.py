"""Tests for Gate E1 — Decision quality benchmark (bench_decide.py)."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.benchmarks import bench_decide
from tests.benchmarks.bench_decide import DECISION_TASKS, QUALITY_FLOOR


# ---------------------------------------------------------------------------
# Structural / constant tests (no I/O)
# ---------------------------------------------------------------------------


def test_quality_floor_constant_is_0_75():
    assert QUALITY_FLOOR == 0.75


def test_task_count_matches_decision_tasks_length():
    assert len(DECISION_TASKS) == 12


def test_all_tasks_have_three_elements():
    for task in DECISION_TASKS:
        assert len(task) == 3, f"task length != 3: {task}"


def test_all_tasks_have_string_content():
    for content, _, _ in DECISION_TASKS:
        assert isinstance(content, str) and content


def test_all_tasks_have_dict_enrichment():
    for _, enrichment, _ in DECISION_TASKS:
        assert isinstance(enrichment, dict)


def test_all_tasks_have_valid_expected_decision():
    valid = {"escalate", "investigate", "monitor", "acknowledge"}
    for _, _, expected in DECISION_TASKS:
        assert expected in valid, f"unexpected decision value: {expected!r}"


def test_escalate_tasks_have_high_anomaly_or_keyword():
    """Escalate tasks must be triggered by anomaly_score >= 0.9 OR content keyword."""
    high_anomaly_keywords = ("critical", "urgent", "outage")
    for content, enrichment, expected in DECISION_TASKS:
        if expected != "escalate":
            continue
        score = float(enrichment.get("anomaly_score") or 0.0)
        detected = bool(enrichment.get("anomaly_detected"))
        has_keyword = any(w in content.lower() for w in high_anomaly_keywords)
        assert (detected and score >= 0.9) or has_keyword, (
            f"escalate task has neither high-score anomaly nor keyword: {content!r}"
        )


def test_investigate_tasks_have_mid_anomaly_score():
    for content, enrichment, expected in DECISION_TASKS:
        if expected != "investigate":
            continue
        score = float(enrichment.get("anomaly_score") or 0.0)
        detected = bool(enrichment.get("anomaly_detected"))
        assert detected and 0.7 <= score < 0.9, (
            f"investigate task needs anomaly_detected=True and 0.7<=score<0.9: {content!r}"
        )


def test_monitor_tasks_have_warning_or_caution_in_content():
    for content, _, expected in DECISION_TASKS:
        if expected != "monitor":
            continue
        lower = content.lower()
        assert "warning" in lower or "caution" in lower, (
            f"monitor task lacks warning/caution in content: {content!r}"
        )


def test_acknowledge_tasks_have_low_score_and_no_keywords():
    neutral_terms = ("outage", "critical", "urgent", "warning", "caution")
    for content, enrichment, expected in DECISION_TASKS:
        if expected != "acknowledge":
            continue
        score = float(enrichment.get("anomaly_score") or 0.0)
        assert score < 0.3, f"acknowledge task has high anomaly_score: {content!r}"
        for term in neutral_terms:
            assert term not in content.lower(), (
                f"acknowledge task contains trigger keyword {term!r}: {content!r}"
            )


# ---------------------------------------------------------------------------
# Benchmark run tests (async, heuristic mode — no vLLM needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_required_keys():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    for key in ("benchmark", "mode", "strategy", "accuracy", "quality_floor",
                "passed", "task_count", "correct", "results", "status"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_benchmark_key_is_decide_quality():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    assert result["benchmark"] == "decide_quality"


@pytest.mark.asyncio
async def test_mode_propagated_to_output():
    result = await bench_decide.run(mode="unit", strategy="heuristic")
    assert result["mode"] == "unit"


@pytest.mark.asyncio
async def test_strategy_propagated_to_output():
    result = await bench_decide.run(mode="heuristic", strategy="mytest")
    assert result["strategy"] == "mytest"


@pytest.mark.asyncio
async def test_task_count_in_output_matches_tasks():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    assert result["task_count"] == len(DECISION_TASKS)


@pytest.mark.asyncio
async def test_results_list_has_entry_per_task():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    assert len(result["results"]) == len(DECISION_TASKS)


@pytest.mark.asyncio
async def test_each_result_has_required_fields():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        for key in ("content", "expected", "got", "match"):
            assert key in r, f"result entry missing key: {key}"


@pytest.mark.asyncio
async def test_result_content_is_truncated_to_40_chars():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        assert len(r["content"]) <= 40


@pytest.mark.asyncio
async def test_heuristic_mode_passes_quality_floor():
    """Core E1 gate: heuristic decide() must achieve >= 75% accuracy."""
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    assert result["passed"] is True, (
        f"E1 FAIL: accuracy={result['accuracy']}, floor={QUALITY_FLOOR}. "
        f"Wrong predictions: {[r for r in result['results'] if not r['match']]}"
    )


@pytest.mark.asyncio
async def test_accuracy_is_ratio_of_correct_to_total():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    expected_accuracy = result["correct"] / result["task_count"]
    assert abs(result["accuracy"] - round(expected_accuracy, 4)) < 1e-6


@pytest.mark.asyncio
async def test_correct_count_matches_matched_results():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    matched = sum(1 for r in result["results"] if r["match"])
    assert result["correct"] == matched


@pytest.mark.asyncio
async def test_status_ok_when_passed():
    result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    if result["passed"]:
        assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_returns_passed_false_when_below_floor():
    """Mock decide() to always return 'wrong_decision' → below floor."""
    from app.services.cognitive_gateway_service import GatewayDecideResult

    mock_result = GatewayDecideResult(
        decision="wrong_decision",
        confidence=0.5,
        tone="informational",
        action_recommended=None,
        enrichment={},
        agents_run=[],
    )

    with patch(
        "tests.benchmarks.bench_decide.CognitiveGatewayService.decide",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    assert result["passed"] is False
    assert result["status"] == "below_floor"
    assert result["accuracy"] == 0.0


@pytest.mark.asyncio
async def test_status_below_floor_when_accuracy_zero():
    from app.services.cognitive_gateway_service import GatewayDecideResult

    mock_result = GatewayDecideResult(
        decision="wrong",
        confidence=0.5,
        tone="informational",
        action_recommended=None,
        enrichment={},
        agents_run=[],
    )
    with patch(
        "tests.benchmarks.bench_decide.CognitiveGatewayService.decide",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        result = await bench_decide.run(mode="heuristic", strategy="heuristic")
    assert result["status"] == "below_floor"


@pytest.mark.asyncio
async def test_run_all_integration():
    """bench_decide is importable from tests.benchmarks and runnable via run_all BENCHMARKS."""
    from tests.benchmarks import run_all
    # bench_decide should be registered in BENCHMARKS after run_all is updated
    bench_names = [fn.__module__.split(".")[-1] for fn in run_all.BENCHMARKS]
    assert "bench_decide" in bench_names
