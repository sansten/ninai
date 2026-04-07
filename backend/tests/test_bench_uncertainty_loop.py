"""Tests for Gate E4 — Uncertainty loop closure benchmark."""
from __future__ import annotations

import pytest

from tests.benchmarks import bench_uncertainty_loop
from tests.benchmarks.bench_uncertainty_loop import (
    LOOP_FLOOR,
    _UNCERTAINTY_TASKS,
    run,
)


# ---------------------------------------------------------------------------
# Structural / constant tests
# ---------------------------------------------------------------------------


def test_loop_floor_is_0_80():
    assert LOOP_FLOOR == 0.80


def test_five_uncertainty_tasks():
    assert len(_UNCERTAINTY_TASKS) == 5


def test_each_task_has_required_keys():
    for task in _UNCERTAINTY_TASKS:
        for key in ("content", "goal", "required_entities", "enrichment"):
            assert key in task, f"task missing key: {key}"


def test_each_task_has_nonempty_required_entities():
    for task in _UNCERTAINTY_TASKS:
        assert task["required_entities"], f"required_entities empty for: {task['content']}"


def test_each_task_enrichment_has_low_confidence_fields():
    low_fields = ("credibility_score", "playbook_confidence", "temporal_confidence", "causal_confidence")
    for task in _UNCERTAINTY_TASKS:
        e = task["enrichment"]
        for field in low_fields:
            assert field in e, f"enrichment missing {field} in: {task['content']}"
            assert e[field] < 0.55, f"{field}={e[field]} is not below threshold in: {task['content']}"


def test_each_task_enrichment_has_resolution_rate_zero():
    for task in _UNCERTAINTY_TASKS:
        assert task["enrichment"].get("resolution_rate") == 0.0


def test_each_task_enrichment_has_conflicts():
    for task in _UNCERTAINTY_TASKS:
        assert task["enrichment"].get("conflicts"), f"no conflicts in: {task['content']}"


def test_each_task_enrichment_has_high_severity_conflicts():
    for task in _UNCERTAINTY_TASKS:
        assert task["enrichment"].get("high_severity_conflicts"), (
            f"no high_severity_conflicts in: {task['content']}"
        )


# ---------------------------------------------------------------------------
# Unit tests for uncertainty heuristic
# ---------------------------------------------------------------------------


def test_low_confidence_enrichment_produces_high_uncertainty():
    """Enrichment with multiple low-confidence fields must produce high/critical uncertainty."""
    from app.agents import uncertainty_reporting_agent as ura

    task = _UNCERTAINTY_TASKS[0]
    result = ura.run_heuristic(task["enrichment"])
    assert result["uncertainty_level"] in {"high", "critical"}, (
        f"uncertainty_level={result['uncertainty_level']!r}, expected high or critical"
    )


def test_empty_enrichment_produces_low_uncertainty():
    from app.agents import uncertainty_reporting_agent as ura

    result = ura.run_heuristic({})
    assert result["uncertainty_level"] in {"low", "medium"}


def test_high_credibility_enrichment_not_high_uncertainty():
    from app.agents import uncertainty_reporting_agent as ura

    result = ura.run_heuristic({
        "credibility_score": 0.95,
        "resolution_rate": 1.0,
        "conflicts": [],
    })
    assert result["uncertainty_level"] in {"low", "medium"}


# ---------------------------------------------------------------------------
# Unit tests for knowledge seeker heuristic
# ---------------------------------------------------------------------------


def test_empty_memories_produces_knowledge_gaps():
    from app.agents import active_knowledge_seeker_agent as aksa

    result = aksa.run_heuristic(
        goal="identify unknown error",
        available_memories=[],
        required_entities=["error signature", "affected service"],
    )
    assert result["knowledge_gaps"], "expected knowledge_gaps when memories are empty"


def test_matching_memories_produces_no_gaps():
    from app.agents import active_knowledge_seeker_agent as aksa

    result = aksa.run_heuristic(
        goal="check auth status",
        available_memories=[
            {"content": "auth service is healthy", "title": "auth status", "tags": ["auth"]}
        ],
        required_entities=["auth"],
    )
    assert result["knowledge_gaps"] == [] or result["coverage_score"] > 0


def test_seeker_top_question_non_none_when_gaps_exist():
    from app.agents import active_knowledge_seeker_agent as aksa

    result = aksa.run_heuristic(
        goal="investigate missing entity",
        available_memories=[],
        required_entities=["missing entity"],
    )
    assert result["knowledge_gaps"]
    assert result["top_question"] is not None


# ---------------------------------------------------------------------------
# Async run tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_required_keys():
    result = await run(mode="heuristic", strategy="heuristic")
    for key in ("benchmark", "mode", "strategy", "loop_rate", "quality_floor",
                "passed", "task_count", "loops_closed", "results", "status"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_benchmark_key_is_uncertainty_loop():
    result = await run(mode="heuristic", strategy="heuristic")
    assert result["benchmark"] == "uncertainty_loop"


@pytest.mark.asyncio
async def test_mode_propagated():
    result = await run(mode="eval", strategy="heuristic")
    assert result["mode"] == "eval"


@pytest.mark.asyncio
async def test_strategy_propagated():
    result = await run(mode="heuristic", strategy="bench")
    assert result["strategy"] == "bench"


@pytest.mark.asyncio
async def test_results_list_length_matches_tasks():
    result = await run(mode="heuristic", strategy="heuristic")
    assert len(result["results"]) == len(_UNCERTAINTY_TASKS)


@pytest.mark.asyncio
async def test_each_result_has_required_fields():
    result = await run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        for key in ("content", "uncertainty_level", "is_high_uncertainty",
                    "gap_count", "has_next_step", "loop_closed"):
            assert key in r, f"result missing key: {key}"


@pytest.mark.asyncio
async def test_each_task_produces_high_or_critical_uncertainty():
    result = await run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        assert r["is_high_uncertainty"] is True, (
            f"expected high/critical uncertainty for: {r['content']!r}, "
            f"got level={r['uncertainty_level']!r}"
        )


@pytest.mark.asyncio
async def test_each_task_produces_knowledge_gaps():
    result = await run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        assert r["gap_count"] > 0, f"expected gaps for: {r['content']!r}"


@pytest.mark.asyncio
async def test_each_task_produces_next_step():
    result = await run(mode="heuristic", strategy="heuristic")
    for r in result["results"]:
        assert r["has_next_step"] is True, f"expected next_step for: {r['content']!r}"


@pytest.mark.asyncio
async def test_heuristic_mode_passes_loop_floor():
    """Core E4 gate: loop closure rate must be >= 80%."""
    result = await run(mode="heuristic", strategy="heuristic")
    failing = [r for r in result["results"] if not r["loop_closed"]]
    assert result["passed"] is True, (
        f"E4 FAIL: loop_rate={result['loop_rate']}, floor={LOOP_FLOOR}. "
        f"Not closed: {failing}"
    )


@pytest.mark.asyncio
async def test_loop_rate_calculation():
    result = await run(mode="heuristic", strategy="heuristic")
    expected = result["loops_closed"] / result["task_count"]
    assert abs(result["loop_rate"] - round(expected, 4)) < 1e-6


@pytest.mark.asyncio
async def test_status_ok_when_passed():
    result = await run(mode="heuristic", strategy="heuristic")
    if result["passed"]:
        assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_all_integration():
    from tests.benchmarks import run_all
    bench_names = [fn.__module__.split(".")[-1] for fn in run_all.BENCHMARKS]
    assert "bench_uncertainty_loop" in bench_names
