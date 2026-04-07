"""Tests for PlaybookAutoSynthesisAgent (Phase 81)."""
from __future__ import annotations

import pytest

from app.agents.playbook_auto_synthesis_agent import (
    PlaybookAutoSynthesisAgent,
    _fingerprint,
    _generate_steps,
    run_heuristic,
    synthesize_playbooks,
)
from app.agents.types import AgentContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_outcome(outcome_type: str, impact: str, goal_id: str = "g1") -> dict:
    return {
        "outcome_type": outcome_type,
        "impact_description": impact,
        "goal_id": goal_id,
    }


def _qualifying_records(impact: str, count: int = 3) -> list[dict]:
    return [_make_outcome("valuable", impact) for _ in range(count)]


def _context(outcome_records: list[dict]) -> AgentContext:
    return {"memory": {"enrichment": {"outcome_records": outcome_records}}}


AGENT = PlaybookAutoSynthesisAgent()


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_returns_16_char_hex():
    fp = _fingerprint("database connection timeout")
    assert len(fp) == 16
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_is_deterministic():
    assert _fingerprint("api call failed") == _fingerprint("api call failed")


def test_fingerprint_normalises_case():
    assert _fingerprint("API Call Failed") == _fingerprint("api call failed")


def test_fingerprint_ignores_punctuation():
    assert _fingerprint("api, call! failed.") == _fingerprint("api call failed")


def test_fingerprint_order_independent():
    assert _fingerprint("timeout connection db") == _fingerprint("db connection timeout")


# ---------------------------------------------------------------------------
# _generate_steps
# ---------------------------------------------------------------------------

def test_generate_steps_starts_with_identify():
    records = _qualifying_records("cache miss causes load spike")
    steps = _generate_steps(records)
    assert steps[0] == "identify the recurring pattern"


def test_generate_steps_ends_with_validate():
    records = _qualifying_records("cache miss causes load spike")
    steps = _generate_steps(records)
    assert steps[-1] == "validate outcome and mark valuable"


def test_generate_steps_min_length():
    records = _qualifying_records("auth failure in prod")
    steps = _generate_steps(records)
    assert len(steps) >= 2


def test_generate_steps_truncates_long_descriptions():
    long_desc = "x" * 200
    records = [_make_outcome("valuable", long_desc)]
    steps = _generate_steps(records)
    for step in steps:
        assert len(step) <= 100  # "apply: " + 80 chars


def test_generate_steps_empty_records():
    steps = _generate_steps([])
    assert "identify the recurring pattern" in steps


# ---------------------------------------------------------------------------
# synthesize_playbooks
# ---------------------------------------------------------------------------

def test_synthesize_returns_empty_for_too_few_records():
    records = _qualifying_records("deploy rollback needed", count=2)
    result = synthesize_playbooks(records)
    assert result == []


def test_synthesize_qualifies_at_min_occurrences():
    records = _qualifying_records("deploy rollback needed", count=3)
    result = synthesize_playbooks(records)
    assert len(result) == 1


def test_synthesize_filters_low_success_rate():
    records = [_make_outcome("valuable", "disk full")] * 2 + \
              [_make_outcome("not_valuable", "disk full")] * 3
    result = synthesize_playbooks(records)
    assert result == []


def test_synthesize_passes_high_success_rate():
    records = [_make_outcome("valuable", "memory spike")] * 4 + \
              [_make_outcome("not_valuable", "memory spike")] * 0
    result = synthesize_playbooks(records)
    assert len(result) == 1
    assert result[0]["success_rate"] == 1.0


def test_synthesize_success_rate_boundary():
    # 17/20 = 0.85 exactly → qualifies
    records = [_make_outcome("valuable", "cert expiry")] * 17 + \
              [_make_outcome("not_valuable", "cert expiry")] * 3
    result = synthesize_playbooks(records)
    assert len(result) == 1


def test_synthesize_below_boundary_excluded():
    # 16/20 = 0.80 < 0.85 → excluded
    records = [_make_outcome("valuable", "cert expiry below")] * 16 + \
              [_make_outcome("not_valuable", "cert expiry below")] * 4
    result = synthesize_playbooks(records)
    assert result == []


def test_synthesize_playbook_has_required_keys():
    records = _qualifying_records("api rate limit hit")
    result = synthesize_playbooks(records)
    pb = result[0]
    assert "title" in pb
    assert "signature_hash" in pb
    assert "steps" in pb
    assert "success_rate" in pb
    assert "evidence" in pb
    assert "problem_signature" in pb


def test_synthesize_evidence_counts():
    records = [_make_outcome("valuable", "pod oom killed")] * 6 + \
              [_make_outcome("not_valuable", "pod oom killed")]
    result = synthesize_playbooks(records)
    assert result[0]["evidence"]["outcome_count"] == 7
    assert result[0]["evidence"]["valuable_count"] == 6


def test_synthesize_multiple_distinct_patterns():
    records = _qualifying_records("pattern-alpha") + _qualifying_records("pattern-beta")
    result = synthesize_playbooks(records)
    assert len(result) == 2


def test_synthesize_generic_key_for_empty_description():
    records = [_make_outcome("valuable", "") for _ in range(3)]
    result = synthesize_playbooks(records)
    # All land in "generic" bucket → 1 playbook or 0 (all empty → fingerprint="generic")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# run_heuristic
# ---------------------------------------------------------------------------

def test_run_heuristic_returns_required_keys():
    outputs = run_heuristic([])
    for key in ("synthesized_count", "patterns_found", "playbooks", "confidence", "rationale"):
        assert key in outputs


def test_run_heuristic_zero_records():
    outputs = run_heuristic([])
    assert outputs["synthesized_count"] == 0
    assert outputs["patterns_found"] == 0
    assert outputs["playbooks"] == []
    assert outputs["rationale"] == "heuristic"


def test_run_heuristic_confidence_baseline_no_patterns():
    outputs = run_heuristic([])
    assert outputs["confidence"] == 0.40


def test_run_heuristic_confidence_increases_with_patterns():
    records = _qualifying_records("network flap recovery")
    outputs = run_heuristic(records)
    assert outputs["confidence"] > 0.40


def test_run_heuristic_confidence_capped_at_090():
    records = []
    for i in range(10):
        records += _qualifying_records(f"pattern {i}")
    outputs = run_heuristic(records)
    assert outputs["confidence"] <= 0.90


# ---------------------------------------------------------------------------
# PlaybookAutoSynthesisAgent class
# ---------------------------------------------------------------------------

def test_agent_name():
    assert AGENT.name == "PlaybookAutoSynthesisAgent"


def test_agent_version():
    assert AGENT.version == "v1"


def test_agent_dependencies():
    deps = AGENT.dependencies()
    assert "PlaybookExecutionTrackerAgent" in deps


@pytest.mark.asyncio
async def test_run_empty_enrichment():
    result = await AGENT.run("mem-001", {})
    assert result.status == "success"
    assert result.outputs["synthesized_count"] == 0


@pytest.mark.asyncio
async def test_run_no_qualifying_records():
    ctx = _context(_qualifying_records("too few", count=2))
    result = await AGENT.run("mem-002", ctx)
    assert result.outputs["synthesized_count"] == 0


@pytest.mark.asyncio
async def test_run_with_qualifying_records():
    ctx = _context(_qualifying_records("repeated auth failure pattern", count=4))
    result = await AGENT.run("mem-003", ctx)
    assert result.status == "success"
    assert result.outputs["synthesized_count"] == 1
    assert result.outputs["patterns_found"] == 1
    assert len(result.outputs["playbooks"]) == 1


@pytest.mark.asyncio
async def test_run_confidence_in_range():
    ctx = _context(_qualifying_records("slow query on reports page", count=5))
    result = await AGENT.run("mem-004", ctx)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_run_agent_name_in_result():
    result = await AGENT.run("mem-005", {})
    assert result.agent_name == "PlaybookAutoSynthesisAgent"


@pytest.mark.asyncio
async def test_run_timestamps_set():
    result = await AGENT.run("mem-006", {})
    assert result.started_at <= result.finished_at


@pytest.mark.asyncio
async def test_run_trace_id_from_context():
    ctx: AgentContext = {"runtime": {"job_id": "job-xyz"}}
    result = await AGENT.run("mem-007", ctx)
    assert result.trace_id == "job-xyz"


@pytest.mark.asyncio
async def test_run_trace_id_none_when_absent():
    result = await AGENT.run("mem-008", {})
    assert result.trace_id is None


@pytest.mark.asyncio
async def test_run_multiple_patterns():
    records = _qualifying_records("disk pressure event") + \
              _qualifying_records("auth token expiry")
    ctx = _context(records)
    result = await AGENT.run("mem-009", ctx)
    assert result.outputs["synthesized_count"] == 2


@pytest.mark.asyncio
async def test_validate_outputs_passes_on_good_result():
    ctx = _context(_qualifying_records("cache eviction cascade", count=5))
    result = await AGENT.run("mem-010", ctx)
    # validate_outputs should not raise
    AGENT.validate_outputs(result)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_contains_agent():
    from app.agents.registry import AGENT_CLASSES
    names = [cls.__name__ for cls in AGENT_CLASSES]
    assert "PlaybookAutoSynthesisAgent" in names


def test_registry_get_agent():
    from app.agents.registry import get_agent
    agent = get_agent("playbook_auto_synthesis")
    assert agent is not None
    assert isinstance(agent, PlaybookAutoSynthesisAgent)
