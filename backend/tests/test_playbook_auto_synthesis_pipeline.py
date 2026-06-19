"""Tests for Phase 81: PlaybookAutoSynthesisAgent wiring + nightly pipeline."""

from __future__ import annotations

import pytest

from app.agents.playbook_auto_synthesis_agent import (
    _fingerprint,
    run_heuristic,
    synthesize_playbooks,
)


# ---------------------------------------------------------------------------
# Unit tests for the synthesis logic (no DB needed)
# ---------------------------------------------------------------------------

def _outcome(outcome_type: str, impact: str | None = None) -> dict:
    return {"outcome_type": outcome_type, "impact_description": impact, "goal_id": None}


def _valuable(impact: str) -> dict:
    return _outcome("valuable", impact)


def _not_valuable(impact: str) -> dict:
    return _outcome("not_valuable", impact)


def test_fingerprint_is_deterministic():
    assert _fingerprint("Led to better customer retention") == _fingerprint("Led to better customer retention")


def test_fingerprint_normalises_case_and_punctuation():
    a = _fingerprint("Led to better customer retention!")
    b = _fingerprint("led to better CUSTOMER retention")
    assert a == b


def test_fingerprint_different_for_different_descriptions():
    assert _fingerprint("foo bar") != _fingerprint("baz qux")


def test_synthesize_empty_input():
    assert synthesize_playbooks([]) == []


def test_synthesize_below_min_occurrences_returns_empty():
    # Only 2 records — below the 3-record minimum
    records = [_valuable("Improved meeting efficiency")] * 2
    assert synthesize_playbooks(records) == []


def test_synthesize_below_success_rate_floor_returns_empty():
    # 3 records but 2/3 = 66% success rate < 85% floor
    records = [_valuable("Saved time on reporting")] * 2 + [_not_valuable("Saved time on reporting")]
    assert synthesize_playbooks(records) == []


def test_synthesize_qualifying_cluster_produces_playbook():
    records = [_valuable("Reduced onboarding friction by 30%")] * 4
    result = synthesize_playbooks(records)
    assert len(result) == 1
    pb = result[0]
    assert pb["success_rate"] == 1.0
    assert "auto-synthesized" in pb["title"].lower()
    assert pb["evidence"]["outcome_count"] == 4
    assert pb["evidence"]["valuable_count"] == 4


def test_synthesize_multiple_distinct_patterns():
    records = (
        [_valuable("Accelerated hiring pipeline")] * 3
        + [_valuable("Reduced customer churn via proactive outreach")] * 3
    )
    result = synthesize_playbooks(records)
    assert len(result) == 2


def test_synthesize_steps_are_non_empty():
    records = [_valuable("Identified bottleneck in invoice processing")] * 3
    result = synthesize_playbooks(records)
    assert result[0]["steps"]


def test_run_heuristic_returns_correct_structure():
    records = [_valuable("Improved sprint velocity")] * 5
    out = run_heuristic(records)
    assert out["rationale"] == "heuristic"
    assert out["synthesized_count"] == 1
    assert out["patterns_found"] == 1
    assert 0.0 < out["confidence"] <= 1.0


def test_run_heuristic_no_outcomes():
    out = run_heuristic([])
    assert out["synthesized_count"] == 0
    assert out["patterns_found"] == 0
    assert out["confidence"] == 0.40


def test_run_heuristic_mixed_outcomes_below_floor():
    records = [_valuable("Cut deploy time")] + [_not_valuable("Cut deploy time")] * 2
    out = run_heuristic(records)
    assert out["synthesized_count"] == 0


def test_run_heuristic_confidence_scales_with_pattern_count():
    # Two distinct qualifying patterns → confidence > one pattern
    records_a = [_valuable("Pattern alpha description")] * 3
    records_b = [_valuable("Pattern beta description differs significantly")] * 3
    one = run_heuristic(records_a)["confidence"]
    two = run_heuristic(records_a + records_b)["confidence"]
    assert two > one


def test_generic_fingerprint_for_missing_description():
    records = [_outcome("valuable", None)] * 4
    result = synthesize_playbooks(records)
    # Records without descriptions all land in the "generic" bucket
    assert len(result) == 1
    assert result[0]["signature_hash"] == "generic"
