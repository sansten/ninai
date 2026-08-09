"""Unit tests for InterferencePreventionService — Phase 90."""
from __future__ import annotations

import pytest

from app.services.interference_prevention_service import (
    InterferencePreventionService,
    WriteStrategy,
)


@pytest.fixture
def svc() -> InterferencePreventionService:
    return InterferencePreventionService()


def _mem(text: str, credibility: float = 0.5, ref_count: int = 0, mem_id: str = "x") -> dict:
    return {"id": mem_id, "text": text, "credibility": credibility, "reference_count": ref_count}


# ---------------------------------------------------------------------------
# text_overlap
# ---------------------------------------------------------------------------

class TestTextOverlap:
    def test_identical_texts_max_overlap(self, svc):
        assert svc.text_overlap("Alice joined the team in 2022.", "Alice joined the team in 2022.") > 0.90

    def test_unrelated_texts_low_overlap(self, svc):
        assert svc.text_overlap("Alice joined the team.", "The project budget was $1M.") < 0.20

    def test_partial_overlap(self, svc):
        score = svc.text_overlap("Alice led the project launch.", "Alice announced the project results.")
        assert 0.10 < score < 0.80

    def test_empty_strings(self, svc):
        assert svc.text_overlap("", "") == 0.0


# ---------------------------------------------------------------------------
# texts_conflict
# ---------------------------------------------------------------------------

class TestTextsConflict:
    def test_negation_conflict_detected(self, svc):
        assert svc.texts_conflict(
            "Alice approved the budget.",
            "Alice did not approve the budget.",
        ) is True

    def test_consistent_texts_no_conflict(self, svc):
        assert svc.texts_conflict(
            "Alice approved the budget in Q3.",
            "Alice confirmed the budget allocation.",
        ) is False

    def test_low_overlap_not_flagged_as_conflict(self, svc):
        # Even with negation, unrelated subjects shouldn't conflict
        assert svc.texts_conflict(
            "Bob never submitted the report.",
            "Alice completed the project.",
        ) is False


# ---------------------------------------------------------------------------
# evaluate_write
# ---------------------------------------------------------------------------

class TestEvaluateWrite:
    def test_low_credibility_existing_is_superseded(self, svc):
        existing = _mem("Alice approved the budget.", credibility=0.30, ref_count=0)
        new_text = "Alice approved the Q3 budget allocation."
        decision = svc.evaluate_write(new_text, existing)
        assert decision.strategy == WriteStrategy.SUPERSEDE

    def test_load_bearing_conflict_preserved_both(self, svc):
        existing = _mem(
            "Alice approved the budget allocation for the project.",
            credibility=0.80,
            ref_count=5,
        )
        # Use "approved" and "budget" explicitly to ensure high token overlap
        new_text = "Alice did not approved the budget allocation for the project — rejected."
        decision = svc.evaluate_write(new_text, existing)
        assert decision.strategy == WriteStrategy.PRESERVE_BOTH

    def test_load_bearing_consistent_soft_update(self, svc):
        existing = _mem(
            "Alice approved the 2022 annual budget in Q4.",
            credibility=0.80,
            ref_count=4,
        )
        new_text = "Alice approved the annual budget in Q4 2022 after review."
        decision = svc.evaluate_write(new_text, existing)
        assert decision.strategy in (WriteStrategy.SOFT_UPDATE, WriteStrategy.PRESERVE_BOTH)

    def test_low_overlap_is_supersede_regardless(self, svc):
        existing = _mem("The weather was sunny.", credibility=0.90, ref_count=10)
        new_text = "Alice joined the engineering team."
        decision = svc.evaluate_write(new_text, existing)
        assert decision.strategy == WriteStrategy.SUPERSEDE

    def test_decision_has_overlap_score(self, svc):
        existing = _mem("Alice approved the plan.")
        new_text = "Alice approved the project plan."
        decision = svc.evaluate_write(new_text, existing)
        assert 0.0 <= decision.overlap_score <= 1.0


# ---------------------------------------------------------------------------
# batch_evaluate
# ---------------------------------------------------------------------------

class TestBatchEvaluate:
    def test_only_high_overlap_memories_returned(self, svc):
        candidates = [
            _mem("Alice approved the budget plan.", credibility=0.80, ref_count=2, mem_id="a"),
            _mem("The weather was fine today.", credibility=0.90, ref_count=5, mem_id="b"),
        ]
        decisions = svc.batch_evaluate("Alice approved the annual budget.", candidates)
        # Only "a" has high overlap
        assert all(d.existing_id == "a" for d in decisions)

    def test_empty_candidates_returns_empty(self, svc):
        assert svc.batch_evaluate("New text.", []) == []
