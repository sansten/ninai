"""Unit tests for PredictionErrorService — Phase 85."""
from __future__ import annotations

import pytest

from app.services.prediction_error_service import (
    DIVERGENCE_THRESHOLD,
    Anticipation,
    PredictionErrorService,
)


@pytest.fixture
def svc() -> PredictionErrorService:
    return PredictionErrorService()


def _chunk(text: str) -> dict:
    return {"id": hash(text), "payload": {"text": text}}


# ---------------------------------------------------------------------------
# anticipate() — pre-inference expectation
# ---------------------------------------------------------------------------

class TestAnticipate:
    def test_date_question(self, svc):
        a = svc.anticipate("When did Alice go to Paris?", [_chunk("Alice went to Paris in 2022.")])
        assert a.category == "date"

    def test_entity_question(self, svc):
        a = svc.anticipate("Who introduced Bob to Alice?", [_chunk("Bob met Alice at the conference.")])
        assert a.category == "entity"

    def test_boolean_question(self, svc):
        a = svc.anticipate("Did Alice attend the event?", [_chunk("Alice was present.")])
        assert a.category == "boolean"

    def test_narrative_question(self, svc):
        a = svc.anticipate("Describe what happened at the summit.", [_chunk("The summit was eventful.")])
        assert a.category == "narrative"

    def test_high_coverage_raises_confidence(self, svc):
        chunks = [
            _chunk("Alice went to Paris for a work conference in June 2022."),
            _chunk("The work trip was in Paris."),
            _chunk("Alice attended the Paris conference."),
        ]
        a = svc.anticipate("When did Alice go to Paris for work?", chunks)
        assert a.confidence > 0.35

    def test_no_coverage_low_confidence(self, svc):
        chunks = [_chunk("The weather was sunny.")]
        a = svc.anticipate("When did Alice go to Paris?", chunks)
        assert a.confidence < 0.50

    def test_empty_chunks_minimal_confidence(self, svc):
        a = svc.anticipate("Where did Bob go?", [])
        assert a.confidence <= 0.30


# ---------------------------------------------------------------------------
# measure() — divergence calculation
# ---------------------------------------------------------------------------

class TestMeasure:
    def _antici(self, category: str, confidence: float) -> Anticipation:
        return Anticipation(category=category, confidence=confidence)

    def test_matching_answer_low_divergence(self, svc):
        a = self._antici("entity", 0.70)
        r = svc.measure(a, "Alice Johnson", 0.80)
        assert r.divergence_score < DIVERGENCE_THRESHOLD
        assert not r.is_surprising

    def test_refusal_when_high_confidence_is_surprising(self, svc):
        a = self._antici("entity", 0.80)
        r = svc.measure(a, "not mentioned in the conversation", 0.10)
        assert r.is_surprising
        assert r.divergence_score >= DIVERGENCE_THRESHOLD

    def test_category_mismatch_raises_divergence(self, svc):
        a = self._antici("date", 0.70)
        r = svc.measure(a, "Alice is a researcher who joined in spring", 0.60)
        assert r.divergence_score > 0.20

    def test_confidence_collapse_adds_penalty(self, svc):
        a = self._antici("narrative", 0.85)
        r = svc.measure(a, "Bob", 0.10)
        # confidence dropped from 0.85 → 0.10 = collapse of 0.75 * 0.35 = ~0.26
        assert r.divergence_score > 0.20

    def test_result_fields(self, svc):
        a = self._antici("date", 0.60)
        r = svc.measure(a, "2022", 0.75)
        assert r.expected_category == "date"
        assert r.actual_category == "date"
        assert 0.0 <= r.divergence_score <= 1.0

    def test_empty_answer_is_refusal(self, svc):
        a = self._antici("entity", 0.80)
        r = svc.measure(a, "", 0.0)
        assert r.is_surprising


# ---------------------------------------------------------------------------
# _classify_expected_category (via anticipate)
# ---------------------------------------------------------------------------

class TestCategoryClassification:
    def test_when_question_is_date(self, svc):
        a = svc.anticipate("What year did Alice join?", [])
        assert a.category == "date"

    def test_who_question_is_entity(self, svc):
        a = svc.anticipate("Who is the project lead?", [])
        assert a.category == "entity"

    def test_did_question_is_boolean(self, svc):
        a = svc.anticipate("Did Bob approve the plan?", [])
        assert a.category == "boolean"

    def test_describe_question_is_narrative(self, svc):
        a = svc.anticipate("Explain the key decisions made.", [])
        assert a.category == "narrative"


# ---------------------------------------------------------------------------
# is_surprising flag
# ---------------------------------------------------------------------------

class TestIsSurprising:
    def test_above_threshold_is_surprising(self, svc):
        a = Anticipation(category="date", confidence=0.90)
        r = svc.measure(a, "not provided", 0.05)
        assert r.is_surprising

    def test_below_threshold_not_surprising(self, svc):
        a = Anticipation(category="entity", confidence=0.55)
        r = svc.measure(a, "Alice", 0.70)
        assert not r.is_surprising
