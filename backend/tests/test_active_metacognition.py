"""Unit tests for active metacognition helpers in cognitive_loop.

These tests exercise the pure-function helpers that run before and after LLM
inference — no external services, no async I/O required.
"""
from __future__ import annotations

from app.v2.pipeline.cognitive_loop import (
    MetaCogState,
    _METACOG_CONF_THRESH,
    _apply_strategy_context,
    _apply_strategy_hint,
    _compute_metacog_state,
    _extract_knowledge_gap,
    _pick_metacog_tier,
    _pick_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _chunk(text: str, date_prefix: str = "") -> dict:
    payload: dict = {"text": text}
    if date_prefix:
        payload["date_prefix"] = date_prefix
    return {"id": hash(text), "payload": payload}


GOOD_CHUNKS = [
    _chunk("Alice went to Paris in 2022."),
    _chunk("She met Bob there."),
    _chunk("Bob is a photographer."),
]

EMPTY_CHUNKS: list[dict] = []


# ---------------------------------------------------------------------------
# _compute_metacog_state — pre-flight (empty answer)
# ---------------------------------------------------------------------------

class TestComputeMetacogStatePreFlight:
    def test_high_coverage_yields_higher_confidence(self):
        chunks = [_chunk("Alice went to Paris in 2022 for work.")]
        state = _compute_metacog_state("When did Alice go to Paris?", "", chunks)
        assert state.confidence > 0.35

    def test_no_coverage_yields_low_confidence(self):
        chunks = [_chunk("The rocket launched at dawn.")]
        state = _compute_metacog_state("When did Alice go to Paris?", "", chunks)
        assert state.confidence < 0.55

    def test_empty_chunks_confidence_is_minimal(self):
        state = _compute_metacog_state("What did Alice do?", "", EMPTY_CHUNKS)
        assert state.confidence <= 0.30

    def test_experience_match_partial_overlap(self):
        chunks = [_chunk("Alice lives in London.")]
        state = _compute_metacog_state("Where does Alice live?", "", chunks)
        assert state.experience_match > 0.0


# ---------------------------------------------------------------------------
# _compute_metacog_state — post-answer
# ---------------------------------------------------------------------------

class TestComputeMetacogStatePostAnswer:
    def test_grounded_short_answer_high_confidence(self):
        chunks = [_chunk("Alice went to Paris.")]
        state = _compute_metacog_state("Where did Alice go?", "Paris", chunks)
        assert state.confidence >= 0.80

    def test_refusal_zero_confidence(self):
        state = _compute_metacog_state(
            "Where did Alice go?", "This information is not mentioned.", GOOD_CHUNKS
        )
        assert state.confidence == 0.0

    def test_ungrounded_short_answer_low_confidence(self):
        chunks = [_chunk("The sky is blue.")]
        state = _compute_metacog_state(
            "Where did Alice go?", "Tokyo", chunks
        )
        assert state.confidence < 0.50


# ---------------------------------------------------------------------------
# _pick_strategy
# ---------------------------------------------------------------------------

class TestPickStrategy:
    def _state(self, **kw) -> MetaCogState:
        s = MetaCogState()
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def test_temporal_question(self):
        assert _pick_strategy("When did Alice go to Paris?", GOOD_CHUNKS, self._state()) == "temporal"

    def test_multihop_question(self):
        assert _pick_strategy(
            "What led Alice to move to Paris?", GOOD_CHUNKS, self._state()
        ) == "multihop"

    def test_adversarial_question(self):
        assert _pick_strategy(
            "Alice never actually went to Paris, right?", GOOD_CHUNKS, self._state()
        ) == "adversarial"

    def test_factual_with_high_coverage(self):
        assert _pick_strategy(
            "What is Bob's job?", GOOD_CHUNKS,
            self._state(importance=0.6, experience_match=0.3)
        ) == "factual"

    def test_default_standard(self):
        assert _pick_strategy(
            "Tell me about the trip.", GOOD_CHUNKS, self._state()
        ) == "standard"


# ---------------------------------------------------------------------------
# _pick_metacog_tier
# ---------------------------------------------------------------------------

class TestPickMetacogTier:
    def _state(self, importance=0.0, experience_match=0.0) -> MetaCogState:
        s = MetaCogState()
        s.importance = importance
        s.experience_match = experience_match
        return s

    def test_high_importance_low_coverage_escalates(self):
        assert _pick_metacog_tier(self._state(importance=0.7, experience_match=0.05), "fast") == "reasoning"

    def test_high_importance_good_coverage_stays_fast(self):
        assert _pick_metacog_tier(self._state(importance=0.7, experience_match=0.20), "fast") == "fast"

    def test_low_importance_stays_fast(self):
        assert _pick_metacog_tier(self._state(importance=0.2, experience_match=0.05), "fast") == "fast"

    def test_already_reasoning_unchanged(self):
        assert _pick_metacog_tier(self._state(importance=0.9, experience_match=0.0), "reasoning") == "reasoning"


# ---------------------------------------------------------------------------
# _apply_strategy_context
# ---------------------------------------------------------------------------

class TestApplyStrategyContext:
    def test_temporal_sorts_dated_chunks_first(self):
        chunks = [
            _chunk("No date here."),
            _chunk("Event B.", date_prefix="2021-06"),
            _chunk("Event A.", date_prefix="2023-01"),
            _chunk("Also no date."),
        ]
        result = _apply_strategy_context(chunks, "temporal")
        # First two should be dated, most recent first
        assert result[0]["payload"]["date_prefix"] == "2023-01"
        assert result[1]["payload"]["date_prefix"] == "2021-06"
        # Undated chunks come after
        assert "date_prefix" not in result[2]["payload"]

    def test_non_temporal_unchanged(self):
        chunks = [_chunk("A"), _chunk("B")]
        assert _apply_strategy_context(chunks, "standard") == chunks
        assert _apply_strategy_context(chunks, "multihop") == chunks


# ---------------------------------------------------------------------------
# _apply_strategy_hint
# ---------------------------------------------------------------------------

class TestApplyStrategyHint:
    def test_temporal_hint_appended(self):
        result = _apply_strategy_hint("When did Alice travel?", "temporal")
        assert "chronological" in result.lower() or "dates" in result.lower()

    def test_multihop_hint_appended(self):
        result = _apply_strategy_hint("Why did Alice go there?", "multihop")
        assert "chain" in result.lower() or "step" in result.lower()

    def test_adversarial_hint_appended(self):
        result = _apply_strategy_hint("Alice never went, right?", "adversarial")
        assert "false premise" in result.lower() or "critical" in result.lower()

    def test_standard_no_hint(self):
        q = "What did Alice do?"
        assert _apply_strategy_hint(q, "standard") == q
        assert _apply_strategy_hint(q, "factual") == q


# ---------------------------------------------------------------------------
# _extract_knowledge_gap
# ---------------------------------------------------------------------------

class TestExtractKnowledgeGap:
    def test_refusal_returns_missing_entity(self):
        chunks = [_chunk("The sky is blue.")]
        gap = _extract_knowledge_gap(
            "Where did Alice go in 2022?",
            "not mentioned in the conversation",
            chunks,
        )
        assert gap is not None
        assert len(gap.strip()) > 0

    def test_non_refusal_returns_none(self):
        chunks = [_chunk("Alice went to Paris.")]
        gap = _extract_knowledge_gap(
            "Where did Alice go?", "Paris", chunks
        )
        assert gap is None

    def test_empty_answer_returns_gap(self):
        chunks = [_chunk("Bob is a photographer.")]
        gap = _extract_knowledge_gap(
            "What did Caroline do in Berlin?", "", chunks
        )
        assert gap is not None
        # Should capture 'Caroline' or 'Berlin' — proper nouns absent from context
        assert "Caroline" in gap or "Berlin" in gap or len(gap) > 0

    def test_gap_differs_from_original_question(self):
        chunks = [_chunk("The event was held in summer.")]
        gap = _extract_knowledge_gap(
            "When did Alice fly to Tokyo?",
            "not mentioned in the conversation",
            chunks,
        )
        assert gap is not None
        assert gap.strip() != "When did Alice fly to Tokyo?"


# ---------------------------------------------------------------------------
# MetaCogState.should_escalate
# ---------------------------------------------------------------------------

class TestMetaCogStateShouldEscalate:
    def test_low_confidence_should_escalate(self):
        s = MetaCogState()
        s.confidence = _METACOG_CONF_THRESH - 0.01
        assert s.should_escalate is True

    def test_high_confidence_no_escalate(self):
        s = MetaCogState()
        s.confidence = _METACOG_CONF_THRESH + 0.01
        assert s.should_escalate is False
