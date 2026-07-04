"""Unit tests for the deterministic temporal-arithmetic helper (NINAI_TEMPORAL_MATH)."""
from __future__ import annotations

from app.v2.llm.temporal_arithmetic import compute_temporal_arithmetic


def _chunk(text: str, anchor_date: str | None = None) -> dict:
    payload = {"text": text}
    if anchor_date:
        payload["anchor_date"] = anchor_date
    return {"id": text[:8], "payload": payload}


class TestComputeTemporalArithmetic:
    def test_computes_gap_between_two_dated_events(self):
        chunks = [
            _chunk("We had a barbecue at the lake house.", "2023-04-10"),
            _chunk("Sarah's retirement party was held downtown.", "2023-05-02"),
        ]
        hint = compute_temporal_arithmetic(
            "How many days between the barbecue and the retirement party?", chunks
        )
        assert hint is not None
        assert "2023-04-10" in hint
        assert "2023-05-02" in hint
        assert "22 days" in hint

    def test_returns_none_when_not_a_duration_question(self):
        chunks = [
            _chunk("We had a barbecue.", "2023-04-10"),
            _chunk("The retirement party.", "2023-05-02"),
        ]
        assert compute_temporal_arithmetic(
            "What was served at the barbecue?", chunks
        ) is None

    def test_returns_none_when_no_chunks(self):
        assert compute_temporal_arithmetic("How many days between X and Y?", []) is None

    def test_returns_none_when_only_one_date_found(self):
        chunks = [_chunk("We had a barbecue.", "2023-04-10")]
        assert compute_temporal_arithmetic(
            "How many days between the barbecue and the retirement party?", chunks
        ) is None

    def test_returns_none_when_dates_identical(self):
        chunks = [
            _chunk("The barbecue and the retirement party happened the same day.", "2023-04-10"),
        ]
        assert compute_temporal_arithmetic(
            "How many days between the barbecue and the retirement party?", chunks
        ) is None

    def test_falls_back_to_regex_date_in_raw_text_when_no_anchor_field(self):
        chunks = [
            _chunk("[2023-04-10] We had a barbecue at the lake house."),
            _chunk("[2023-05-02] Sarah's retirement party was downtown."),
        ]
        hint = compute_temporal_arithmetic(
            "How many weeks between the barbecue and the retirement party?", chunks
        )
        assert hint is not None
        assert "22 days" in hint

    def test_from_to_anchor_not_confused_by_and_in_event_name(self):
        """Regression: the old code tried a bare '" and "' split BEFORE '" to "',
        so "from the trip Bob and Alice took to the wedding" split at "and"
        (wrong) instead of "to" (right), matching the wrong chunks and
        injecting a confidently wrong date hint. Anchoring on the explicit
        "from ... to ..." pattern picks the correct split regardless of an
        "and" appearing earlier in the event description."""
        chunks = [
            _chunk("Bob and Alice went on a trip to the lake.", "2023-04-10"),
            _chunk("The wedding was held downtown.", "2023-05-02"),
        ]
        hint = compute_temporal_arithmetic(
            "How many days from the trip Bob and Alice took to the wedding?", chunks
        )
        assert hint is not None
        assert "22 days" in hint

    def test_bare_and_without_anchor_keyword_does_not_trigger(self):
        """No 'between' or 'from...to' anchor at all — must not guess a split."""
        chunks = [
            _chunk("We had a barbecue at the lake house.", "2023-04-10"),
            _chunk("Sarah's retirement party was held downtown.", "2023-05-02"),
        ]
        assert compute_temporal_arithmetic(
            "How many days did the barbecue and the retirement party span?", chunks
        ) is None

    def test_from_to_anchor_computes_gap(self):
        chunks = [
            _chunk("We had a barbecue at the lake house.", "2023-04-10"),
            _chunk("Sarah's retirement party was held downtown.", "2023-05-02"),
        ]
        hint = compute_temporal_arithmetic(
            "How many days from the barbecue to the retirement party?", chunks
        )
        assert hint is not None
        assert "22 days" in hint

    def test_order_independent(self):
        chunks = [
            _chunk("Sarah's retirement party was downtown.", "2023-05-02"),
            _chunk("We had a barbecue at the lake house.", "2023-04-10"),
        ]
        hint = compute_temporal_arithmetic(
            "How many months between the retirement party and the barbecue?", chunks
        )
        assert hint is not None
        assert "22 days" in hint
