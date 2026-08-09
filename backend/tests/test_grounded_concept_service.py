"""Unit tests for GroundedConceptService — Phase 91."""
from __future__ import annotations

from datetime import date

import pytest

from app.services.grounded_concept_service import (
    GroundedConceptService,
    GroundedConcept,
)


@pytest.fixture
def svc() -> GroundedConceptService:
    return GroundedConceptService()


# ---------------------------------------------------------------------------
# Temporal extraction
# ---------------------------------------------------------------------------

class TestTemporalExtraction:
    def test_iso_date(self, svc):
        concepts = svc.extract("The deal closed on 2022-06-15.")
        temporal = [c for c in concepts if c.grounding_type == "temporal"]
        assert any(c.normalized_value == date(2022, 6, 15) for c in temporal)

    def test_month_day_year(self, svc):
        concepts = svc.extract("Alice joined on March 5, 2021.")
        temporal = [c for c in concepts if c.grounding_type == "temporal"]
        assert any(c.normalized_value == date(2021, 3, 5) for c in temporal)

    def test_year_only(self, svc):
        concepts = svc.extract("The company was founded in 2019.")
        temporal = [c for c in concepts if c.grounding_type == "temporal"]
        assert any(c.normalized_value.year == 2019 for c in temporal)

    def test_quarter_year(self, svc):
        concepts = svc.extract("Revenue was reported in Q3 2022.")
        temporal = [c for c in concepts if c.grounding_type == "temporal"]
        assert any(c.normalized_value.year == 2022 for c in temporal)


# ---------------------------------------------------------------------------
# Scalar extraction
# ---------------------------------------------------------------------------

class TestScalarExtraction:
    def test_money_with_multiplier(self, svc):
        concepts = svc.extract("Acme paid $500M for the startup.")
        scalars = [c for c in concepts if c.grounding_type == "scalar" and c.unit == "$"]
        assert any(c.normalized_value == 500_000_000 for c in scalars)

    def test_percentage(self, svc):
        concepts = svc.extract("Revenue grew by 30%.")
        scalars = [c for c in concepts if c.grounding_type == "scalar" and c.unit == "%"]
        assert any(c.normalized_value == 30.0 for c in scalars)

    def test_count_with_unit(self, svc):
        concepts = svc.extract("The team grew to 200 employees in 2022.")
        scalars = [c for c in concepts if c.grounding_type == "scalar"]
        assert any(c.normalized_value == 200.0 for c in scalars)


# ---------------------------------------------------------------------------
# Ordinal extraction
# ---------------------------------------------------------------------------

class TestOrdinalExtraction:
    def test_first_extracted(self, svc):
        concepts = svc.extract("Alice was the first to join the team.")
        ordinals = [c for c in concepts if c.grounding_type == "ordinal"]
        assert any(c.normalized_value == 1 for c in ordinals)

    def test_last_extracted(self, svc):
        concepts = svc.extract("Bob was the last to leave.")
        ordinals = [c for c in concepts if c.grounding_type == "ordinal"]
        assert any("last" in str(c.normalized_value) for c in ordinals)


# ---------------------------------------------------------------------------
# Boolean extraction
# ---------------------------------------------------------------------------

class TestBooleanExtraction:
    def test_affirmative_detected(self, svc):
        concepts = svc.extract("The proposal was approved.")
        booleans = [c for c in concepts if c.grounding_type == "boolean"]
        assert any(c.normalized_value is True for c in booleans)

    def test_negative_detected(self, svc):
        concepts = svc.extract("The request was rejected by the board.")
        booleans = [c for c in concepts if c.grounding_type == "boolean"]
        assert any(c.normalized_value is False for c in booleans)


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------

class TestCompare:
    def _date_concept(self, d: date) -> GroundedConcept:
        return GroundedConcept(str(d), "temporal", d, None, 0.9)

    def _scalar_concept(self, value: float, unit: str = "$") -> GroundedConcept:
        return GroundedConcept(str(value), "scalar", value, unit, 0.85)

    def test_temporal_before(self, svc):
        a = self._date_concept(date(2021, 1, 1))
        b = self._date_concept(date(2022, 6, 1))
        result = svc.compare(a, b)
        assert result.relation == "before"

    def test_temporal_after(self, svc):
        a = self._date_concept(date(2023, 1, 1))
        b = self._date_concept(date(2022, 1, 1))
        result = svc.compare(a, b)
        assert result.relation == "after"

    def test_temporal_equal(self, svc):
        d = date(2022, 3, 15)
        result = svc.compare(self._date_concept(d), self._date_concept(d))
        assert result.relation == "equal"

    def test_scalar_greater(self, svc):
        result = svc.compare(self._scalar_concept(1000), self._scalar_concept(500))
        assert result.relation == "greater"

    def test_scalar_less(self, svc):
        result = svc.compare(self._scalar_concept(100), self._scalar_concept(200))
        assert result.relation == "less"

    def test_incomparable_types(self, svc):
        temporal = self._date_concept(date(2022, 1, 1))
        scalar = self._scalar_concept(500)
        result = svc.compare(temporal, scalar)
        assert result.relation == "incomparable"


# ---------------------------------------------------------------------------
# enrich_chunk
# ---------------------------------------------------------------------------

class TestEnrichChunk:
    def test_adds_groundings_to_payload(self, svc):
        chunk = {"id": "1", "payload": {"text": "Alice approved $500M deal in 2022-03-15."}}
        enriched = svc.enrich_chunk(chunk)
        assert "groundings" in enriched["payload"]
        assert len(enriched["payload"]["groundings"]) > 0

    def test_original_chunk_unchanged(self, svc):
        chunk = {"id": "1", "payload": {"text": "Alice joined in 2021."}}
        _ = svc.enrich_chunk(chunk)
        assert "groundings" not in chunk["payload"]

    def test_grounding_entries_have_required_keys(self, svc):
        chunk = {"id": "1", "payload": {"text": "Revenue grew 30% in Q2 2023."}}
        enriched = svc.enrich_chunk(chunk)
        for g in enriched["payload"]["groundings"]:
            assert "type" in g and "value" in g and "confidence" in g
