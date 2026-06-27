"""Unit tests for EpistemicProvenanceService — Phase 88."""
from __future__ import annotations

import pytest

from app.services.epistemic_provenance_service import EpistemicProvenanceService


@pytest.fixture
def svc() -> EpistemicProvenanceService:
    return EpistemicProvenanceService()


def _chunk(text: str, chunk_id: str = "x", source: str | None = None) -> dict:
    payload: dict = {"text": text}
    if source:
        payload["source"] = source
    return {"id": chunk_id, "payload": payload}


# ---------------------------------------------------------------------------
# source_fingerprint
# ---------------------------------------------------------------------------

class TestSourceFingerprint:
    def test_explicit_source_field_used(self, svc):
        c = _chunk("Some fact.", source="https://example.com/report")
        fp = svc.source_fingerprint(c)
        assert "example.com" in fp.origin_label

    def test_url_in_text_extracted(self, svc):
        c = _chunk("See https://arxiv.org/paper/123 for details.")
        fp = svc.source_fingerprint(c)
        assert "arxiv.org" in fp.origin_label

    def test_same_text_same_hash(self, svc):
        c1 = _chunk("The revenue grew by 30% in Q3 2022.", "a")
        c2 = _chunk("The revenue grew by 30% in Q3 2022.", "b")
        assert svc.source_fingerprint(c1).origin_hash == svc.source_fingerprint(c2).origin_hash

    def test_different_text_different_hash(self, svc):
        c1 = _chunk("Alice joined in 2021.", "a")
        c2 = _chunk("Bob left in 2022.", "b")
        assert svc.source_fingerprint(c1).origin_hash != svc.source_fingerprint(c2).origin_hash

    def test_citation_marker_detected(self, svc):
        c = _chunk("According to the annual report, revenue was $2B.")
        fp = svc.source_fingerprint(c)
        assert fp.is_derivative is True

    def test_primary_source_not_derivative(self, svc):
        c = _chunk("Revenue was $2B in fiscal 2022.")
        fp = svc.source_fingerprint(c)
        assert fp.is_derivative is False


# ---------------------------------------------------------------------------
# assess_provenance
# ---------------------------------------------------------------------------

class TestAssessProvenance:
    def test_single_source_is_flagged(self, svc):
        text = "The acquisition closed in Q2 2022 for $1B."
        chunks = [_chunk(text, str(i)) for i in range(5)]  # same text, 5 copies
        result = svc.assess_provenance(chunks, raw_credibility=0.80)
        assert result.is_single_source is True
        assert result.adjusted_credibility < 0.80

    def test_independent_sources_boost_credibility(self, svc):
        chunks = [
            _chunk("Acme acquired TechCorp.", "a", source="https://reuters.com/a"),
            _chunk("Acme acquired TechCorp.", "b", source="https://bloomberg.com/b"),
            _chunk("Acme acquired TechCorp.", "c", source="https://wsj.com/c"),
        ]
        result = svc.assess_provenance(chunks, raw_credibility=0.70)
        assert not result.is_single_source
        assert result.adjusted_credibility >= 0.70

    def test_empty_chunks_passthrough(self, svc):
        result = svc.assess_provenance([], raw_credibility=0.60)
        assert result.adjusted_credibility == 0.60
        assert result.total_chunks == 0

    def test_unique_origins_counted_correctly(self, svc):
        chunks = [
            _chunk("Fact A.", "1", source="http://src1.com"),
            _chunk("Fact B.", "2", source="http://src1.com"),
            _chunk("Fact C.", "3", source="http://src2.com"),
        ]
        result = svc.assess_provenance(chunks)
        assert result.unique_origins == 2
        assert result.total_chunks == 3

    def test_independence_score_range(self, svc):
        chunks = [_chunk(f"Fact {i}.", str(i), source=f"http://src{i}.com") for i in range(4)]
        result = svc.assess_provenance(chunks)
        assert 0.0 <= result.independence_score <= 1.0


# ---------------------------------------------------------------------------
# trace_citation_chain
# ---------------------------------------------------------------------------

class TestTraceCitationChain:
    def test_primary_source_identified(self, svc):
        chunks = [
            _chunk("Revenue grew by 30% in 2022.", source="http://annual-report.com"),
            _chunk("According to the annual report, revenue grew 30%.", "b"),
            _chunk("Per the company report, 30% growth was recorded.", "c"),
        ]
        chain = svc.trace_citation_chain(chunks)
        assert chain["primary_source"] is not None
        assert chain["chain_depth"] >= 1

    def test_chain_depth_zero_for_all_primaries(self, svc):
        chunks = [
            _chunk("Alice joined in 2021.", "a", source="http://src1.com"),
            _chunk("Alice started in 2021.", "b", source="http://src2.com"),
        ]
        chain = svc.trace_citation_chain(chunks)
        assert chain["chain_depth"] == 0

    def test_empty_chunks(self, svc):
        chain = svc.trace_citation_chain([])
        assert chain["primary_source"] == "unknown"
