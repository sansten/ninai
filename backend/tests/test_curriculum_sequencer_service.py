"""Unit tests for CurriculumSequencerService — Phase 89."""
from __future__ import annotations

import pytest

from app.services.curriculum_sequencer_service import (
    CurriculumSequencerService,
    KnowledgeGap,
)


@pytest.fixture
def svc() -> CurriculumSequencerService:
    return CurriculumSequencerService()


def _chunk(text: str, score: float = 0.5) -> dict:
    return {"id": hash(text), "score": score, "payload": {"text": text}}


# ---------------------------------------------------------------------------
# identify_gaps_from_chunks
# ---------------------------------------------------------------------------

class TestIdentifyGapsFromChunks:
    def test_missing_entity_detected(self, svc):
        chunks = [_chunk("The event was held in London.")]
        gaps = svc.identify_gaps_from_chunks("Where did Alice go?", chunks)
        gap_concepts = {g.concept for g in gaps}
        assert "Alice" in gap_concepts

    def test_low_confidence_gap_detected(self, svc):
        chunks = [_chunk("Alice was somewhere.", score=0.20)]
        gaps = svc.identify_gaps_from_chunks("Where did Alice go?", chunks)
        gap_types = {g.gap_type for g in gaps}
        assert "missing" in gap_types or "low_confidence" in gap_types

    def test_no_gaps_when_context_covers_question(self, svc):
        chunks = [_chunk("Alice went to Paris in 2022.", score=0.90)]
        gaps = svc.identify_gaps_from_chunks("Where did Alice go?", chunks)
        # Alice and Paris are both in context — should have no missing gaps for these
        missing = [g for g in gaps if g.gap_type == "missing" and g.concept == "Alice"]
        assert len(missing) == 0

    def test_suggested_query_is_non_empty(self, svc):
        chunks = [_chunk("Nothing relevant here.")]
        gaps = svc.identify_gaps_from_chunks("Who is Alice Johnson?", chunks)
        for gap in gaps:
            assert gap.suggested_query.strip() != ""


# ---------------------------------------------------------------------------
# build_acquisition_plan
# ---------------------------------------------------------------------------

class TestBuildAcquisitionPlan:
    def test_returns_plan_with_ordered_gaps(self, svc):
        gaps = [
            KnowledgeGap("Alice", "missing", 0.0, 0.8, 3, "Alice background"),
            KnowledgeGap("Bob", "low_confidence", 0.3, 0.5, 0, "Bob details"),
            KnowledgeGap("ProjectX", "conflicted", 0.2, 0.9, 5, "ProjectX clarification"),
        ]
        plan = svc.build_acquisition_plan(gaps)
        assert plan.total_gaps == 3
        # ProjectX has high importance and blocking_count — should rank first
        assert plan.gaps[0].concept in ("Alice", "ProjectX")

    def test_estimated_reduction_in_range(self, svc):
        gaps = [
            KnowledgeGap("Alice", "missing", 0.0, 0.8, 2, "query"),
        ]
        plan = svc.build_acquisition_plan(gaps)
        assert 0.0 <= plan.estimated_uncertainty_reduction <= 1.0

    def test_dependency_graph_increases_blocking_count(self, svc):
        gaps = [
            KnowledgeGap("Alice", "missing", 0.0, 0.8, 0, "Alice query"),
            KnowledgeGap("Bob", "missing", 0.0, 0.7, 0, "Bob query"),
        ]
        dep_graph = {"Alice": ["Bob", "ProjectX"]}  # Alice blocks 2 concepts
        plan = svc.build_acquisition_plan(gaps, dependency_graph=dep_graph)
        alice_gap = next(g for g in plan.gaps if g.concept == "Alice")
        assert alice_gap.blocking_count >= 2

    def test_empty_gaps_returns_empty_plan(self, svc):
        plan = svc.build_acquisition_plan([])
        assert plan.total_gaps == 0


# ---------------------------------------------------------------------------
# identify_gaps_from_state
# ---------------------------------------------------------------------------

class TestIdentifyGapsFromState:
    def test_converts_uncertain_concepts_to_gaps(self, svc):
        state = [
            {"concept": "Q4 Revenue", "confidence": 0.20, "importance": 0.90},
            {"concept": "Alice Johnson", "confidence": 0.50, "importance": 0.60},
        ]
        gaps = svc.identify_gaps_from_state(state)
        assert len(gaps) == 2
        assert gaps[0].concept == "Q4 Revenue"

    def test_skips_entries_without_concept(self, svc):
        state = [{"confidence": 0.10, "importance": 0.80}]  # missing concept
        gaps = svc.identify_gaps_from_state(state)
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# Transitive blocking
# ---------------------------------------------------------------------------

class TestTransitiveBlocking:
    def test_chain_counted_correctly(self, svc):
        graph = {"A": ["B"], "B": ["C"], "C": []}
        count = svc._count_transitive_blocked("A", graph)
        assert count == 2  # A blocks B, B blocks C

    def test_no_deps_returns_zero(self, svc):
        assert svc._count_transitive_blocked("X", {}) == 0
