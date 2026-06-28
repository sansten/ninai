"""Phase 92 — MultiHopRetrievalService tests."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, patch

import pytest

from app.services.multi_hop_retrieval_service import (
    MultiHopRetrievalService,
    MultiHopResult,
    HopResult,
)


# ---------------------------------------------------------------------------
# Fake ReadResult for testing
# ---------------------------------------------------------------------------

@dataclass
class FakeReadResult:
    qdrant_chunks: list[dict] = field(default_factory=list)


def _make_chunks(texts: list[str], id_prefix: str = "c") -> list[dict]:
    return [
        {"id": f"{id_prefix}-{i}", "payload": {"text": t, "type": "regular"}}
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# Chain detection
# ---------------------------------------------------------------------------

class TestChainDetection:
    def test_who_reported_to_person_is_chain(self):
        svc = MultiHopRetrievalService()
        assert svc._is_chain_question("Who reported to the person who approved the budget?")

    def test_who_approved_is_chain(self):
        svc = MultiHopRetrievalService()
        assert svc._is_chain_question("Who approved the request that led to the incident?")

    def test_led_to_is_chain(self):
        svc = MultiHopRetrievalService()
        assert svc._is_chain_question("What decision led to the project delay?")

    def test_simple_question_not_chain(self):
        svc = MultiHopRetrievalService()
        assert not svc._is_chain_question("What is Alice's job title?")

    def test_single_fact_question_not_chain(self):
        svc = MultiHopRetrievalService()
        assert not svc._is_chain_question("When did the team launch the product?")


# ---------------------------------------------------------------------------
# Anchor extraction
# ---------------------------------------------------------------------------

class TestAnchorExtraction:
    def test_extracts_proper_nouns(self):
        svc = MultiHopRetrievalService()
        anchors = svc._extract_anchors("Alice Johnson approved the budget with Bob Smith.")
        assert "Alice Johnson" in anchors or "Alice" in anchors

    def test_filters_stopwords(self):
        svc = MultiHopRetrievalService()
        anchors = svc._extract_anchors("The person who did the work")
        assert "The" not in anchors

    def test_empty_text_returns_empty(self):
        svc = MultiHopRetrievalService()
        assert svc._extract_anchors("") == []

    def test_multi_word_proper_noun(self):
        svc = MultiHopRetrievalService()
        anchors = svc._extract_anchors("The Engineering Department approved the New York project.")
        combined = " ".join(anchors)
        assert "Engineering" in combined or "York" in combined


# ---------------------------------------------------------------------------
# Basic retrieval flow (heuristic mode)
# ---------------------------------------------------------------------------

class TestHeuristicRetrievalFlow:
    @pytest.mark.asyncio
    async def test_single_hop_no_chain_entities(self):
        async def retrieve(q):
            return FakeReadResult(qdrant_chunks=_make_chunks(["fact a", "fact b"]))

        svc = MultiHopRetrievalService(max_hops=2)
        result = await svc.retrieve("simple question", retrieve)
        assert result.n_hops_taken >= 1
        assert isinstance(result, MultiHopResult)

    @pytest.mark.asyncio
    async def test_original_query_used_for_hop0(self):
        queries_seen = []

        async def retrieve(q):
            queries_seen.append(q)
            return FakeReadResult(qdrant_chunks=_make_chunks([f"result for {q}"]))

        svc = MultiHopRetrievalService(max_hops=1)
        await svc.retrieve("find Alice's manager", retrieve)
        assert queries_seen[0] == "find Alice's manager"

    @pytest.mark.asyncio
    async def test_deduplication_across_hops(self):
        call_count = [0]

        async def retrieve(q):
            call_count[0] += 1
            # Always return the same chunk
            return FakeReadResult(qdrant_chunks=[{"id": "same-id", "payload": {"text": "same content"}}])

        svc = MultiHopRetrievalService(max_hops=2)
        result = await svc.retrieve("question", retrieve)
        # Despite multiple hops, the duplicate chunk appears only once
        assert len(result.all_chunks) == 1

    @pytest.mark.asyncio
    async def test_all_chunks_aggregated(self):
        hop_idx = [0]

        async def retrieve(q):
            hop_idx[0] += 1
            return FakeReadResult(qdrant_chunks=_make_chunks(["text"], id_prefix=f"h{hop_idx[0]}"))

        svc = MultiHopRetrievalService(max_hops=2)
        result = await svc.retrieve("Alice's budget approval", retrieve)
        # All hops contribute to all_chunks
        assert result.total_chunks_retrieved >= 1

    @pytest.mark.asyncio
    async def test_empty_first_hop_stops_chain(self):
        async def retrieve(q):
            return FakeReadResult(qdrant_chunks=[])

        svc = MultiHopRetrievalService(max_hops=3)
        result = await svc.retrieve("question", retrieve)
        # If hop 0 returns nothing, chain stops
        assert result.total_chunks_retrieved == 0

    @pytest.mark.asyncio
    async def test_max_hops_respected(self):
        hop_count = [0]

        async def retrieve(q):
            hop_count[0] += 1
            # Return enough chunks to keep going, with different IDs
            return FakeReadResult(
                qdrant_chunks=[{"id": f"hop{hop_count[0]}-{i}", "payload": {"text": f"Alice Brown {i}"}}
                                for i in range(5)]
            )

        svc = MultiHopRetrievalService(max_hops=2)
        await svc.retrieve("chain question", retrieve)
        # Should not exceed max_hops + 1 calls
        assert hop_count[0] <= 3

    @pytest.mark.asyncio
    async def test_min_new_chunks_threshold_stops_chain(self):
        hop_count = [0]

        async def retrieve(q):
            hop_count[0] += 1
            if hop_count[0] == 1:
                # First hop: many chunks with names as anchors
                return FakeReadResult(
                    qdrant_chunks=[{"id": f"h1-{i}", "payload": {"text": "Alice Brown Manager"}}
                                   for i in range(10)]
                )
            # Subsequent hops: fewer than threshold new chunks
            return FakeReadResult(qdrant_chunks=[{"id": f"h{hop_count[0]}-new", "payload": {"text": "tiny"}}])

        svc = MultiHopRetrievalService(max_hops=3, min_new_chunks_to_continue=3)
        result = await svc.retrieve("chain question", retrieve)
        # Should stop after second hop (only 1 new chunk < threshold 3)
        assert hop_count[0] <= 3

    @pytest.mark.asyncio
    async def test_retrieval_exception_is_graceful(self):
        async def retrieve(q):
            raise RuntimeError("backend unavailable")

        svc = MultiHopRetrievalService(max_hops=1)
        result = await svc.retrieve("question", retrieve)
        assert result.total_chunks_retrieved == 0
        assert result.n_hops_taken >= 1

    @pytest.mark.asyncio
    async def test_chain_rationale_populated(self):
        async def retrieve(q):
            return FakeReadResult(qdrant_chunks=_make_chunks(["Alice Smith budget"]))

        svc = MultiHopRetrievalService(max_hops=1)
        result = await svc.retrieve("Who approved the budget?", retrieve)
        assert result.chain_rationale
        assert "Hop 0" in result.chain_rationale

    @pytest.mark.asyncio
    async def test_hop_results_have_rationale(self):
        async def retrieve(q):
            return FakeReadResult(qdrant_chunks=_make_chunks(["Alice Smith approved"]))

        svc = MultiHopRetrievalService(max_hops=1)
        result = await svc.retrieve("budget chain question", retrieve)
        assert result.hops[0].rationale


# ---------------------------------------------------------------------------
# LLM-guided mode
# ---------------------------------------------------------------------------

class TestLLMGuidedRetrievalFlow:
    @pytest.mark.asyncio
    async def test_llm_next_query_used(self):
        queries_seen = []

        async def retrieve(q):
            queries_seen.append(q)
            return FakeReadResult(
                qdrant_chunks=[{"id": f"c-{len(queries_seen)}", "payload": {"text": "result"}}]
            )

        async def inference_fn(prompt):
            return "Who is the Q3 budget owner"

        svc = MultiHopRetrievalService(inference_fn=inference_fn, max_hops=1)
        await svc.retrieve("who approved the Q3 budget", retrieve)
        assert "Who is the Q3 budget owner" in queries_seen

    @pytest.mark.asyncio
    async def test_llm_empty_response_falls_back_to_heuristic(self):
        async def retrieve(q):
            return FakeReadResult(
                qdrant_chunks=[{"id": "x", "payload": {"text": "Alice Budget Manager"}}]
            )

        async def inference_fn(prompt):
            return ""   # LLM returns nothing

        svc = MultiHopRetrievalService(inference_fn=inference_fn, max_hops=1)
        result = await svc.retrieve("chain question", retrieve)
        # Falls back to heuristic — should still produce a result
        assert result.n_hops_taken >= 1

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_heuristic(self):
        async def retrieve(q):
            return FakeReadResult(
                qdrant_chunks=[{"id": "y", "payload": {"text": "Alice Smith approved"}}]
            )

        async def inference_fn(prompt):
            raise RuntimeError("LLM crash")

        svc = MultiHopRetrievalService(inference_fn=inference_fn, max_hops=1)
        result = await svc.retrieve("chain question", retrieve)
        assert result is not None


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestMultiHopResultStructure:
    @pytest.mark.asyncio
    async def test_is_chain_question_set(self):
        async def retrieve(q):
            return FakeReadResult()

        svc = MultiHopRetrievalService()
        result = await svc.retrieve("Who reported to the person who approved?", retrieve)
        assert result.is_chain_question is True

    @pytest.mark.asyncio
    async def test_non_chain_is_chain_question_false(self):
        async def retrieve(q):
            return FakeReadResult()

        svc = MultiHopRetrievalService()
        result = await svc.retrieve("What is the capital of France?", retrieve)
        assert result.is_chain_question is False

    @pytest.mark.asyncio
    async def test_original_query_preserved(self):
        async def retrieve(q):
            return FakeReadResult()

        svc = MultiHopRetrievalService()
        result = await svc.retrieve("my original question", retrieve)
        assert result.original_query == "my original question"

    @pytest.mark.asyncio
    async def test_n_hops_at_least_one(self):
        async def retrieve(q):
            return FakeReadResult(qdrant_chunks=_make_chunks(["text"]))

        svc = MultiHopRetrievalService(max_hops=2)
        result = await svc.retrieve("q", retrieve)
        assert result.n_hops_taken >= 1
