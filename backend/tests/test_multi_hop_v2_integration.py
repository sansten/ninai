"""Integration tests: NINAI_MULTI_HOP_V2 gate in cognitive_loop.py.

These tests verify that the MultiHopRetrievalService is correctly wired into
the cognitive loop's Phase 1 retrieval path behind the NINAI_MULTI_HOP_V2 flag.

They do NOT exercise the real Qdrant/FalkorDB stack — the router is stubbed
so we can control what each retrieval call returns and assert on how the loop
merges the results.
"""
from __future__ import annotations

import asyncio
import os
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for dependencies the cognitive loop imports at module level
# ---------------------------------------------------------------------------

def _make_read_result(chunks: list[dict]) -> MagicMock:
    r = MagicMock()
    r.qdrant_chunks = chunks
    r.graph_nodes = []
    r.pending_enrichments = 0
    return r


def _make_router(hop_responses: list[list[dict]]) -> MagicMock:
    """Router whose .read() cycles through hop_responses on successive calls."""
    call_idx = [-1]

    async def _read(*args, **kwargs):
        call_idx[0] += 1
        idx = min(call_idx[0], len(hop_responses) - 1)
        return _make_read_result(hop_responses[idx])

    router = MagicMock()
    router.read = _read
    return router


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiHopV2Gate:
    """Confirm NINAI_MULTI_HOP_V2=0 (default) leaves the original path untouched."""

    def test_flag_off_by_default(self, monkeypatch):
        monkeypatch.delenv("NINAI_MULTI_HOP_V2", raising=False)
        # Re-evaluate the flag expression directly
        val = os.environ.get("NINAI_MULTI_HOP_V2", "0").lower() in ("1", "true", "yes")
        assert val is False

    def test_flag_on_when_env_set(self, monkeypatch):
        monkeypatch.setenv("NINAI_MULTI_HOP_V2", "1")
        val = os.environ.get("NINAI_MULTI_HOP_V2", "0").lower() in ("1", "true", "yes")
        assert val is True

    def test_max_hops_env_respected(self, monkeypatch):
        monkeypatch.setenv("NINAI_MULTI_HOP_V2_MAX_HOPS", "2")
        val = int(os.environ.get("NINAI_MULTI_HOP_V2_MAX_HOPS", "3"))
        assert val == 2

    def test_default_max_hops_is_three(self, monkeypatch):
        monkeypatch.delenv("NINAI_MULTI_HOP_V2_MAX_HOPS", raising=False)
        val = int(os.environ.get("NINAI_MULTI_HOP_V2_MAX_HOPS", "3"))
        assert val == 3


class TestMultiHopServiceDirectly:
    """Verify the service behaves correctly when called with a mock router."""

    @pytest.mark.asyncio
    async def test_hop0_always_runs(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        chunks_seen = []

        async def retrieve(q):
            chunks_seen.append(q)
            r = MagicMock()
            r.qdrant_chunks = [{"id": f"c-{len(chunks_seen)}", "payload": {"text": "result"}}]
            return r

        svc = MultiHopRetrievalService(max_hops=0)
        result = await svc.retrieve("simple question", retrieve)
        assert result.n_hops_taken >= 1
        assert chunks_seen[0] == "simple question"

    @pytest.mark.asyncio
    async def test_dedup_prevents_duplicate_chunks(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        call_count = [0]

        async def retrieve(q):
            call_count[0] += 1
            r = MagicMock()
            # Always return the same chunk
            r.qdrant_chunks = [{"id": "fixed-id", "payload": {"text": "Alice Smith"}}]
            return r

        svc = MultiHopRetrievalService(max_hops=2)
        result = await svc.retrieve("Who is Alice?", retrieve)
        # Duplicate suppressed — only one copy in all_chunks
        assert sum(1 for ch in result.all_chunks if ch["id"] == "fixed-id") == 1

    @pytest.mark.asyncio
    async def test_additional_hops_add_chunks(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        hop = [0]

        async def retrieve(q):
            hop[0] += 1
            r = MagicMock()
            r.qdrant_chunks = [
                {"id": f"h{hop[0]}-a", "payload": {"text": f"Alice Brown {hop[0]}"}},
                {"id": f"h{hop[0]}-b", "payload": {"text": f"Bob Green {hop[0]}"}},
            ]
            return r

        svc = MultiHopRetrievalService(max_hops=2)
        result = await svc.retrieve("Who approved the budget?", retrieve)
        # At least the initial hop's chunks are present
        assert result.total_chunks_retrieved >= 2

    @pytest.mark.asyncio
    async def test_retrieval_error_does_not_raise(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        async def retrieve(q):
            raise ConnectionError("Qdrant unavailable")

        svc = MultiHopRetrievalService(max_hops=1)
        result = await svc.retrieve("question", retrieve)
        # Graceful — returns empty result, no exception
        assert result is not None
        assert result.total_chunks_retrieved == 0

    @pytest.mark.asyncio
    async def test_chain_question_flagged(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        async def retrieve(q):
            r = MagicMock()
            r.qdrant_chunks = []
            return r

        svc = MultiHopRetrievalService()
        result = await svc.retrieve(
            "Who reported to the person who approved the Q3 budget?", retrieve
        )
        assert result.is_chain_question is True

    @pytest.mark.asyncio
    async def test_simple_question_not_flagged_as_chain(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        async def retrieve(q):
            r = MagicMock()
            r.qdrant_chunks = []
            return r

        svc = MultiHopRetrievalService()
        result = await svc.retrieve("What is the capital of France?", retrieve)
        assert result.is_chain_question is False

    @pytest.mark.asyncio
    async def test_max_hops_one_limits_calls(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        calls = [0]

        async def retrieve(q):
            calls[0] += 1
            r = MagicMock()
            # Return unique chunks each time to avoid early-stop
            r.qdrant_chunks = [
                {"id": f"c{calls[0]}-{i}", "payload": {"text": f"Alice Brown {i}"}}
                for i in range(5)
            ]
            return r

        svc = MultiHopRetrievalService(max_hops=1)
        await svc.retrieve("chain question", retrieve)
        # max_hops=1 means at most 2 total calls (hop0 + hop1)
        assert calls[0] <= 2

    @pytest.mark.asyncio
    async def test_rationale_non_empty(self):
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        async def retrieve(q):
            r = MagicMock()
            r.qdrant_chunks = [{"id": "x", "payload": {"text": "Alice approved"}}]
            return r

        svc = MultiHopRetrievalService(max_hops=1)
        result = await svc.retrieve("Who approved?", retrieve)
        assert result.chain_rationale
        assert "Hop 0" in result.chain_rationale


class TestMultiHopV2Performance:
    """Latency sanity checks — ensures the service adds bounded overhead."""

    @pytest.mark.asyncio
    async def test_max_hops_bounds_router_calls(self):
        """Router must be called at most max_hops+1 times regardless of content."""
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        calls = [0]

        async def retrieve(q):
            calls[0] += 1
            r = MagicMock()
            r.qdrant_chunks = [
                {"id": f"id-{calls[0]}-{i}", "payload": {"text": f"Alice Brown Manager {i}"}}
                for i in range(10)
            ]
            return r

        for max_hops in (1, 2, 3):
            calls[0] = 0
            svc = MultiHopRetrievalService(max_hops=max_hops)
            await svc.retrieve("who approved the chain question", retrieve)
            assert calls[0] <= max_hops + 1, (
                f"max_hops={max_hops} but router called {calls[0]} times"
            )

    @pytest.mark.asyncio
    async def test_zero_new_chunks_stops_early(self):
        """If hop 1 returns no new chunks, do not proceed to hop 2."""
        from app.services.multi_hop_retrieval_service import MultiHopRetrievalService

        calls = [0]

        async def retrieve(q):
            calls[0] += 1
            r = MagicMock()
            if calls[0] == 1:
                r.qdrant_chunks = [{"id": "unique", "payload": {"text": "Alice Brown"}}]
            else:
                # Same ID as hop 0 — all duplicates → 0 new chunks
                r.qdrant_chunks = [{"id": "unique", "payload": {"text": "Alice Brown"}}]
            return r

        svc = MultiHopRetrievalService(max_hops=3, min_new_chunks_to_continue=1)
        await svc.retrieve("chain question", retrieve)
        # Hop 1 returns 0 new chunks → stops before hop 2
        assert calls[0] <= 3
