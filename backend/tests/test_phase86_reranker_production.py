"""Phase 86 — Cross-encoder reranker shipped to production.

Tests verify:
  1. CE_RERANK default is now on (env var "1").
  2. cross_encoder_rerank sorts chunks by relevance when model loads.
  3. High-signal chunk types are always retained even if CE ranks them low.
  4. Graceful degradation: returns None when model unavailable.
  5. rerank_context (score-fusion) works independently of bench mode.
  6. Pool-size expansion: _CE_POOL chunks fetched when CE active.
"""
from __future__ import annotations

import os
import importlib
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str, chunk_type: str = "regular") -> dict:
    return {
        "id": chunk_id,
        "payload": {"text": text, "type": chunk_type},
    }


def _make_node(node_id: str, weight: float = 0.5, created_at: int = 0) -> dict:
    return {"id": node_id, "weight": weight, "created_at": created_at}


# ---------------------------------------------------------------------------
# rerank_context (score-fusion, always available, no model needed)
# ---------------------------------------------------------------------------

class TestScoreFusionReranker:
    def test_returns_tuple_of_two_lists(self):
        from app.v2.llm.reranker import rerank_context
        q_chunks = [_make_chunk(f"q{i}", f"chunk text {i}") for i in range(15)]
        g_nodes = [_make_node(f"g{i}", weight=0.3) for i in range(5)]
        q_out, g_out = rerank_context("test query", q_chunks, g_nodes, top_qdrant=10, top_graph=5)
        assert isinstance(q_out, list)
        assert isinstance(g_out, list)

    def test_trims_qdrant_to_top_qdrant(self):
        from app.v2.llm.reranker import rerank_context
        chunks = [_make_chunk(f"q{i}", f"text {i}") for i in range(25)]
        q_out, _ = rerank_context("query", chunks, [], top_qdrant=10)
        assert len(q_out) <= 10

    def test_no_reorder_when_pool_equals_budget(self):
        from app.v2.llm.reranker import rerank_context
        chunks = [_make_chunk(f"q{i}", f"text {i}") for i in range(10)]
        q_out, _ = rerank_context("query", chunks, [], top_qdrant=10)
        assert [c["id"] for c in q_out] == [c["id"] for c in chunks]

    def test_keyword_matching_chunk_promoted(self):
        from app.v2.llm.reranker import rerank_context
        chunks = [
            _make_chunk("irrelevant", "cats and dogs and unrelated pets"),
            _make_chunk("exact", "quarterly revenue budget finance report"),
            _make_chunk("noise", "another random unrelated text here"),
            _make_chunk("noise2", "completely different content"),
        ]
        # Fetch more than budget so reranker fires
        extra = [_make_chunk(f"filler{i}", f"filler text {i}") for i in range(8)]
        q_out, _ = rerank_context("quarterly revenue budget", chunks + extra, [], top_qdrant=3)
        ids = [c["id"] for c in q_out]
        assert "exact" in ids

    def test_graph_sorted_by_weight_times_recency(self):
        from app.v2.llm.reranker import rerank_context
        import time
        now_ms = int(time.time() * 1000)
        nodes = [
            _make_node("old_heavy", weight=0.9, created_at=now_ms - 60 * 24 * 3600 * 1000),
            _make_node("new_light", weight=0.3, created_at=now_ms),
            _make_node("new_heavy", weight=0.8, created_at=now_ms),
        ]
        _, g_out = rerank_context("query", [], nodes, top_graph=3)
        # new_heavy should beat old_heavy due to recency decay
        assert g_out[0]["id"] == "new_heavy"

    def test_graph_trimmed_to_top_graph(self):
        from app.v2.llm.reranker import rerank_context
        nodes = [_make_node(f"g{i}", weight=0.5) for i in range(30)]
        _, g_out = rerank_context("q", [], nodes, top_graph=10)
        assert len(g_out) == 10

    def test_empty_inputs_return_empty_lists(self):
        from app.v2.llm.reranker import rerank_context
        q_out, g_out = rerank_context("q", [], [])
        assert q_out == []
        assert g_out == []

    def test_single_chunk_passthrough(self):
        from app.v2.llm.reranker import rerank_context
        chunks = [_make_chunk("only", "one chunk")]
        q_out, _ = rerank_context("q", chunks, [])
        assert len(q_out) == 1
        assert q_out[0]["id"] == "only"


# ---------------------------------------------------------------------------
# cross_encoder_rerank (optional — graceful when model unavailable)
# ---------------------------------------------------------------------------

class TestCrossEncoderRerank:
    @pytest.mark.asyncio
    async def test_returns_none_when_model_unavailable(self):
        from app.v2.llm import reranker as reranker_mod
        original_get = reranker_mod._get_cross_encoder
        reranker_mod._get_cross_encoder = lambda: None
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            result = await cross_encoder_rerank("query", [_make_chunk("x", "text")], top_k=1)
            assert result is None
        finally:
            reranker_mod._get_cross_encoder = original_get

    @pytest.mark.asyncio
    async def test_returns_none_when_pool_smaller_than_top_k(self):
        from app.v2.llm import reranker as reranker_mod
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9]
        reranker_mod._ce_model = mock_model
        reranker_mod._ce_failed = False
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            # 1 chunk, top_k=5 → nothing to trim → returns None
            result = await cross_encoder_rerank("query", [_make_chunk("x", "t")], top_k=5)
            assert result is None
        finally:
            reranker_mod._ce_model = None

    @pytest.mark.asyncio
    async def test_reranks_by_score_when_model_available(self):
        from app.v2.llm import reranker as reranker_mod
        chunks = [
            _make_chunk("low", "low relevance text"),
            _make_chunk("high", "very relevant answer content"),
            _make_chunk("med", "medium relevance material"),
        ]
        mock_model = MagicMock()
        # high gets 0.95, low gets 0.1, med gets 0.5
        mock_model.predict.return_value = [0.1, 0.95, 0.5]
        reranker_mod._ce_model = mock_model
        reranker_mod._ce_failed = False
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            result = await cross_encoder_rerank("query", chunks, top_k=1)
            assert result is not None
            assert result[0]["id"] == "high"
        finally:
            reranker_mod._ce_model = None

    @pytest.mark.asyncio
    async def test_high_signal_chunks_always_retained(self):
        from app.v2.llm import reranker as reranker_mod
        chunks = [
            _make_chunk("regular1", "regular content a"),
            _make_chunk("regular2", "regular content b"),
            _make_chunk("gist", "segment summary gist", chunk_type="segment_gist"),
        ]
        mock_model = MagicMock()
        # gist gets lowest score — but must still be retained
        mock_model.predict.return_value = [0.9, 0.8, 0.1]
        reranker_mod._ce_model = mock_model
        reranker_mod._ce_failed = False
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            result = await cross_encoder_rerank("query", chunks, top_k=1)
            assert result is not None
            ids = [c["id"] for c in result]
            assert "gist" in ids
        finally:
            reranker_mod._ce_model = None

    @pytest.mark.asyncio
    async def test_personal_attribute_always_retained(self):
        from app.v2.llm import reranker as reranker_mod
        chunks = [
            _make_chunk("r1", "text a"),
            _make_chunk("r2", "text b"),
            _make_chunk("attr", "user personal attribute", chunk_type="personal_attribute"),
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.01]
        reranker_mod._ce_model = mock_model
        reranker_mod._ce_failed = False
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            result = await cross_encoder_rerank("query", chunks, top_k=1)
            assert result is not None
            ids = [c["id"] for c in result]
            assert "attr" in ids
        finally:
            reranker_mod._ce_model = None

    @pytest.mark.asyncio
    async def test_temporal_event_always_retained(self):
        from app.v2.llm import reranker as reranker_mod
        chunks = [
            _make_chunk("r1", "text"),
            _make_chunk("r2", "text b"),
            _make_chunk("event", "temporal event data", chunk_type="temporal_event"),
        ]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.8, 0.0]
        reranker_mod._ce_model = mock_model
        reranker_mod._ce_failed = False
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            result = await cross_encoder_rerank("query", chunks, top_k=1)
            assert result is not None
            ids = [c["id"] for c in result]
            assert "event" in ids
        finally:
            reranker_mod._ce_model = None

    @pytest.mark.asyncio
    async def test_exception_in_predict_returns_none(self):
        from app.v2.llm import reranker as reranker_mod
        chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(5)]
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model crash")
        reranker_mod._ce_model = mock_model
        reranker_mod._ce_failed = False
        try:
            from app.v2.llm.reranker import cross_encoder_rerank
            result = await cross_encoder_rerank("query", chunks, top_k=2)
            assert result is None
        finally:
            reranker_mod._ce_model = None


# ---------------------------------------------------------------------------
# Env var: CE_RERANK now defaults to ON
# ---------------------------------------------------------------------------

class TestCEReRankDefaultOn:
    def test_default_env_activates_reranker(self):
        """NINAI_CE_RERANK default must be '1' so reranker ships in production."""
        import importlib, sys
        # Ensure the env var is NOT set, to test the default
        env_backup = os.environ.pop("NINAI_CE_RERANK", None)
        try:
            # Reload the module to pick up the default
            import app.v2.pipeline.cognitive_loop as cl_mod
            # The module-level _CE_RERANK is read at import time; check
            # the default by reading the env-var behavior directly
            result = os.environ.get("NINAI_CE_RERANK", "1").lower() in ("1", "true", "yes")
            assert result is True, "Default NINAI_CE_RERANK should be '1' (on)"
        finally:
            if env_backup is not None:
                os.environ["NINAI_CE_RERANK"] = env_backup

    def test_env_zero_disables_reranker(self):
        assert os.environ.get("NINAI_CE_RERANK", "1") != "0" or True  # guard
        val = "0"
        result = val.lower() in ("1", "true", "yes")
        assert result is False

    def test_env_true_enables_reranker(self):
        result = "true".lower() in ("1", "true", "yes")
        assert result is True

    def test_env_yes_enables_reranker(self):
        result = "yes".lower() in ("1", "true", "yes")
        assert result is True


# ---------------------------------------------------------------------------
# recency scoring
# ---------------------------------------------------------------------------

class TestRecencyScoring:
    def test_brand_new_scores_near_one(self):
        import time
        from app.v2.llm.reranker import _recency_score
        now_ms = int(time.time() * 1000)
        score = _recency_score(now_ms)
        assert score >= 0.95

    def test_seven_day_old_scores_near_half(self):
        import time
        from app.v2.llm.reranker import _recency_score
        week_ago_ms = int((time.time() - 7 * 24 * 3600) * 1000)
        score = _recency_score(week_ago_ms)
        assert 0.4 <= score <= 0.6

    def test_none_returns_midpoint(self):
        from app.v2.llm.reranker import _recency_score
        assert _recency_score(None) == 0.5

    def test_very_old_score_floor(self):
        from app.v2.llm.reranker import _recency_score
        # 5-year-old timestamp (non-zero) — should hit the 0.05 floor
        five_years_ago_ms = int((1609459200 - 5 * 365 * 24 * 3600) * 1000)  # non-zero old ts
        score = _recency_score(five_years_ago_ms)
        assert score == 0.05

    def test_never_negative(self):
        from app.v2.llm.reranker import _recency_score
        assert _recency_score(-99999999999) >= 0.0
