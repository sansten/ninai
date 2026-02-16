"""Tests for SemanticDistillationService (GAP-1 Level 3).

Covers:
    - Pure function: _geometric_mean_4
    - Pure function: _sha256
    - distill_episode: episode not found, empty messages, quality gate, dedup
    - distill_batch: skips already distilled, processes closed episodes
    - _extract_facts: mocked LLM response parsing
"""

from __future__ import annotations

import hashlib
import math
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.semantic_distillation_service import (
    SemanticDistillationService,
    _geometric_mean_4,
    _sha256,
    TAU_QUALITY,
)
from .conftest import (
    ORG_ID, USER_ID, FAKE_EMBEDDING,
    FakeMemory, FakeEpisode, FakeSemanticNode,
    ScalarOneResult, ScalarsListResult,
)


# ════════════════════════════════════════════════════════════════════════
# Pure functions
# ════════════════════════════════════════════════════════════════════════

class TestGeometricMean4:

    def test_all_ones_returns_one(self):
        assert _geometric_mean_4(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_all_zeros_returns_zero(self):
        assert _geometric_mean_4(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_mixed_values(self):
        # (0.8 * 0.6 * 0.7 * 0.9) ^ 0.25
        product = 0.8 * 0.6 * 0.7 * 0.9
        expected = product ** 0.25
        assert _geometric_mean_4(0.8, 0.6, 0.7, 0.9) == pytest.approx(expected, abs=1e-6)

    def test_one_zero_returns_zero(self):
        """If any score is 0, the geometric mean is 0."""
        assert _geometric_mean_4(0.9, 0.0, 0.8, 0.7) == 0.0

    def test_negative_clamped_to_zero(self):
        """Negative scores are clamped via max(0, x)."""
        assert _geometric_mean_4(-0.5, 0.8, 0.7, 0.6) == 0.0

    def test_typical_passing_scores(self):
        """Scores that should pass the TAU_QUALITY=0.55 gate."""
        q = _geometric_mean_4(0.7, 0.7, 0.7, 0.7)
        assert q == pytest.approx(0.7)
        assert q > TAU_QUALITY

    def test_typical_failing_scores(self):
        """Scores that should fail the TAU_QUALITY gate."""
        q = _geometric_mean_4(0.3, 0.3, 0.3, 0.3)
        assert q == pytest.approx(0.3)
        assert q < TAU_QUALITY


class TestSha256:

    def test_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_different_inputs(self):
        assert _sha256("hello") != _sha256("world")

    def test_matches_hashlib(self):
        text = "test content"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert _sha256(text) == expected

    def test_length(self):
        assert len(_sha256("any string")) == 64


# ════════════════════════════════════════════════════════════════════════
# SemanticDistillationService.distill_episode
# ════════════════════════════════════════════════════════════════════════

class TestDistillEpisode:

    async def test_raises_if_episode_not_found(self, mock_session):
        mock_session.execute = AsyncMock(return_value=ScalarOneResult(None))
        svc = SemanticDistillationService(mock_session)

        with pytest.raises(ValueError, match="not found"):
            await svc.distill_episode("bad-id", organization_id=ORG_ID)

    async def test_returns_empty_for_episode_with_no_messages(self, mock_session):
        episode = FakeEpisode(status="closed")

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ScalarOneResult(episode)
            # Membership query returns empty
            return ScalarsListResult([])

        mock_session.execute = AsyncMock(side_effect=side_effect)
        svc = SemanticDistillationService(mock_session)

        result = await svc.distill_episode(episode.id, organization_id=ORG_ID)
        assert result == []

    async def test_quality_gate_filters_low_quality_facts(self, mock_session, mock_embed, mock_qdrant_upsert):
        episode = FakeEpisode(status="closed")
        msg = FakeMemory(content="some discussion")

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ScalarOneResult(episode)
            elif call_count == 2:
                # Membership memory_ids
                return ScalarsListResult([msg.id])
            elif call_count == 3:
                # Actual messages
                return ScalarsListResult([msg])
            # Dedup check: no existing
            return ScalarOneResult(None)

        mock_session.execute = AsyncMock(side_effect=side_effect)
        svc = SemanticDistillationService(mock_session)

        # Mock LLM to return a low-quality fact
        with patch.object(svc, "_extract_facts", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                {
                    "fact": "Something vague",
                    "persistence": 0.2,
                    "specificity": 0.1,
                    "utility": 0.3,
                    "independence": 0.2,
                    "entities": [],
                    "tags": [],
                }
            ]
            result = await svc.distill_episode(episode.id, organization_id=ORG_ID)
            # geometric_mean(0.2, 0.1, 0.3, 0.2) ≈ 0.19 < 0.55
            assert result == []

    async def test_high_quality_fact_creates_node(self, mock_session, mock_embed, mock_qdrant_upsert):
        episode = FakeEpisode(status="closed")
        msg = FakeMemory(content="important info")

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ScalarOneResult(episode)
            elif call_count == 2:
                return ScalarsListResult([msg.id])
            elif call_count == 3:
                return ScalarsListResult([msg])
            # Dedup check: no existing node
            return ScalarOneResult(None)

        mock_session.execute = AsyncMock(side_effect=side_effect)
        svc = SemanticDistillationService(mock_session)

        with patch.object(svc, "_extract_facts", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                {
                    "fact": "User prefers dark mode",
                    "persistence": 0.9,
                    "specificity": 0.8,
                    "utility": 0.85,
                    "independence": 0.9,
                    "entities": ["user"],
                    "tags": ["preference"],
                }
            ]
            result = await svc.distill_episode(episode.id, organization_id=ORG_ID)
            assert len(result) == 1
            assert result[0]["content"] == "User prefers dark mode"
            assert result[0]["quality"] > TAU_QUALITY
            assert mock_session.add.called


# ════════════════════════════════════════════════════════════════════════
# SemanticDistillationService._extract_facts (LLM mock)
# ════════════════════════════════════════════════════════════════════════

class TestExtractFacts:

    async def test_parses_valid_json_array(self, mock_session):
        svc = SemanticDistillationService(mock_session)
        msgs = [FakeMemory(content="User said they love pizza")]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": '[{"fact": "User loves pizza", "persistence": 0.8, "specificity": 0.7, "utility": 0.6, "independence": 0.9, "entities": ["user"], "tags": ["food"]}]'
        }

        with patch("app.services.semantic_distillation_service.httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=mock_response)
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = ctx

            facts = await svc._extract_facts(msgs)
            assert len(facts) == 1
            assert facts[0]["fact"] == "User loves pizza"

    async def test_returns_empty_on_llm_error(self, mock_session):
        svc = SemanticDistillationService(mock_session)
        msgs = [FakeMemory(content="test")]

        with patch("app.services.semantic_distillation_service.httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.post = AsyncMock(side_effect=Exception("timeout"))
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = ctx

            facts = await svc._extract_facts(msgs)
            assert facts == []

    async def test_returns_empty_on_invalid_json(self, mock_session):
        svc = SemanticDistillationService(mock_session)
        msgs = [FakeMemory(content="test")]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": "I cannot extract any facts from this conversation."
        }

        with patch("app.services.semantic_distillation_service.httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=mock_response)
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = ctx

            facts = await svc._extract_facts(msgs)
            assert facts == []


# ════════════════════════════════════════════════════════════════════════
# SemanticDistillationService.distill_batch
# ════════════════════════════════════════════════════════════════════════

class TestDistillBatch:

    async def test_returns_stats_with_no_episodes(self, mock_session):
        mock_session.execute = AsyncMock(return_value=ScalarsListResult([]))
        svc = SemanticDistillationService(mock_session)

        stats = await svc.distill_batch(organization_id=ORG_ID)
        assert stats == {"episodes_processed": 0, "nodes_created": 0}

    async def test_skips_already_distilled_episodes(self, mock_session):
        ep = FakeEpisode(status="closed")
        existing_node = FakeSemanticNode(source_episode_ids=[ep.id])

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Episodes query
                return ScalarsListResult([ep])
            elif call_count == 2:
                # Check if already distilled → yes
                return ScalarOneResult(existing_node)
            return ScalarOneResult(None)

        mock_session.execute = AsyncMock(side_effect=side_effect)
        svc = SemanticDistillationService(mock_session)

        stats = await svc.distill_batch(organization_id=ORG_ID)
        assert stats["episodes_processed"] == 0
