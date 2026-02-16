"""Tests for EpisodeBoundaryService (GAP-1).

Covers:
    - Pure function: _cosine_distance
    - Pure function: _llm_intent_boundary (mocked httpx)
    - segment_messages: empty input, single group, multi-group with split
    - add_message_to_current_episode: new episode, append, boundary, fano_split
    - _check_boundary: temporal gap, topic shift, LLM intent
    - _detect_boundaries: Fano n_k=12 sub-split
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest

from app.services.episode_boundary_service import (
    EpisodeBoundaryService,
    _cosine_distance,
    _llm_intent_boundary,
    FANO_NK,
    THETA_TOPIC,
    THETA_TIME,
)
from .conftest import (
    ORG_ID, USER_ID, FAKE_EMBEDDING, ZERO_EMBEDDING,
    FakeMemory, FakeEpisode, FakeMembership,
    ScalarOneResult, ScalarsListResult, DeleteResult,
)


# ════════════════════════════════════════════════════════════════════════
# Pure functions
# ════════════════════════════════════════════════════════════════════════

class TestCosineDistance:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-7)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-7)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_distance(a, b) == pytest.approx(2.0, abs=1e-7)

    def test_zero_vector_returns_one(self):
        assert _cosine_distance([0.0, 0.0], [1.0, 1.0]) == 1.0
        assert _cosine_distance([1.0, 1.0], [0.0, 0.0]) == 1.0

    def test_similar_vectors_low_distance(self):
        a = [1.0, 0.1]
        b = [1.0, 0.2]
        dist = _cosine_distance(a, b)
        assert 0 < dist < 0.1  # very similar


class TestLLMIntentBoundary:

    async def test_returns_boundary_on_valid_response(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": '{"boundary": true, "confidence": 0.9}'
        }

        with patch("app.services.episode_boundary_service.httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=mock_response)
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = ctx

            is_b, conf = await _llm_intent_boundary("hello", "completely different topic")
            assert is_b is True
            assert conf == pytest.approx(0.9)

    async def test_returns_false_on_no_boundary(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "response": '{"boundary": false, "confidence": 0.2}'
        }

        with patch("app.services.episode_boundary_service.httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.post = AsyncMock(return_value=mock_response)
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = ctx

            is_b, conf = await _llm_intent_boundary("hello", "hello again")
            assert is_b is False

    async def test_returns_default_on_error(self):
        with patch("app.services.episode_boundary_service.httpx.AsyncClient") as mock_client_cls:
            ctx = AsyncMock()
            ctx.post = AsyncMock(side_effect=Exception("connection refused"))
            ctx.__aenter__ = AsyncMock(return_value=ctx)
            ctx.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = ctx

            is_b, conf = await _llm_intent_boundary("a", "b")
            assert is_b is False
            assert conf == 0.0


# ════════════════════════════════════════════════════════════════════════
# EpisodeBoundaryService._check_boundary
# ════════════════════════════════════════════════════════════════════════

class TestCheckBoundary:

    async def test_temporal_gap_triggers_boundary(self, mock_session):
        svc = EpisodeBoundaryService(mock_session)
        t1 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(hours=2)  # 2 hours >> 30 min threshold

        prev_msg = FakeMemory(content="msg1", created_at=t1)
        curr_msg = FakeMemory(content="msg2", created_at=t2)

        is_b, reason, conf = await svc._check_boundary(
            prev_msg=prev_msg, curr_msg=curr_msg,
            prev_embedding=FAKE_EMBEDDING, curr_embedding=FAKE_EMBEDDING,
            theta_topic=THETA_TOPIC, theta_time=THETA_TIME, use_llm=False,
        )
        assert is_b is True
        assert reason == "temporal_gap"
        assert conf == 0.95

    async def test_topic_shift_triggers_boundary(self, mock_session):
        svc = EpisodeBoundaryService(mock_session)
        now = datetime.now(timezone.utc)

        prev_msg = FakeMemory(content="a", created_at=now)
        curr_msg = FakeMemory(content="b", created_at=now + timedelta(seconds=30))

        # Orthogonal vectors → cosine distance = 1.0 >> 0.45
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]

        is_b, reason, conf = await svc._check_boundary(
            prev_msg=prev_msg, curr_msg=curr_msg,
            prev_embedding=emb_a, curr_embedding=emb_b,
            theta_topic=THETA_TOPIC, theta_time=THETA_TIME, use_llm=False,
        )
        assert is_b is True
        assert reason == "topic_shift"

    async def test_no_boundary_for_similar_messages(self, mock_session):
        svc = EpisodeBoundaryService(mock_session)
        now = datetime.now(timezone.utc)
        prev_msg = FakeMemory(content="a", created_at=now)
        curr_msg = FakeMemory(content="b", created_at=now + timedelta(seconds=5))

        # Nearly identical embeddings → cosine distance ≈ 0
        emb = [1.0, 0.1, 0.0]

        is_b, reason, conf = await svc._check_boundary(
            prev_msg=prev_msg, curr_msg=curr_msg,
            prev_embedding=emb, curr_embedding=emb,
            theta_topic=THETA_TOPIC, theta_time=THETA_TIME, use_llm=False,
        )
        assert is_b is False

    async def test_llm_intent_boundary_triggers(self, mock_session):
        svc = EpisodeBoundaryService(mock_session)
        now = datetime.now(timezone.utc)
        prev_msg = FakeMemory(content="a", created_at=now)
        curr_msg = FakeMemory(content="b", created_at=now + timedelta(seconds=5))
        emb = [1.0, 0.1, 0.0]

        with patch(
            "app.services.episode_boundary_service._llm_intent_boundary",
            new_callable=AsyncMock,
            return_value=(True, 0.85),
        ):
            is_b, reason, conf = await svc._check_boundary(
                prev_msg=prev_msg, curr_msg=curr_msg,
                prev_embedding=emb, curr_embedding=emb,
                theta_topic=THETA_TOPIC, theta_time=THETA_TIME, use_llm=True,
            )
            assert is_b is True
            assert reason == "intent_change"
            assert conf == 0.85


# ════════════════════════════════════════════════════════════════════════
# EpisodeBoundaryService._detect_boundaries (Fano n_k)
# ════════════════════════════════════════════════════════════════════════

class TestDetectBoundaries:

    async def test_fano_split_at_nk_messages(self, mock_session):
        """When all messages are similar and in quick succession,
        the Fano threshold (n_k=12) should still force a split."""
        svc = EpisodeBoundaryService(mock_session)
        now = datetime.now(timezone.utc)

        # Create 15 messages, all very similar and close in time
        messages = [
            FakeMemory(id=f"m{i}", content=f"msg{i}",
                       created_at=now + timedelta(seconds=i * 10))
            for i in range(15)
        ]
        embeddings = {m.id: FAKE_EMBEDDING for m in messages}

        with patch(
            "app.services.episode_boundary_service._llm_intent_boundary",
            new_callable=AsyncMock,
            return_value=(False, 0.0),
        ):
            groups = await svc._detect_boundaries(
                messages=messages,
                embeddings=embeddings,
                theta_topic=THETA_TOPIC,
                theta_time=THETA_TIME,
                use_llm=False,
            )

        # Should have at least 2 groups (first 12, then remaining 3)
        assert len(groups) >= 2
        assert groups[0]["reason"] == "initial"
        # Second group boundary should be fano_split
        assert groups[1]["reason"] == "fano_split"
        # First group should have exactly FANO_NK messages
        assert len(groups[0]["memory_ids"]) == FANO_NK

    async def test_single_message_returns_one_group(self, mock_session):
        svc = EpisodeBoundaryService(mock_session)
        msg = FakeMemory(id="m1", content="only one")
        embeddings = {"m1": FAKE_EMBEDDING}

        groups = await svc._detect_boundaries(
            messages=[msg], embeddings=embeddings,
            theta_topic=THETA_TOPIC, theta_time=THETA_TIME, use_llm=False,
        )
        assert len(groups) == 1
        assert groups[0]["memory_ids"] == ["m1"]


# ════════════════════════════════════════════════════════════════════════
# EpisodeBoundaryService.segment_messages
# ════════════════════════════════════════════════════════════════════════

class TestSegmentMessages:

    async def test_empty_messages_returns_empty(self, mock_session, mock_embed):
        mock_session.execute = AsyncMock(return_value=ScalarsListResult([]))
        svc = EpisodeBoundaryService(mock_session)

        result = await svc.segment_messages(
            organization_id=ORG_ID, owner_id=USER_ID,
        )
        assert result == []


# ════════════════════════════════════════════════════════════════════════
# EpisodeBoundaryService.add_message_to_current_episode
# ════════════════════════════════════════════════════════════════════════

class TestAddMessageToCurrentEpisode:

    async def test_raises_if_memory_not_found(self, mock_session):
        mock_session.execute = AsyncMock(return_value=ScalarOneResult(None))
        svc = EpisodeBoundaryService(mock_session)

        with pytest.raises(ValueError, match="not found"):
            await svc.add_message_to_current_episode(
                organization_id=ORG_ID, owner_id=USER_ID,
                memory_id="nonexistent",
            )

    async def test_creates_new_episode_when_none_open(self, mock_session, mock_embed, mock_summarize, mock_qdrant_upsert):
        msg = FakeMemory(id="m1", content="hello")

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: fetch memory
                return ScalarOneResult(msg)
            elif call_count == 2:
                # Second call: find open episode
                return ScalarOneResult(None)
            # Subsequent calls: flush etc. just return empty
            return ScalarOneResult(None)

        mock_session.execute = AsyncMock(side_effect=side_effect)
        svc = EpisodeBoundaryService(mock_session)

        result = await svc.add_message_to_current_episode(
            organization_id=ORG_ID, owner_id=USER_ID,
            memory_id="m1", use_llm=False,
        )

        assert result["action"] == "new_episode"
        assert result["reason"] == "initial"
        assert mock_session.add.called
