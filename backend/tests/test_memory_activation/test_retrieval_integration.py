"""Integration tests for Memory Activation Scoring Retrieval

Tests cover:
- Score and rank results
- Activation state loading
- Metadata loading
- Explanation log writing
- Scope/episode/goal matching
- Age computation

These tests are Postgres-backed (with migrations + RLS). Enable with:
  RUN_POSTGRES_TESTS=1
Unit tests for ActivationScorer and task functions are in test_scorer.py and test_tasks.py
"""

from __future__ import annotations

import os
import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import set_tenant_context
from app.models.memory import MemoryMetadata
from app.services.memory_activation.retrieval import MemoryRetrievalService
from app.models.memory_activation import MemoryActivationState, MemoryRetrievalExplanation


RUN_POSTGRES_TESTS = os.environ.get("RUN_POSTGRES_TESTS", "").lower() in {"1", "true", "yes"}
requires_postgres = pytest.mark.skipif(not RUN_POSTGRES_TESTS, reason="Set RUN_POSTGRES_TESTS=1 to run")

class TestMemoryRetrievalService:
    """Test MemoryRetrievalService scoring and ranking."""

    @pytest.fixture
    def service(self) -> MemoryRetrievalService:
        """Create a retrieval service for unit-only tests.

        Note: Some tests exercise pure helper methods and must not require a
        Postgres fixture; use a mocked AsyncSession.
        """
        return MemoryRetrievalService(
            session=AsyncMock(spec=AsyncSession),
            org_id=str(uuid4()),
            user_id=str(uuid4()),
        )

    @pytest.mark.asyncio
    async def test_score_empty_results(self, service: MemoryRetrievalService):
        """Test scoring empty result set."""
        ranked, explanations = await service.score_and_rank_results(
            memory_ids=[],
            query="test",
            similarities={},
        )
        assert ranked == []
        assert explanations == []

    @requires_postgres
    @pytest.mark.asyncio
    async def test_score_single_memory(
        self,
        async_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
        test_org,
        test_user,
    ):
        """Test scoring single memory."""
        service = MemoryRetrievalService(session=async_session, org_id=test_org_id, user_id=test_user_id)

        mem_id = str(uuid4())
        async with async_session.begin():
            await set_tenant_context(async_session, user_id=test_user_id, org_id=test_org_id)

            async_session.add(
                MemoryMetadata(
                    id=mem_id,
                    organization_id=test_org_id,
                    owner_id=test_user_id,
                    scope="personal",
                    content_preview="integration memory",
                    content_hash=uuid4().hex,
                )
            )
            async_session.add(
                MemoryActivationState(
                    organization_id=test_org_id,
                    memory_id=mem_id,
                    base_importance=0.8,
                    confidence=0.9,
                )
            )
            await async_session.flush()

            ranked, explanations = await service.score_and_rank_results(
                memory_ids=[mem_id],
                query="test query",
                similarities={mem_id: 0.85},
            )

        assert len(ranked) == 1
        assert len(explanations) == 1
        assert ranked[0]["id"] == mem_id
        assert ranked[0]["activation_score"] > 0.0

    @requires_postgres
    @pytest.mark.asyncio
    async def test_ranking_order(
        self,
        async_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
        test_org,
        test_user,
    ):
        """Test memories ranked by activation descending."""
        service = MemoryRetrievalService(session=async_session, org_id=test_org_id, user_id=test_user_id)

        mem_ids = [str(uuid4()) for _ in range(3)]

        async with async_session.begin():
            await set_tenant_context(async_session, user_id=test_user_id, org_id=test_org_id)

            for mem_id in mem_ids:
                async_session.add(
                    MemoryMetadata(
                        id=mem_id,
                        organization_id=test_org_id,
                        owner_id=test_user_id,
                        scope="personal",
                        content_preview=f"integration memory {mem_id[:8]}",
                        content_hash=uuid4().hex,
                    )
                )

            async_session.add_all(
                [
                    MemoryActivationState(
                        organization_id=test_org_id,
                        memory_id=mem_ids[0],
                        base_importance=0.3,
                    ),
                    MemoryActivationState(
                        organization_id=test_org_id,
                        memory_id=mem_ids[1],
                        base_importance=0.8,
                    ),
                    MemoryActivationState(
                        organization_id=test_org_id,
                        memory_id=mem_ids[2],
                        base_importance=0.5,
                    ),
                ]
            )

            await async_session.flush()

            similarities = {mid: 0.8 for mid in mem_ids}

            ranked, _ = await service.score_and_rank_results(
                memory_ids=mem_ids,
                query="test",
                similarities=similarities,
            )

        # Should be ordered by importance (descending)
        assert ranked[0]["id"] == mem_ids[1]  # importance 0.8
        assert ranked[1]["id"] == mem_ids[2]  # importance 0.5
        assert ranked[2]["id"] == mem_ids[0]  # importance 0.3

    @requires_postgres
    @pytest.mark.asyncio
    async def test_explanation_log(
        self,
        async_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
        test_org,
        test_user,
    ):
        """Test explanation log writing."""
        service = MemoryRetrievalService(session=async_session, org_id=test_org_id, user_id=test_user_id)

        mem_id = str(uuid4())

        async with async_session.begin():
            await set_tenant_context(async_session, user_id=test_user_id, org_id=test_org_id)

            async_session.add(
                MemoryMetadata(
                    id=mem_id,
                    organization_id=test_org_id,
                    owner_id=test_user_id,
                    scope="personal",
                    content_preview="integration memory",
                    content_hash=uuid4().hex,
                )
            )
            async_session.add(
                MemoryActivationState(
                    organization_id=test_org_id,
                    memory_id=mem_id,
                    base_importance=0.7,
                )
            )
            await async_session.flush()

            _ranked, explanations = await service.score_and_rank_results(
                memory_ids=[mem_id],
                query="test query",
                similarities={mem_id: 0.9},
            )

            log_id = await service.write_retrieval_explanation(
                query="test query",
                results=explanations,
                top_k=10,
            )

            assert log_id is not None

            from sqlalchemy import select

            stmt = select(MemoryRetrievalExplanation).where(MemoryRetrievalExplanation.id == log_id)
            result = await async_session.execute(stmt)
            log_entry = result.scalar_one()

            assert str(log_entry.user_id) == str(service.user_id)
            assert log_entry.top_k == 10
            assert len(log_entry.results) == 1

    @pytest.mark.asyncio
    async def test_scope_matching(self, service: MemoryRetrievalService):
        """Test scope affinity computation."""
        metadata = {"scope": "team"}

        # Same scope
        match = service._compute_scope_match("team", metadata)
        assert match == 1.0

        # Broader scope
        match = service._compute_scope_match("organization", metadata)
        assert match == 0.7

        # Narrower scope
        match = service._compute_scope_match("personal", metadata)
        assert match < 0.5

    @pytest.mark.asyncio
    async def test_episode_matching(self, service: MemoryRetrievalService):
        """Test episode affinity computation."""
        episode_id = str(uuid4())
        metadata = {"episode_id": episode_id}

        # Same episode
        match = service._compute_episode_match(episode_id, metadata)
        assert match == 1.0

        # Different episode
        match = service._compute_episode_match(str(uuid4()), metadata)
        assert match == 0.3

    @pytest.mark.asyncio
    async def test_age_computation(self, service: MemoryRetrievalService):
        """Test memory age computation."""
        now = datetime.now(timezone.utc)

        # Recent (1 hour ago)
        recent = now - timedelta(hours=1)
        age = service._compute_age_days(recent)
        assert 0 < age < 1

        # Old (30 days ago)
        old = now - timedelta(days=30)
        age = service._compute_age_days(old)
        assert 29 < age < 31

        # None
        age = service._compute_age_days(None)
        assert age == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
