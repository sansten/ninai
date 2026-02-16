"""Tests for HierarchyService — the facade orchestrator (GAP-1 + GAP-6).

Covers:
    - ingest_message: append (no distillation), new_episode (triggers distillation + kNN)
    - rebuild: calls distill_batch + rebuild_all, returns combined stats
    - hierarchical_search: top-down 4-level search
    - _find_closed_undistilled: filters correctly
    - _search_topics: qdrant hit parsing + fallback
    - _search_semantic_nodes: qdrant hit parsing + fallback
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest

from app.services.hierarchy_service import HierarchyService
from .conftest import (
    ORG_ID, USER_ID, FAKE_EMBEDDING,
    FakeEpisode, FakeSemanticNode, FakeTopic, FakeMemory,
    ScalarOneResult, ScalarsListResult,
)


# Helper to build a HierarchyService with mocked sub-services
def _make_service(session) -> HierarchyService:
    svc = HierarchyService(session)
    svc.episode_svc = AsyncMock()
    svc.distillation_svc = AsyncMock()
    svc.knn_svc = AsyncMock()
    svc.topic_structure_svc = AsyncMock()
    return svc


# ════════════════════════════════════════════════════════════════════════
# ingest_message
# ════════════════════════════════════════════════════════════════════════

class TestIngestMessage:

    async def test_append_no_distillation(self, mock_session):
        """When episode_svc returns 'append', no distillation occurs."""
        svc = _make_service(mock_session)
        mem_id = str(uuid4())
        ep_id = str(uuid4())

        svc.episode_svc.add_message_to_current_episode = AsyncMock(
            return_value={"episode_id": ep_id, "action": "append"}
        )

        result = await svc.ingest_message(
            organization_id=ORG_ID,
            owner_id=USER_ID,
            memory_id=mem_id,
        )

        assert result["episode_id"] == ep_id
        assert result["episode_action"] == "append"
        assert "episode:append" in result["actions"]
        svc.distillation_svc.distill_episode.assert_not_called()
        svc.knn_svc.update_for_node.assert_not_called()

    async def test_new_episode_triggers_distillation_and_knn(self, mock_session):
        """When 'new_episode' action, distill closed episodes and update kNN."""
        svc = _make_service(mock_session)
        mem_id = str(uuid4())
        ep_id = str(uuid4())
        closed_ep_id = str(uuid4())
        sn_id = str(uuid4())

        # Episode service creates a new episode
        svc.episode_svc.add_message_to_current_episode = AsyncMock(
            return_value={"episode_id": ep_id, "action": "new_episode"}
        )

        # One closed episode exists
        closed_ep = FakeEpisode(status="closed", vector_id="ep_vec1")
        closed_ep.id = closed_ep_id

        with patch.object(svc, "_find_closed_undistilled", new_callable=AsyncMock, return_value=[closed_ep]):
            # Distillation returns one semantic node
            svc.distillation_svc.distill_episode = AsyncMock(
                return_value=[{"semantic_node_id": sn_id}]
            )
            svc.knn_svc.update_for_node = AsyncMock(
                return_value={"edges_created": 3, "edges_removed": 0}
            )

            result = await svc.ingest_message(
                organization_id=ORG_ID,
                owner_id=USER_ID,
                memory_id=mem_id,
            )

        assert result["episode_action"] == "new_episode"
        assert any("distilled" in a for a in result["actions"])
        assert any("knn_updated" in a for a in result["actions"])

        # kNN should have been called for the semantic node AND the episode
        assert svc.knn_svc.update_for_node.call_count >= 2

    async def test_new_episode_rebalances_topics_when_enabled(self, mock_session):
        svc = _make_service(mock_session)
        svc.episode_svc.add_message_to_current_episode = AsyncMock(
            return_value={"episode_id": str(uuid4()), "action": "new_episode"}
        )
        svc.topic_structure_svc.rebalance_topics = AsyncMock(return_value={
            "sparsity_score": 0.75,
            "semantic_score": 0.62,
        })

        result = await svc.ingest_message(
            organization_id=ORG_ID,
            owner_id=USER_ID,
            memory_id=str(uuid4()),
            auto_distill=False,
            auto_rebalance_topics=True,
        )

        svc.topic_structure_svc.rebalance_topics.assert_awaited_once()
        assert any("topics_rebalanced" in a for a in result["actions"])

    async def test_new_episode_no_auto_distill(self, mock_session):
        """With auto_distill=False, skip distillation even on new_episode."""
        svc = _make_service(mock_session)
        svc.episode_svc.add_message_to_current_episode = AsyncMock(
            return_value={"episode_id": str(uuid4()), "action": "new_episode"}
        )

        result = await svc.ingest_message(
            organization_id=ORG_ID,
            owner_id=USER_ID,
            memory_id=str(uuid4()),
            auto_distill=False,
        )

        svc.distillation_svc.distill_episode.assert_not_called()
        assert result["episode_action"] == "new_episode"

    async def test_distillation_failure_is_logged_not_raised(self, mock_session):
        """Distillation errors should be caught, not propagated."""
        svc = _make_service(mock_session)
        svc.episode_svc.add_message_to_current_episode = AsyncMock(
            return_value={"episode_id": str(uuid4()), "action": "new_episode"}
        )

        closed_ep = FakeEpisode(status="closed", vector_id=None)

        with patch.object(svc, "_find_closed_undistilled", new_callable=AsyncMock, return_value=[closed_ep]):
            svc.distillation_svc.distill_episode = AsyncMock(
                side_effect=Exception("LLM unavailable")
            )

            # Should NOT raise
            result = await svc.ingest_message(
                organization_id=ORG_ID,
                owner_id=USER_ID,
                memory_id=str(uuid4()),
            )
            assert "episode:new_episode" in result["actions"]


# ════════════════════════════════════════════════════════════════════════
# rebuild
# ════════════════════════════════════════════════════════════════════════

class TestRebuild:

    async def test_rebuild_calls_distill_batch_and_knn_rebuild(self, mock_session):
        svc = _make_service(mock_session)

        svc.distillation_svc.distill_batch = AsyncMock(return_value={
            "episodes_processed": 3, "nodes_created": 7
        })
        svc.topic_structure_svc.rebalance_topics = AsyncMock(return_value={
            "sparsity_score": 0.8, "semantic_score": 0.7
        })
        svc.knn_svc.rebuild_all = AsyncMock(return_value={
            "nodes_processed": 10, "edges_created": 25, "generation": 2
        })

        stats = await svc.rebuild(organization_id=ORG_ID)

        assert stats["distillation"]["episodes_processed"] == 3
        assert stats["topic_structure"]["sparsity_score"] == 0.8
        assert stats["knn_rebuild"]["nodes_processed"] == 10
        assert stats["knn_rebuild"]["generation"] == 2

        svc.distillation_svc.distill_batch.assert_awaited_once()
        svc.topic_structure_svc.rebalance_topics.assert_awaited_once()
        svc.knn_svc.rebuild_all.assert_awaited_once()

    async def test_rebuild_passes_limits(self, mock_session):
        svc = _make_service(mock_session)
        svc.distillation_svc.distill_batch = AsyncMock(return_value={})
        svc.topic_structure_svc.rebalance_topics = AsyncMock(return_value={})
        svc.knn_svc.rebuild_all = AsyncMock(return_value={})

        await svc.rebuild(
            organization_id=ORG_ID,
            segment_limit=100,
            distill_limit=20,
        )

        svc.distillation_svc.distill_batch.assert_awaited_once_with(
            organization_id=ORG_ID,
            limit=20,
        )


# ════════════════════════════════════════════════════════════════════════
# hierarchical_search
# ════════════════════════════════════════════════════════════════════════

class TestHierarchicalSearch:

    async def test_returns_all_four_levels(self, mock_session, mock_embed):
        svc = _make_service(mock_session)

        topic_id = str(uuid4())
        sn_id = str(uuid4())
        ep_id = str(uuid4())

        # Mock _search_topics
        with patch.object(svc, "_search_topics", new_callable=AsyncMock, return_value=[
            {"id": topic_id, "score": 0.9, "label": "ML"},
        ]):
            # kNN traverse: returns a semantic node
            svc.knn_svc.traverse = AsyncMock(return_value=[
                {"type": "semantic_node", "id": sn_id, "similarity": 0.85},
            ])

            # Mock _search_semantic_nodes
            with patch.object(svc, "_search_semantic_nodes", new_callable=AsyncMock, return_value=[]):
                # DB calls for expanding to episodes and messages
                sn_obj = FakeSemanticNode(source_episode_ids=[ep_id])
                sn_obj.id = sn_id

                ep_obj = FakeEpisode()
                ep_obj.id = ep_id
                ep_obj.title = "ML Discussion"
                ep_obj.narrative_summary = "Discussed ML topics"
                ep_obj.message_count = 5
                ep_obj.boundary_start = None
                ep_obj.boundary_end = None

                call_count = 0
                async def mock_execute(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        # Semantic node lookup
                        return ScalarOneResult(sn_obj)
                    elif call_count == 2:
                        # Episode lookup
                        return ScalarsListResult([ep_obj])
                    elif call_count == 3:
                        # Membership lookup (memory IDs)
                        return ScalarsListResult([str(uuid4())])
                    elif call_count == 4:
                        # Message lookup
                        msg = FakeMemory()
                        return ScalarsListResult([msg])
                    return ScalarsListResult([])

                mock_session.execute = AsyncMock(side_effect=mock_execute)

                result = await svc.hierarchical_search(
                    organization_id=ORG_ID,
                    query="machine learning",
                    limit=10,
                )

        assert "topics" in result
        assert "semantic_nodes" in result
        assert "episodes" in result
        assert "messages" in result
        assert len(result["topics"]) == 1
        assert result["topics"][0]["label"] == "ML"

    async def test_empty_query_returns_empty_structure(self, mock_session, mock_embed):
        svc = _make_service(mock_session)

        with patch.object(svc, "_search_topics", new_callable=AsyncMock, return_value=[]):
            with patch.object(svc, "_search_semantic_nodes", new_callable=AsyncMock, return_value=[]):
                result = await svc.hierarchical_search(
                    organization_id=ORG_ID,
                    query="nonexistent query",
                    limit=10,
                )

        assert result["topics"] == []
        assert result["semantic_nodes"] == []
        assert result["episodes"] == []
        assert result["messages"] == []


# ════════════════════════════════════════════════════════════════════════
# _find_closed_undistilled
# ════════════════════════════════════════════════════════════════════════

class TestFindClosedUndistilled:

    async def test_returns_episodes_without_semantic_nodes(self, mock_session):
        closed_ep = FakeEpisode(status="closed")

        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # closed episodes query
                return ScalarsListResult([closed_ep])
            else:
                # check for existing semantic nodes → none
                return ScalarOneResult(None)

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        svc = HierarchyService(mock_session)
        result = await svc._find_closed_undistilled(
            organization_id=ORG_ID,
            owner_id=USER_ID,
        )
        assert len(result) == 1
        assert result[0] is closed_ep

    async def test_excludes_already_distilled_episodes(self, mock_session):
        closed_ep = FakeEpisode(status="closed")
        sn = FakeSemanticNode()

        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ScalarsListResult([closed_ep])
            else:
                # existing semantic node found → already distilled
                return ScalarOneResult(sn)

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        svc = HierarchyService(mock_session)
        result = await svc._find_closed_undistilled(
            organization_id=ORG_ID,
            owner_id=USER_ID,
        )
        assert result == []


# ════════════════════════════════════════════════════════════════════════
# _search_topics fallback
# ════════════════════════════════════════════════════════════════════════

class TestSearchTopics:

    async def test_fallback_to_db_when_qdrant_fails(self, mock_session):
        topic = FakeTopic(label="AI Ethics")

        mock_session.execute = AsyncMock(return_value=ScalarsListResult([topic]))
        svc = HierarchyService(mock_session)

        # Force Qdrant to fail → triggers DB fallback
        with patch("app.core.qdrant.QdrantService.build_org_filter", side_effect=Exception("no qdrant")):
            result = await svc._search_topics(
                organization_id=ORG_ID,
                query_embedding=FAKE_EMBEDDING,
                limit=5,
            )

        assert len(result) == 1
        assert result[0]["label"] == "AI Ethics"
        assert result[0]["score"] == 0.5  # fallback score

    async def test_qdrant_hit_parsing(self, mock_session):
        svc = HierarchyService(mock_session)
        topic_id = str(uuid4())

        hit = MagicMock()
        hit.payload = {"topic_id": topic_id, "label": "Quantum", "type": "topic"}
        hit.score = 0.92

        with patch("app.core.qdrant.QdrantService.build_org_filter", return_value=MagicMock()):
            with patch("app.core.qdrant.QdrantService.get_client") as mock_client:
                mc = MagicMock()
                mc.search = MagicMock(return_value=[hit])
                mock_client.return_value = mc

                with patch("app.core.config.settings") as mock_settings:
                    mock_settings.QDRANT_COLLECTION_NAME = "test_collection"

                    result = await svc._search_topics(
                        organization_id=ORG_ID,
                        query_embedding=FAKE_EMBEDDING,
                        limit=5,
                    )

        assert len(result) == 1
        assert result[0]["id"] == topic_id
        assert result[0]["score"] == 0.92
        assert result[0]["label"] == "Quantum"


# ════════════════════════════════════════════════════════════════════════
# _search_semantic_nodes fallback
# ════════════════════════════════════════════════════════════════════════

class TestSearchSemanticNodes:

    async def test_returns_empty_on_qdrant_failure(self, mock_session):
        svc = HierarchyService(mock_session)

        with patch("app.core.qdrant.QdrantService.build_org_filter", side_effect=Exception("broken")):
            result = await svc._search_semantic_nodes(
                organization_id=ORG_ID,
                query_embedding=FAKE_EMBEDDING,
                limit=10,
            )
        assert result == []

    async def test_qdrant_hit_parsing(self, mock_session):
        svc = HierarchyService(mock_session)
        sn_id = str(uuid4())

        hit = MagicMock()
        hit.payload = {"semantic_node_id": sn_id, "type": "semantic_node"}
        hit.score = 0.78

        with patch("app.core.qdrant.QdrantService.build_org_filter", return_value=MagicMock()):
            with patch("app.core.qdrant.QdrantService.get_client") as mock_client:
                mc = MagicMock()
                mc.search = MagicMock(return_value=[hit])
                mock_client.return_value = mc

                with patch("app.core.config.settings") as mock_settings:
                    mock_settings.QDRANT_COLLECTION_NAME = "test_collection"

                    result = await svc._search_semantic_nodes(
                        organization_id=ORG_ID,
                        query_embedding=FAKE_EMBEDDING,
                        limit=10,
                    )

        assert len(result) == 1
        assert result[0]["id"] == sn_id
        assert result[0]["type"] == "semantic_node"
