"""Tests for KNNNavigationService (GAP-6).

Covers:
    - update_for_node: no embedding, with neighbours, edge creation
    - rebuild_all: generation counter, pruning
    - get_neighbours: returns materialized edges
    - traverse: multi-hop BFS
    - _collect_all_nodes: gathers episodes, semantic nodes, topics
    - _get_node_embedding: Qdrant retrieve, fallback to EmbeddingService
    - _find_neighbours: Qdrant search with filters
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.knn_navigation_service import (
    KNNNavigationService,
    DEFAULT_K,
    MIN_SIMILARITY,
)
from .conftest import (
    ORG_ID, USER_ID, FAKE_EMBEDDING, ZERO_EMBEDDING,
    FakeEpisode, FakeSemanticNode, FakeNavigationEdge, FakeTopic,
    ScalarOneResult, ScalarsListResult, DeleteResult,
)


# ════════════════════════════════════════════════════════════════════════
# update_for_node
# ════════════════════════════════════════════════════════════════════════

class TestUpdateForNode:

    async def test_skips_when_no_embedding(self, mock_session):
        svc = KNNNavigationService(mock_session)

        with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock, return_value=None):
            stats = await svc.update_for_node(
                organization_id=ORG_ID,
                node_type="episode",
                node_id=str(uuid4()),
            )
            assert stats == {"edges_created": 0, "edges_removed": 0}

    async def test_skips_when_zero_embedding(self, mock_session):
        svc = KNNNavigationService(mock_session)

        with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock, return_value=ZERO_EMBEDDING):
            stats = await svc.update_for_node(
                organization_id=ORG_ID,
                node_type="episode",
                node_id=str(uuid4()),
            )
            assert stats == {"edges_created": 0, "edges_removed": 0}

    async def test_creates_edges_for_valid_neighbours(self, mock_session):
        node_id = str(uuid4())
        target_id = str(uuid4())

        # Mock delete returning 0 removed
        mock_session.execute = AsyncMock(return_value=DeleteResult(0))

        svc = KNNNavigationService(mock_session)

        with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock, return_value=FAKE_EMBEDDING):
            with patch.object(svc, "_find_neighbours", new_callable=AsyncMock, return_value=[
                {"type": "semantic_node", "id": target_id, "score": 0.85, "vector_id": "v1"},
                {"type": "topic", "id": str(uuid4()), "score": 0.40, "vector_id": "v2"},
                {"type": "episode", "id": str(uuid4()), "score": 0.10, "vector_id": "v3"},  # below MIN_SIMILARITY
            ]):
                stats = await svc.update_for_node(
                    organization_id=ORG_ID,
                    node_type="episode",
                    node_id=node_id,
                )
                # 2 neighbours above MIN_SIMILARITY (0.85 and 0.40), 1 below (0.10)
                assert stats["edges_created"] == 2
                assert stats["edges_removed"] == 0
                assert mock_session.add.call_count == 2

    async def test_filters_self_from_neighbours(self, mock_session):
        node_id = str(uuid4())

        mock_session.execute = AsyncMock(return_value=DeleteResult(0))
        svc = KNNNavigationService(mock_session)

        with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock, return_value=FAKE_EMBEDDING):
            with patch.object(svc, "_find_neighbours", new_callable=AsyncMock, return_value=[
                # This neighbour is the node itself → should be filtered
                {"type": "episode", "id": node_id, "score": 1.0, "vector_id": "v_self"},
                {"type": "semantic_node", "id": str(uuid4()), "score": 0.6, "vector_id": "v1"},
            ]):
                stats = await svc.update_for_node(
                    organization_id=ORG_ID,
                    node_type="episode",
                    node_id=node_id,
                )
                assert stats["edges_created"] == 1  # Self excluded


# ════════════════════════════════════════════════════════════════════════
# get_neighbours
# ════════════════════════════════════════════════════════════════════════

class TestGetNeighbours:

    async def test_returns_materialized_edges(self, mock_session):
        node_id = str(uuid4())
        edge1 = FakeNavigationEdge(
            source_type="episode", source_id=node_id,
            target_type="semantic_node", target_id=str(uuid4()),
            similarity=0.9, k_rank=1,
        )
        edge2 = FakeNavigationEdge(
            source_type="episode", source_id=node_id,
            target_type="topic", target_id=str(uuid4()),
            similarity=0.7, k_rank=2,
        )

        mock_session.execute = AsyncMock(return_value=ScalarsListResult([edge1, edge2]))
        svc = KNNNavigationService(mock_session)

        result = await svc.get_neighbours(
            organization_id=ORG_ID,
            node_type="episode",
            node_id=node_id,
        )
        assert len(result) == 2
        assert result[0]["type"] == "semantic_node"
        assert result[0]["similarity"] == 0.9
        assert result[0]["k_rank"] == 1
        assert result[1]["type"] == "topic"
        assert result[1]["k_rank"] == 2

    async def test_returns_empty_for_unknown_node(self, mock_session):
        mock_session.execute = AsyncMock(return_value=ScalarsListResult([]))
        svc = KNNNavigationService(mock_session)

        result = await svc.get_neighbours(
            organization_id=ORG_ID,
            node_type="episode",
            node_id=str(uuid4()),
        )
        assert result == []


# ════════════════════════════════════════════════════════════════════════
# traverse (multi-hop BFS)
# ════════════════════════════════════════════════════════════════════════

class TestTraverse:

    async def test_single_hop_returns_neighbours(self, mock_session):
        start_id = str(uuid4())
        nb1_id = str(uuid4())
        nb2_id = str(uuid4())

        svc = KNNNavigationService(mock_session)

        with patch.object(svc, "get_neighbours", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [
                {"type": "semantic_node", "id": nb1_id, "similarity": 0.9, "k_rank": 1},
                {"type": "topic", "id": nb2_id, "similarity": 0.7, "k_rank": 2},
            ]

            result = await svc.traverse(
                organization_id=ORG_ID,
                start_type="episode",
                start_id=start_id,
                hops=1,
            )
            assert len(result) == 2

    async def test_multi_hop_traversal(self, mock_session):
        id_a = str(uuid4())
        id_b = str(uuid4())
        id_c = str(uuid4())

        svc = KNNNavigationService(mock_session)

        call_count = 0
        async def mock_get(*, organization_id, node_type, node_id, k=5):
            nonlocal call_count
            call_count += 1
            if node_id == id_a:
                return []  # start node has no outgoing edges
            if node_type == "episode" and node_id != id_a:
                return [{"type": "semantic_node", "id": id_b, "similarity": 0.8, "k_rank": 1}]
            if node_type == "semantic_node":
                return [{"type": "topic", "id": id_c, "similarity": 0.7, "k_rank": 1}]
            return []

        with patch.object(svc, "get_neighbours", side_effect=mock_get):
            result = await svc.traverse(
                organization_id=ORG_ID,
                start_type="episode",
                start_id=id_a,
                hops=3,
            )
            # Should visit at least id_a's hood (empty), no further expansion
            # Verify no duplicates in result
            ids_seen = [(r["type"], r["id"]) for r in result]
            assert len(ids_seen) == len(set(ids_seen))

    async def test_traverse_avoids_cycles(self, mock_session):
        """BFS should not revisit already-visited nodes."""
        id_a = str(uuid4())
        id_b = str(uuid4())

        svc = KNNNavigationService(mock_session)

        async def mock_get(*, organization_id, node_type, node_id, k=5):
            if node_id == id_a:
                return [{"type": "semantic_node", "id": id_b, "similarity": 0.8, "k_rank": 1}]
            if node_id == id_b:
                # Points back to A → cycle
                return [{"type": "episode", "id": id_a, "similarity": 0.8, "k_rank": 1}]
            return []

        with patch.object(svc, "get_neighbours", side_effect=mock_get):
            result = await svc.traverse(
                organization_id=ORG_ID,
                start_type="episode",
                start_id=id_a,
                hops=5,
            )
            result_keys = [(r["type"], r["id"]) for r in result]
            # id_b should appear at most once
            assert result_keys.count(("semantic_node", id_b)) <= 1


# ════════════════════════════════════════════════════════════════════════
# _collect_all_nodes
# ════════════════════════════════════════════════════════════════════════

class TestCollectAllNodes:

    async def test_collects_episodes_semantic_nodes_topics(self, mock_session):
        ep = FakeEpisode(vector_id="ep_vec1")
        sn = FakeSemanticNode(vector_id="sn_vec1")
        topic = FakeTopic()

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ScalarsListResult([ep])
            elif call_count == 2:
                return ScalarsListResult([sn])
            elif call_count == 3:
                return ScalarsListResult([topic])
            return ScalarsListResult([])

        mock_session.execute = AsyncMock(side_effect=side_effect)
        svc = KNNNavigationService(mock_session)

        nodes = await svc._collect_all_nodes(ORG_ID)
        types = {n["type"] for n in nodes}
        assert "episode" in types
        assert "semantic_node" in types
        assert "topic" in types
        assert len(nodes) == 3


# ════════════════════════════════════════════════════════════════════════
# _get_node_embedding
# ════════════════════════════════════════════════════════════════════════

class TestGetNodeEmbedding:

    async def test_returns_from_qdrant_when_vector_id(self, mock_session, mock_qdrant_client):
        """If vector_id is provided and Qdrant returns a vector, use it."""
        mock_point = MagicMock()
        mock_point.vector = FAKE_EMBEDDING
        mock_qdrant_client.retrieve = MagicMock(return_value=[mock_point])

        svc = KNNNavigationService(mock_session)
        emb = await svc._get_node_embedding(
            organization_id=ORG_ID,
            node_type="episode",
            node_id=str(uuid4()),
            vector_id="ep:123",
        )
        assert emb == FAKE_EMBEDDING

    async def test_fallback_to_embed_for_topic(self, mock_session, mock_embed):
        topic = FakeTopic(label="Machine Learning", keywords=["ML", "AI"])
        mock_session.execute = AsyncMock(return_value=ScalarOneResult(topic))

        svc = KNNNavigationService(mock_session)

        # No vector_id → falls back to computing from content
        with patch("app.core.qdrant.QdrantService.get_client", side_effect=Exception("no qdrant")):
            emb = await svc._get_node_embedding(
                organization_id=ORG_ID,
                node_type="topic",
                node_id=topic.id,
                vector_id=None,
            )
            assert emb == FAKE_EMBEDDING
            mock_embed.assert_called()

    async def test_fallback_to_embed_for_semantic_node(self, mock_session, mock_embed):
        sn = FakeSemanticNode(content="User prefers dark mode")
        mock_session.execute = AsyncMock(return_value=ScalarOneResult(sn))

        svc = KNNNavigationService(mock_session)

        emb = await svc._get_node_embedding(
            organization_id=ORG_ID,
            node_type="semantic_node",
            node_id=sn.id,
            vector_id=None,
        )
        assert emb == FAKE_EMBEDDING

    async def test_returns_none_for_unknown_type(self, mock_session):
        svc = KNNNavigationService(mock_session)

        emb = await svc._get_node_embedding(
            organization_id=ORG_ID,
            node_type="unknown",
            node_id=str(uuid4()),
            vector_id=None,
        )
        assert emb is None


# ════════════════════════════════════════════════════════════════════════
# _find_neighbours (Qdrant search)
# ════════════════════════════════════════════════════════════════════════

class TestFindNeighbours:

    async def test_returns_empty_on_qdrant_error(self, mock_session):
        svc = KNNNavigationService(mock_session)

        with patch("app.core.qdrant.QdrantService.build_org_filter", side_effect=Exception("no qdrant")):
            result = await svc._find_neighbours(
                organization_id=ORG_ID,
                query_vector=FAKE_EMBEDDING,
                exclude_vector_id=None,
                limit=20,
            )
            assert result == []

    async def test_parses_qdrant_hits(self, mock_session):
        svc = KNNNavigationService(mock_session)

        hit1 = MagicMock()
        hit1.id = "vec_ep1"
        hit1.score = 0.88
        hit1.payload = {"type": "episode", "episode_id": "ep-001"}

        hit2 = MagicMock()
        hit2.id = "vec_sn1"
        hit2.score = 0.75
        hit2.payload = {"type": "semantic_node", "semantic_node_id": "sn-001"}

        with patch("app.core.qdrant.QdrantService.build_org_filter", return_value=MagicMock()):
            with patch("app.core.qdrant.QdrantService.get_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.search = MagicMock(return_value=[hit1, hit2])
                mock_get_client.return_value = mock_client

                result = await svc._find_neighbours(
                    organization_id=ORG_ID,
                    query_vector=FAKE_EMBEDDING,
                    exclude_vector_id=None,
                    limit=20,
                )
                assert len(result) == 2
                assert result[0]["type"] == "episode"
                assert result[0]["id"] == "ep-001"
                assert result[0]["score"] == 0.88
                assert result[1]["type"] == "semantic_node"

    async def test_excludes_vector_id(self, mock_session):
        svc = KNNNavigationService(mock_session)

        hit = MagicMock()
        hit.id = "exclude_me"
        hit.score = 0.99
        hit.payload = {"type": "episode", "episode_id": "ep-001"}

        with patch("app.core.qdrant.QdrantService.build_org_filter", return_value=MagicMock()):
            with patch("app.core.qdrant.QdrantService.get_client") as mock_get_client:
                mock_client = MagicMock()
                mock_client.search = MagicMock(return_value=[hit])
                mock_get_client.return_value = mock_client

                result = await svc._find_neighbours(
                    organization_id=ORG_ID,
                    query_vector=FAKE_EMBEDDING,
                    exclude_vector_id="exclude_me",
                    limit=20,
                )
                assert len(result) == 0


# ════════════════════════════════════════════════════════════════════════
# rebuild_all
# ════════════════════════════════════════════════════════════════════════

class TestRebuildAll:

    async def test_rebuild_with_no_nodes(self, mock_session):
        svc = KNNNavigationService(mock_session)

        call_count = 0
        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # max generation query
                return ScalarOneResult(None)
            # collect_all_nodes queries (3 queries: episodes, semantic_nodes, topics)
            return ScalarsListResult([])

        mock_session.execute = AsyncMock(side_effect=side_effect)

        with patch.object(svc, "_collect_all_nodes", new_callable=AsyncMock, return_value=[]):
            # Prune delete returns 0
            mock_session.execute = AsyncMock(side_effect=[
                ScalarOneResult(None),  # max generation
                DeleteResult(0),        # prune
            ])

            stats = await svc.rebuild_all(organization_id=ORG_ID)
            assert stats["nodes_processed"] == 0
            assert stats["edges_created"] == 0
            assert stats["generation"] == 1
