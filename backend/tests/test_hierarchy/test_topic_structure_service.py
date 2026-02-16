"""Tests for TopicStructureService (GAP-2)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

from app.services.topic_structure_service import TopicStructureService
from .conftest import ORG_ID, FakeSemanticNode, FakeTopic


def _node_with_vec(vec, topic_id):
    node = FakeSemanticNode(topic_id=topic_id)
    node._vec = vec
    return node


class TestGuidanceScores:

    async def test_scores_for_balanced_topics(self, mock_session):
        svc = TopicStructureService(mock_session)
        t1 = FakeTopic(label="alpha")
        t2 = FakeTopic(label="beta")

        n1 = _node_with_vec([1.0, 0.0], t1.id)
        n2 = _node_with_vec([0.9, 0.1], t1.id)
        n3 = _node_with_vec([0.0, 1.0], t2.id)
        n4 = _node_with_vec([0.1, 0.9], t2.id)

        with patch.object(svc, "_collect_semantic_nodes", new_callable=AsyncMock, return_value=[n1, n2, n3, n4]):
            with patch.object(svc, "_fetch_topics", new_callable=AsyncMock, return_value=[t1, t2]):
                with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock) as mock_emb:
                    mock_emb.side_effect = [n1._vec, n2._vec, n3._vec, n4._vec]
                    scores = await svc.compute_guidance_scores(organization_id=ORG_ID)

        assert scores["sparsity_score"] > 0.0
        assert scores["semantic_score"] > 0.0
        assert len(scores["topic_stats"]) == 2


class TestRebalanceTopics:

    async def test_splits_overcrowded_topic(self, mock_session):
        svc = TopicStructureService(mock_session)
        topic = FakeTopic(label="big")

        nodes = []
        for _ in range(7):
            nodes.append(_node_with_vec([1.0, 0.0], topic.id))
        for _ in range(6):
            nodes.append(_node_with_vec([0.0, 1.0], topic.id))

        with patch.object(svc, "_collect_semantic_nodes", new_callable=AsyncMock, return_value=nodes):
            with patch.object(svc, "_fetch_topics", new_callable=AsyncMock, return_value=[topic]):
                async def _emb(node):
                    return node._vec
                with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock) as mock_emb:
                    mock_emb.side_effect = _emb
                    result = await svc.rebalance_topics(
                        organization_id=ORG_ID,
                        max_size=12,
                        min_size=1,
                    )

        assert result["splits"] == 1
        assert result["nodes_moved"] > 0

    async def test_merges_tiny_topic(self, mock_session):
        svc = TopicStructureService(mock_session)
        big = FakeTopic(label="big")
        tiny = FakeTopic(label="tiny")

        big_nodes = [
            _node_with_vec([1.0, 0.0], big.id),
            _node_with_vec([0.9, 0.1], big.id),
            _node_with_vec([0.8, 0.2], big.id),
        ]
        tiny_nodes = [_node_with_vec([1.0, 0.0], tiny.id)]

        all_nodes = big_nodes + tiny_nodes

        with patch.object(svc, "_collect_semantic_nodes", new_callable=AsyncMock, return_value=all_nodes):
            with patch.object(svc, "_fetch_topics", new_callable=AsyncMock, return_value=[big, tiny]):
                async def _emb(node):
                    return node._vec
                with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock) as mock_emb:
                    mock_emb.side_effect = _emb
                    result = await svc.rebalance_topics(
                        organization_id=ORG_ID,
                        max_size=12,
                        min_size=2,
                    )

        assert result["merges"] == 1
        assert result["nodes_moved"] >= 1


# ── GAP-5: Retroactive Restructuring Tests ─────────────────────────


class Testtrack_reassignment_ratio:

    async def test_zero_ratio_with_no_history(self, mock_session):
        """No history entries → 0% reassignment ratio."""
        svc = TopicStructureService(mock_session)

        call_count = [0]

        async def mock_execute(stmt):
            call_count[0] += 1

            if call_count[0] == 1:
                # Total nodes count
                result_mock = MagicMock()
                result_mock.scalar.return_value = 10
                return result_mock
            elif call_count[0] == 2:
                # Nodes with >1 history entry (reassigned) - return IDs
                result_mock = MagicMock()
                # Create MagicMock that mimics scalars().all() chain
                scalars_obj = MagicMock()
                scalars_obj.all.return_value = []
                result_mock.scalars.return_value = scalars_obj
                return result_mock
            else:
                # Nodes with any history
                result_mock = MagicMock()
                result_mock.scalar.return_value = 0
                return result_mock

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        ratio = await svc.track_reassignment_ratio(organization_id=ORG_ID)

        assert ratio["total_nodes"] == 10
        assert ratio["nodes_reassigned"] == 0
        assert ratio["reassignment_ratio"] == 0.0

    async def test_high_ratio_with_many_reassignments(self, mock_session):
        """Multiple history entries per node → high reassignment ratio."""
        svc = TopicStructureService(mock_session)

        call_count = [0]

        async def mock_execute(stmt):
            call_count[0] += 1

            if call_count[0] == 1:
                # Total nodes
                result_mock = MagicMock()
                result_mock.scalar.return_value = 10
                return result_mock
            elif call_count[0] == 2:
                # Nodes with >1 history entry (reassigned) - return list of node IDs
                result_mock = MagicMock()
                # Create MagicMock that mimics scalars().all() chain
                node_ids = ["node-1", "node-2", "node-3", "node-4"]  # 4 nodes reassigned
                scalars_obj = MagicMock()
                scalars_obj.all.return_value = node_ids
                result_mock.scalars.return_value = scalars_obj
                return result_mock
            else:
                # Nodes with any history
                result_mock = MagicMock()
                result_mock.scalar.return_value = 10
                return result_mock

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        ratio = await svc.track_reassignment_ratio(organization_id=ORG_ID)

        assert ratio["total_nodes"] == 10
        assert ratio["nodes_reassigned"] == 4
        assert ratio["reassignment_ratio"] == 0.4  # 40%


class TestGuidedAttach:

    async def test_attaches_to_best_topic(self, mock_session):
        """Guided attach should choose topic that maximizes guidance score."""
        svc = TopicStructureService(mock_session)

        topic1 = FakeTopic(label="alpha")
        topic2 = FakeTopic(label="beta")

        node = FakeSemanticNode()
        node._vec = [1.0, 0.0]

        # Mock nodes for topic1 (similar to target node)
        topic1_node1 = _node_with_vec([0.9, 0.1], topic1.id)
        topic1_node2 = _node_with_vec([0.95, 0.05], topic1.id)

        # Mock nodes for topic2 (dissimilar to target node)
        topic2_node1 = _node_with_vec([0.0, 1.0], topic2.id)
        topic2_node2 = _node_with_vec([0.1, 0.9], topic2.id)

        call_count = [0]

        async def mock_execute(stmt):
            call_count[0] += 1

            result_mock = MagicMock()
            # First call: get nodes for topic1
            if call_count[0] == 1:
                scalars_obj = MagicMock()
                scalars_obj.all.return_value = [topic1_node1, topic1_node2]
                result_mock.scalars.return_value = scalars_obj
            # Second call: get nodes for topic2
            elif call_count[0] == 2:
                scalars_obj = MagicMock()
                scalars_obj.all.return_value = [topic2_node1, topic2_node2]
                result_mock.scalars.return_value = scalars_obj

            return result_mock

        mock_session.execute = AsyncMock(side_effect=mock_execute)

        with patch.object(svc, "_get_node_embedding", new_callable=AsyncMock) as mock_emb:
            # Return node embedding then topic node embeddings
            async def _side_effect(n):
                return n._vec

            mock_emb.side_effect = _side_effect

            with patch.object(svc, "_record_topic_assignment", new_callable=AsyncMock):
                selected_id = await svc.guided_attach(
                    node=node,
                    candidate_topics=[topic1, topic2],
                )

        # Should select topic1 (more similar)
        assert selected_id == topic1.id

    async def test_creates_topic_when_no_candidates(self, mock_session):
        """Guided attach should create new topic if no candidates."""
        svc = TopicStructureService(mock_session)

        node = FakeSemanticNode()

        with patch.object(svc, "_create_topic_for_node", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "new-topic-id"
            selected_id = await svc.guided_attach(
                node=node,
                candidate_topics=[],
            )

        assert selected_id == "new-topic-id"
        mock_create.assert_called_once()


class TestPeriodicRestructure:

    async def test_reassigns_nodes_and_rebalances(self, mock_session):
        """Periodic restructure should reassign nodes + split/merge."""
        svc = TopicStructureService(mock_session)

        topic1 = FakeTopic(label="alpha")
        topic2 = FakeTopic(label="beta")

        node1 = _node_with_vec([1.0, 0.0], topic1.id)
        node2 = _node_with_vec([0.0, 1.0], topic2.id)

        with patch.object(svc, "compute_guidance_scores", new_callable=AsyncMock) as mock_scores:
            mock_scores.side_effect = [
                {"sparsity_score": 1.0, "semantic_score": 0.5},  # Before
                {"sparsity_score": 1.2, "semantic_score": 0.6},  # After
            ]

            with patch.object(svc, "_collect_semantic_nodes", new_callable=AsyncMock, return_value=[node1, node2]):
                with patch.object(svc, "_fetch_topics", new_callable=AsyncMock, return_value=[topic1, topic2]):
                    with patch.object(svc, "guided_attach", new_callable=AsyncMock) as mock_attach:
                        # Reassign node1 to topic2
                        mock_attach.side_effect = [topic2.id, topic2.id]

                        with patch.object(svc, "_record_topic_assignment", new_callable=AsyncMock):
                            with patch.object(svc, "rebalance_topics", new_callable=AsyncMock) as mock_rebal:
                                mock_rebal.return_value = {"splits": 0, "merges": 1, "nodes_moved": 2}

                                with patch.object(svc, "track_reassignment_ratio", new_callable=AsyncMock) as mock_ratio:
                                    mock_ratio.return_value = {
                                        "total_nodes": 2,
                                        "nodes_reassigned": 1,
                                        "reassignment_ratio": 0.5,
                                    }

                                    result = await svc.periodic_restructure(organization_id=ORG_ID)

        assert result["reassignments"] == 1  # node1 reassigned
        assert result["merges"] == 1
        assert result["guidance_score_after"] > result["guidance_score_before"]
        assert result["reassignment_ratio"] == 0.5


class TestRecordTopicAssignment:

    async def test_creates_history_entry(self, mock_session):
        """_record_topic_assignment should create history record."""
        svc = TopicStructureService(mock_session)

        node = FakeSemanticNode()
        node.id = "node-123"
        node.organization_id = ORG_ID

        await svc._record_topic_assignment(
            node=node,
            new_topic_id="topic-456",
            previous_topic_id="topic-789",
            reason="split",
            score_before=1.0,
            score_after=1.2,
        )

        # Verify session.add was called
        mock_session.add.assert_called_once()
        history_entry = mock_session.add.call_args[0][0]

        assert history_entry.semantic_node_id == "node-123"
        assert history_entry.topic_id == "topic-456"
        assert history_entry.previous_topic_id == "topic-789"
        assert history_entry.reason == "split"
        assert history_entry.guidance_score_before == 1.0
        assert history_entry.guidance_score_after == 1.2
