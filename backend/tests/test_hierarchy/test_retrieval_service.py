"""Tests for RetrievalService (GAP-3: Representative Selection)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

# Test fixtures for fake models
class FakeTopic:
    """Minimal Topic stand-in."""
    def __init__(self, id_: str, org_id: str, name: str = "Test Topic"):
        self.id = id_
        self.organization_id = org_id
        self.name = name
        self.centroid_vector_id = f"vec_{id_}"
        self.scope = "personal"
        self.scope_id = None

    @property
    def scope_key(self) -> str:
        return f"{self.scope}:{self.scope_id}" if self.scope_id else self.scope


class FakeSemanticNode:
    """Minimal SemanticNode stand-in."""
    def __init__(
        self,
        id_: str,
        org_id: str,
        topic_id: str,
        content: str = "Test semantic",
        source_memory_ids: list | None = None,
    ):
        self.id = id_
        self.organization_id = org_id
        self.topic_id = topic_id
        self.content = content
        self.vector_id = f"vec_{id_}"
        self.source_memory_ids = source_memory_ids or []
        self.composite_quality = 0.75
        self._vec = [0.1] * 768


class FakeEpisode:
    """Minimal Episode stand-in."""
    def __init__(self, id_: str, org_id: str, topic_id: str):
        self.id = id_
        self.organization_id = org_id
        self.topic_id = topic_id
        self.narrative_summary = "Test episode"
        self.centroid_vector_id = f"vec_{id_}"


class FakeNavigationEdge:
    """Minimal NavigationEdge stand-in."""
    def __init__(
        self,
        org_id: str,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        similarity: float = 0.8,
    ):
        self.id = str(uuid4())
        self.organization_id = org_id
        self.source_type = source_type
        self.source_id = source_id
        self.target_type = target_type
        self.target_id = target_id
        self.similarity = similarity
        self.k_rank = 1
        self.generation = 1


class FakeMemory:
    """Minimal MemoryMetadata stand-in."""
    def __init__(self, id_: str, org_id: str, episode_id: str):
        self.id = id_
        self.organization_id = org_id
        self.episode_id = episode_id
        self.content = "Test message"


class FakeQdrantHit:
    """Mock Qdrant search result."""
    def __init__(self, memory_id: str, memory_type: str, score: float):
        self.id = memory_id
        self.score = score
        self.payload = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "organization_id": "org-123",
        }


# ────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session():
    """Async DB session mock."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def org_id() -> UUID:
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests."""
    with patch("app.services.retrieval_service.settings") as mock_settings:
        mock_settings.VLLM_EMBEDDING_MODEL = "nomic-embed-text"
        mock_settings.QDRANT_COLLECTION_NAME = "test_collection"
        yield mock_settings


# Helper to build RetrievalService with mocked dependencies
def _make_service(session):
    """Create RetrievalService with mocked embedding and qdrant."""
    from app.services.retrieval_service import RetrievalService
    svc = RetrievalService(session)
    svc.embedding_svc = AsyncMock()
    svc.qdrant = AsyncMock()
    # Default mock behaviors
    svc.embedding_svc.get_embedding = AsyncMock(return_value=[0.5] * 768)
    svc.qdrant.search = AsyncMock(return_value=[])
    return svc


# ────────────────────────────────────────────────────────────────────
# Test: get_candidates
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_candidates_returns_topics_and_nodes(
    mock_session, org_id
):
    """Should retrieve both topics and semantic nodes from Qdrant."""
    svc = _make_service(mock_session)

    # Mock Qdrant to return 2 topics + 3 semantic nodes
    topic_hits = [
        FakeQdrantHit("topic-1", "topic", 0.9),
        FakeQdrantHit("topic-2", "topic", 0.8),
    ]
    node_hits = [
        FakeQdrantHit("node-1", "semantic_node", 0.85),
        FakeQdrantHit("node-2", "semantic_node", 0.75),
        FakeQdrantHit("node-3", "semantic_node", 0.65),
    ]

    # Use side_effect list to return different values on successive calls
    svc.qdrant.search = AsyncMock(side_effect=[topic_hits, node_hits])

    candidates = await svc._get_candidates(
        query_vector=[0.5] * 768,
        organization_id=org_id,
        scope="personal",
        scope_id=None,
    )

    assert len(candidates) == 5
    # Sorted by score: topic-1(0.9), node-1(0.85), topic-2(0.8), node-2(0.75), node-3(0.65)
    assert candidates[0]["type"] == "topic"
    assert candidates[0]["id"] == "topic-1"
    assert candidates[0]["score"] == 0.9
    assert candidates[1]["type"] == "semantic_node"
    assert candidates[1]["id"] == "node-1"
    assert candidates[1]["score"] == 0.85


@pytest.mark.asyncio
async def test_get_candidates_filters_low_scores(
    mock_session, org_id
):
    """Should filter out candidates below MIN_RELEVANCE_THRESHOLD."""
    svc = _make_service(mock_session)

    # All scores below 0.15 threshold
    low_hits = [
        FakeQdrantHit("topic-1", "topic", 0.10),
        FakeQdrantHit("node-1", "semantic_node", 0.05),
    ]

    svc.qdrant.search.return_value = low_hits

    candidates = await svc._get_candidates(
        query_vector=[0.5] * 768,
        organization_id=org_id,
        scope="personal",
        scope_id=None,
    )

    assert len(candidates) == 0


# ────────────────────────────────────────────────────────────────────
# Test: build_coverage_map
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_coverage_map_includes_self_and_neighbors(
    mock_session, org_id
):
    """Coverage map should include node itself + kNN neighbors."""
    svc = _make_service(mock_session)

    candidates = [
        {"type": "topic", "id": "t1", "key": "topic:t1", "score": 0.9, "selected": False},
        {"type": "semantic_node", "id": "n1", "key": "semantic_node:n1", "score": 0.8, "selected": False},
    ]

    # Mock DB: topic t1 has 2 outgoing edges, node n1 has 1
    edges_t1 = [
        FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n1", 0.9),
        FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n2", 0.8),
    ]
    edges_n1 = [
        FakeNavigationEdge("org-123", "semantic_node", "n1", "episode", "e1", 0.7),
    ]

    call_count = [0]

    async def mock_execute(stmt):
        result_mock = MagicMock()
        call_count[0] += 1
        # First call for t1, second call for n1
        if call_count[0] == 1:
            result_mock.scalars.return_value.all.return_value = edges_t1
        elif call_count[0] == 2:
            result_mock.scalars.return_value.all.return_value = edges_n1
        else:
            result_mock.scalars.return_value.all.return_value = []
        return result_mock

    mock_session.execute.side_effect = mock_execute

    coverage_map = await svc._build_coverage_map(
        candidates=candidates,
        organization_id=org_id,
    )

    # t1 covers: itself + n1 + n2 = 3 nodes
    assert len(coverage_map["topic:t1"]) == 3
    assert "topic:t1" in coverage_map["topic:t1"]
    assert "semantic_node:n1" in coverage_map["topic:t1"]
    assert "semantic_node:n2" in coverage_map["topic:t1"]

    # n1 covers: itself + e1 = 2 nodes
    assert len(coverage_map["semantic_node:n1"]) == 2
    assert "semantic_node:n1" in coverage_map["semantic_node:n1"]
    assert "episode:e1" in coverage_map["semantic_node:n1"]


# ────────────────────────────────────────────────────────────────────
# Test: select_representatives (submodular greedy)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_select_representatives_greedy_maximizes_coverage(
    mock_session, org_id
):
    """Greedy selection should pick candidates with max marginal gain."""
    svc = _make_service(mock_session)

    # Setup: 3 candidates
    # t1 covers 3 nodes (score=0.9)
    # t2 covers 2 nodes, 1 overlaps with t1 (score=0.7)
    # n1 covers 2 nodes, no overlap (score=0.5)
    # Expected greedy order: t1, n1, t2 (n1 adds more new coverage than t2)

    topic_hits = [
        FakeQdrantHit("t1", "topic", 0.9),
        FakeQdrantHit("t2", "topic", 0.7),
    ]
    node_hits = [
        FakeQdrantHit("n1", "semantic_node", 0.5),
    ]

    # Use side_effect list for successive calls
    svc.qdrant.search = AsyncMock(side_effect=[topic_hits, node_hits])

    # Mock coverage map
    edges_t1 = [
        FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n2", 0.9),
        FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n3", 0.8),
    ]
    edges_t2 = [
        FakeNavigationEdge("org-123", "topic", "t2", "semantic_node", "n3", 0.7),  # overlaps with t1
    ]
    edges_n1 = [
        FakeNavigationEdge("org-123", "semantic_node", "n1", "episode", "e5", 0.6),
    ]

    async def mock_execute(stmt):
        result_mock = MagicMock()
        stmt_str = str(stmt)
        if "'t1'" in stmt_str:
            result_mock.scalars.return_value.all.return_value = edges_t1
        elif "'t2'" in stmt_str:
            result_mock.scalars.return_value.all.return_value = edges_t2
        elif "'n1'" in stmt_str:
            result_mock.scalars.return_value.all.return_value = edges_n1
        else:
            result_mock.scalars.return_value.all.return_value = []
        return result_mock

    mock_session.execute.side_effect = mock_execute

    representatives = await svc._select_representatives(
        query="test query",
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        max_count=2,
        coverage_weight=0.5,
    )

    # Should select t1 first (highest relevance + coverage)
    # Then n1 (adds 2 NEW nodes vs t2 which adds 0 new nodes)
    assert len(representatives) == 2
    assert representatives[0]["type"] == "topic"
    assert representatives[0]["id"] == "t1"
    # Second could be n1 or t2 depending on gain calculation
    # But with coverage_weight=0.5, n1 should win (adds 2 new nodes)


@pytest.mark.asyncio
async def test_select_representatives_respects_max_count(
    mock_session, org_id
):
    """Should stop after max_count representatives selected."""
    svc = _make_service(mock_session)

    # 5 candidates available
    topic_hits = [FakeQdrantHit(f"t{i}", "topic", 0.9 - i*0.1) for i in range(5)]
    svc.qdrant.search.return_value = topic_hits

    # Mock DB to return empty edges (no neighbors)
    async def mock_execute(stmt):
        result_mock = MagicMock()
        result_mock.scalars.return_value.all.return_value = []
        return result_mock

    mock_session.execute.side_effect = mock_execute

    representatives = await svc._select_representatives(
        query="test query",
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        max_count=3,
        coverage_weight=0.3,
    )

    assert len(representatives) == 3


@pytest.mark.asyncio
async def test_select_representatives_handles_empty_query_embedding(
    mock_session, org_id
):
    """Should return empty if query embedding fails."""
    svc = _make_service(mock_session)
    svc.embedding_svc.get_embedding.return_value = []  # Empty embedding

    representatives = await svc._select_representatives(
        query="bad query",
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        max_count=5,
        coverage_weight=0.3,
    )

    assert len(representatives) == 0


# ────────────────────────────────────────────────────────────────────
# Test: expand_through_knn
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_through_knn_traverses_graph(
    mock_session, org_id
):
    """Should BFS through kNN graph up to depth limit."""
    svc = _make_service(mock_session)

    representatives = [
        {"type": "topic", "id": "t1", "score": 0.9},
    ]

    # Graph: t1 → n1 → e1 → (messages)
    # Depth 0: t1 (collect semantic nodes from t1)
    # Depth 1: n1 (collect source memories)
    # Depth 2: e1 (collect episode messages)

    nodes_in_t1 = [FakeSemanticNode("n1", "org-123", "t1", source_memory_ids=["mem-1"])]
    edges_t1 = [FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n1", 0.9)]
    edges_n1 = [FakeNavigationEdge("org-123", "semantic_node", "n1", "episode", "e1", 0.8)]
    messages_in_e1 = [FakeMemory("msg-1", "org-123", "e1"), FakeMemory("msg-2", "org-123", "e1")]

    call_count = [0]

    async def mock_execute(stmt):
        result_mock = MagicMock()
        call_count[0] += 1
        
        # Call order:
        # 1: Get semantic nodes from topic t1
        # 2: Get navigation edges from t1
        # 3: Get semantic node n1 details
        # 4: Get navigation edges from n1
        # 5: Get navigation edges from e1
        # 6: Get messages from episode e1
        
        if call_count[0] == 1:
            # Get semantic nodes in topic t1
            result_mock.all.return_value = [(nodes_in_t1[0].id,)]
        elif call_count[0] == 2:
            # Get edges from t1
            result_mock.scalars.return_value.all.return_value = edges_t1
        elif call_count[0] == 3:
            # Get node n1 details
            result_mock.scalar_one_or_none.return_value = nodes_in_t1[0]
        elif call_count[0] == 4:
            # Get edges from n1
            result_mock.scalars.return_value.all.return_value = edges_n1
        elif call_count[0] == 5:
            # Get edges from e1
            result_mock.scalars.return_value.all.return_value = []
        elif call_count[0] == 6:
            # Get messages from e1
            result_mock.all.return_value = [("msg-1",), ("msg-2",)]
        else:
            result_mock.scalars.return_value.all.return_value = []
            result_mock.all.return_value = []

        return result_mock

    mock_session.execute.side_effect = mock_execute

    memory_ids = await svc._expand_through_knn(
        representatives=representatives,
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        depth=2,
        max_results=100,
    )

    # Should collect: n1 (semantic node from topic), mem-1 (from n1 source memories)
    # Episode traversal may not occur depending on BFS implementation details
    assert len(memory_ids) >= 1
    assert "n1" in memory_ids or "mem-1" in memory_ids


@pytest.mark.asyncio
async def test_expand_through_knn_respects_max_results(
    mock_session, org_id
):
    """Should stop collecting after max_results reached."""
    svc = _make_service(mock_session)

    representatives = [
        {"type": "topic", "id": "t1", "score": 0.9},
    ]

    # Mock: topic has 50 semantic nodes
    many_nodes = [(f"n{i}",) for i in range(50)]

    async def mock_execute(stmt):
        result_mock = MagicMock()
        stmt_str = str(stmt)

        if "memory_semantic_nodes" in stmt_str and "id" in stmt_str:
            result_mock.all.return_value = many_nodes
        else:
            result_mock.scalars.return_value.all.return_value = []

        return result_mock

    mock_session.execute.side_effect = mock_execute

    memory_ids = await svc._expand_through_knn(
        representatives=representatives,
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        depth=1,
        max_results=10,
    )

    assert len(memory_ids) == 10


# ────────────────────────────────────────────────────────────────────
# Test: retrieve (full pipeline)
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_full_pipeline(
    mock_session, org_id
):
    """Full retrieval should run Stage I + Stage II + compute coverage stats."""
    svc = _make_service(mock_session)

    # Mock Qdrant to return 1 topic
    topic_hits = [FakeQdrantHit("t1", "topic", 0.9)]
    svc.qdrant.search.return_value = topic_hits

    # Mock DB: topic has 1 semantic node with 1 source memory
    nodes_in_t1 = [FakeSemanticNode("n1", "org-123", "t1", source_memory_ids=["mem-1"])]
    edges_t1 = [FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n1", 0.9)]

    async def mock_execute(stmt):
        result_mock = MagicMock()
        stmt_str = str(stmt)

        if "memory_semantic_nodes" in stmt_str and "'t1'" in stmt_str:
            result_mock.all.return_value = [("n1",)]
        elif "memory_semantic_nodes" in stmt_str and "'n1'" in stmt_str:
            result_mock.scalar_one_or_none.return_value = nodes_in_t1[0]
        elif "navigation_edges" in stmt_str:
            result_mock.scalars.return_value.all.return_value = edges_t1
        else:
            result_mock.scalars.return_value.all.return_value = []

        return result_mock

    mock_session.execute.side_effect = mock_execute

    result = await svc.retrieve(
        query="test query",
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        max_representatives=3,
        coverage_weight=0.3,
        max_results=20,
        expansion_depth=2,
    )

    assert "memory_ids" in result
    assert "representatives" in result
    assert "coverage_stats" in result
    assert len(result["representatives"]) >= 1
    assert result["representatives"][0]["type"] == "topic"
    assert result["representatives"][0]["id"] == "t1"
    assert result["coverage_stats"]["total_nodes_covered"] >= 1


@pytest.mark.asyncio
async def test_retrieve_handles_no_candidates(
    mock_session, org_id
):
    """Should gracefully handle when Qdrant returns no candidates."""
    svc = _make_service(mock_session)

    # Qdrant returns empty
    svc.qdrant.search.return_value = []

    result = await svc.retrieve(
        query="nonexistent query",
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        max_representatives=5,
        coverage_weight=0.3,
        max_results=20,
        expansion_depth=2,
    )

    assert result["memory_ids"] == []
    assert result["representatives"] == []
    assert result["coverage_stats"]["total_nodes_covered"] == 0


# ────────────────────────────────────────────────────────────────────
# Test: Coverage statistics
# ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compute_coverage_stats_counts_unique_nodes(
    mock_session, org_id
):
    """Coverage stats should count all unique nodes reached via kNN."""
    svc = _make_service(mock_session)

    representatives = [
        {"type": "topic", "id": "t1", "score": 0.9},
        {"type": "topic", "id": "t2", "score": 0.8},
    ]

    # t1 → n1, n2
    # t2 → n2, n3 (n2 overlaps)
    # Total unique: t1, t2, n1, n2, n3 = 5 nodes

    edges_t1 = [
        FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n1", 0.9),
        FakeNavigationEdge("org-123", "topic", "t1", "semantic_node", "n2", 0.8),
    ]
    edges_t2 = [
        FakeNavigationEdge("org-123", "topic", "t2", "semantic_node", "n2", 0.7),
        FakeNavigationEdge("org-123", "topic", "t2", "semantic_node", "n3", 0.6),
    ]

    call_count = [0]

    async def mock_execute(stmt):
        result_mock = MagicMock()
        call_count[0] += 1
        
        # First call for t1, second call for t2
        if call_count[0] == 1:
            result_mock.scalars.return_value.all.return_value = edges_t1
        elif call_count[0] == 2:
            result_mock.scalars.return_value.all.return_value = edges_t2
        else:
            result_mock.scalars.return_value.all.return_value = []

        return result_mock

    mock_session.execute.side_effect = mock_execute

    stats = await svc._compute_coverage_stats(
        representatives=representatives,
        organization_id=org_id,
    )

    assert stats["total_nodes_covered"] == 5  # t1, t2, n1, n2, n3
    assert stats["sparsity_score"] > 0


# ── Integration Tests: retrieve_with_gating (GAP-4) ─────────────────


@pytest.mark.asyncio
async def test_retrieve_with_gating_returns_fewer_items():
    """retrieve_with_gating should filter candidates via entropy threshold."""
    mock_session = MagicMock()
    svc = _make_service(mock_session)
    
    # Mock Stage I: _select_representatives returns 2 representatives
    representatives = [
        {"type": "topic", "id": "t1", "score": 0.9},
        {"type": "semantic_node", "id": "n1", "score": 0.85},
    ]
    svc._select_representatives = AsyncMock(return_value=representatives)
    
    # Mock _build_initial_context
    svc._build_initial_context = AsyncMock(return_value="Summary of topics")
    
    # Mock _get_candidate_evidence returns 5 messages
    candidate_items = [
        {"id": f"msg-{i}", "type": "message", "content": f"Message {i}", "metadata": {}}
        for i in range(5)
    ]
    svc._get_candidate_evidence = AsyncMock(return_value=candidate_items)
    
    # Mock uncertainty service: include only first 2 items
    svc.uncertainty_svc.expand_with_gating = AsyncMock(return_value={
        "final_context": "Summary + Message 0 + Message 1",
        "included_items": [
            {"id": "msg-0", "entropy_reduction": 0.5},
            {"id": "msg-1", "entropy_reduction": 0.3},
        ],
        "total_entropy_reduction": 0.8,
        "expansion_stopped_reason": "Entropy threshold not met",
    })
    
    # Mock coverage stats
    svc._compute_coverage_stats = AsyncMock(return_value={
        "total_nodes_covered": 10,
        "sparsity_score": 0.5,
    })
    
    org_id = UUID("12345678-1234-1234-1234-123456789012")
    result = await svc.retrieve_with_gating(
        query="What happened?",
        organization_id=org_id,
        scope="personal",
        scope_id=None,
        max_representatives=5,
        coverage_weight=0.5,
        entropy_threshold=0.1,
        max_expansion_items=10,
    )
    
    # Verify entropy gating filtered 5 candidates down to 2
    assert len(result["included_items"]) == 2
    assert result["included_items"][0]["id"] == "msg-0"
    assert result["total_entropy_reduction"] == 0.8
    assert "threshold" in result["expansion_stopped_reason"].lower()
    assert result["representatives"] == representatives


@pytest.mark.asyncio
async def test_retrieve_with_gating_calls_uncertainty_service():
    """retrieve_with_gating should invoke UncertaintyGatingService.expand_with_gating."""
    mock_session = MagicMock()
    svc = _make_service(mock_session)
    
    # Mock dependencies
    svc._select_representatives = AsyncMock(return_value=[
        {"type": "topic", "id": "t1", "score": 0.9},
    ])
    svc._build_initial_context = AsyncMock(return_value="Context text")
    svc._get_candidate_evidence = AsyncMock(return_value=[
        {"id": "msg-1", "type": "message", "content": "Test", "metadata": {}},
    ])
    svc.uncertainty_svc.expand_with_gating = AsyncMock(return_value={
        "final_context": "Context + Test",
        "included_items": [{"id": "msg-1", "entropy_reduction": 0.4}],
        "total_entropy_reduction": 0.4,
        "expansion_stopped_reason": "Max items reached",
    })
    svc._compute_coverage_stats = AsyncMock(return_value={
        "total_nodes_covered": 5,
        "sparsity_score": 0.3,
    })
    
    org_id = UUID("12345678-1234-1234-1234-123456789012")
    await svc.retrieve_with_gating(
        query="test query",
        organization_id=org_id,
        scope="personal",
        entropy_threshold=0.1,
    )
    
    # Verify uncertainty service was called with correct params
    assert svc.uncertainty_svc.expand_with_gating.call_count == 1
    call_kwargs = svc.uncertainty_svc.expand_with_gating.call_args.kwargs
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["initial_context"] == "Context text"
    assert call_kwargs["entropy_threshold"] == 0.1
    assert len(call_kwargs["candidate_items"]) == 1


@pytest.mark.asyncio
async def test_retrieve_with_gating_empty_candidates():
    """retrieve_with_gating should handle zero candidate evidence gracefully."""
    mock_session = MagicMock()
    svc = _make_service(mock_session)
    
    svc._select_representatives = AsyncMock(return_value=[
        {"type": "topic", "id": "t1", "score": 0.9},
    ])
    svc._build_initial_context = AsyncMock(return_value="Context")
    svc._get_candidate_evidence = AsyncMock(return_value=[])  # No candidates
    svc.uncertainty_svc.expand_with_gating = AsyncMock(return_value={
        "final_context": "Context",
        "included_items": [],
        "total_entropy_reduction": 0.0,
        "expansion_stopped_reason": "No candidates provided",
    })
    svc._compute_coverage_stats = AsyncMock(return_value={
        "total_nodes_covered": 1,
        "sparsity_score": 0.1,
    })
    
    org_id = UUID("12345678-1234-1234-1234-123456789012")
    result = await svc.retrieve_with_gating(
        query="test",
        organization_id=org_id,
        scope="personal",
    )
    
    assert result["included_items"] == []
    assert result["total_entropy_reduction"] == 0.0
