"""Tests for GraphRelationshipService (Qdrant-based implementation).

GraphRelationshipService previously used an in-DB embedding field and a NumPy
similarity matrix. It now uses Qdrant "recommend" by point id (vector_id).
"""

from __future__ import annotations

from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
import redis

from app.services.graph_relationship_service import GraphRelationshipService


@pytest.fixture
def mock_db():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_redis():
    client = MagicMock(spec=redis.Redis)
    client.hgetall.return_value = {}
    return client


@pytest.fixture
def service(mock_db, mock_redis):
    return GraphRelationshipService(mock_db, mock_redis)


@pytest.mark.asyncio
async def test_extract_relationships_via_qdrant_threshold_and_dedup(service):
    org_id = str(uuid4())

    memories = [
        {"id": "a", "vector_id": "v1"},
        {"id": "b", "vector_id": "v2"},
        {"id": "c", "vector_id": "v3"},
    ]

    async def _fake_recommend(*, org_id: str, positive_point_id: str, limit: int, score_threshold: float, with_payload: bool):
        assert org_id
        if positive_point_id == "v1":
            return [
                {"id": "v2", "score": 0.92, "payload": {}},
                {"id": "v3", "score": 0.80, "payload": {}},
            ]
        if positive_point_id == "v2":
            return [
                {"id": "v1", "score": 0.92, "payload": {}},
                {"id": "v3", "score": 0.88, "payload": {}},
            ]
        return []

    with patch(
        "app.services.graph_relationship_service.QdrantService.recommend_by_point_id",
        new=AsyncMock(side_effect=_fake_recommend),
    ):
        rels = await service._extract_relationships_via_qdrant(
            org_id=org_id,
            memories=memories,
            threshold=0.85,
            max_per_memory=5,
        )

    # v1->v2 (a<b) and v2->v3 (b<c); v1->v3 is below threshold
    pairs = {(r["from_id"], r["to_id"]) for r in rels}
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert ("a", "c") not in pairs


@pytest.mark.asyncio
async def test_extract_relationships_ignores_unknown_vectors(service):
    org_id = str(uuid4())

    memories = [
        {"id": "a", "vector_id": "v1"},
        {"id": "b", "vector_id": "v2"},
    ]

    with patch(
        "app.services.graph_relationship_service.QdrantService.recommend_by_point_id",
        new=AsyncMock(return_value=[{"id": "v999", "score": 0.99, "payload": {}}]),
    ):
        rels = await service._extract_relationships_via_qdrant(
            org_id=org_id,
            memories=memories,
            threshold=0.5,
            max_per_memory=5,
        )

    assert rels == []


@pytest.mark.asyncio
async def test_create_falkordb_relationships_includes_org_id(service, mock_redis):
    org_id = str(uuid4())
    relationships = [
        {
            "from_id": "mem1",
            "to_id": "mem2",
            "org_id": org_id,
            "similarity_score": 0.85,
            "relationship_type": "RELATES_TO",
        }
    ]

    mock_redis.execute_command.return_value = True

    created = await service._create_falkordb_relationships(relationships)

    assert created == 1
    args = mock_redis.execute_command.call_args[0]
    assert args[0] == "GRAPH.QUERY"
    query = args[2]
    assert "MERGE (a:Memory" in query
    assert org_id in query


@pytest.mark.asyncio
async def test_store_relationship_metadata_commits(service, mock_db):
    org_id = str(uuid4())
    relationships = [
        {"from_id": "a", "to_id": "b", "org_id": org_id, "similarity_score": 0.9, "relationship_type": "RELATES_TO"},
        {"from_id": "b", "to_id": "c", "org_id": org_id, "similarity_score": 0.8, "relationship_type": "RELATES_TO"},
    ]

    mock_db.execute = AsyncMock(side_effect=[MagicMock(), MagicMock(rowcount=2)])
    mock_db.commit = AsyncMock()

    stored = await service._store_relationship_metadata(org_id, relationships)

    assert stored == 2
    assert mock_db.execute.await_count == 2
    mock_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_populate_relationships_happy_path(service, mock_db, mock_redis):
    org_id = str(uuid4())

    with patch.object(service, "_get_memories_with_vectors", new=AsyncMock(return_value=[{"id": "a", "vector_id": "v1"}])):
        with patch.object(service, "_extract_relationships_via_qdrant", new=AsyncMock(return_value=[])):
            result = await service.populate_relationships(org_id=org_id)

    assert result["memories_processed"] == 1
    assert result["relationships_found"] == 0
