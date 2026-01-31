"""
Tests for GraphRelationshipService.

Test coverage:
- Similarity calculation accuracy
- Relationship extraction and filtering
- FalkorDB integration
- PostgreSQL persistence
- Configuration management
- Edge cases and error handling
"""

import pytest

pytest.skip(
    "Legacy tests covered embedding-matrix implementation; GraphRelationshipService now uses Qdrant recommend-by-point-id.",
    allow_module_level=True,
)


@pytest.fixture
def mock_db():
    """Mock async database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return MagicMock(spec=redis.Redis)


@pytest.fixture
def service(mock_db, mock_redis):
    """Create service instance with mocked dependencies."""
    return GraphRelationshipService(mock_db, mock_redis)


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    np.random.seed(42)
    return [
        np.random.randn(384).astype(np.float32),  # Memory 1
        np.random.randn(384).astype(np.float32),  # Memory 2
        np.random.randn(384).astype(np.float32),  # Memory 3
        np.random.randn(384).astype(np.float32),  # Memory 4
    ]


@pytest.fixture
def sample_memories(sample_embeddings):
    """Sample memory objects with embeddings."""
    org_id = str(uuid4())
    return [
        {
            "id": str(uuid4()),
            "title": f"Memory {i}",
            "embedding": emb,
            "created_at": datetime.utcnow(),
            "org_id": org_id
        }
        for i, emb in enumerate(sample_embeddings)
    ]


class TestSimilarityCalculation:
    """Test embedding similarity matrix calculation."""

    def test_similarity_matrix_shape(self, service, sample_embeddings):
        """Test that similarity matrix has correct shape."""
        matrix = service._calculate_similarity_matrix(sample_embeddings)
        
        assert matrix.shape == (len(sample_embeddings), len(sample_embeddings))

    def test_similarity_is_symmetric(self, service, sample_embeddings):
        """Test that similarity matrix is symmetric."""
        matrix = service._calculate_similarity_matrix(sample_embeddings)
        
        # A[i,j] should equal A[j,i]
        assert np.allclose(matrix, matrix.T)

    def test_diagonal_is_one(self, service, sample_embeddings):
        """Test that diagonal (self-similarity) is 1.0."""
        matrix = service._calculate_similarity_matrix(sample_embeddings)
        
        # Self-similarity should be 1.0 for normalized embeddings
        assert np.allclose(np.diag(matrix), 1.0, rtol=1e-5)

    def test_similarity_range(self, service, sample_embeddings):
        """Test that similarities are in valid range [-1, 1]."""
        matrix = service._calculate_similarity_matrix(sample_embeddings)
        
        # Cosine similarity ranges from -1 to 1
        assert np.all(matrix >= -1.0)
        assert np.all(matrix <= 1.0)

    def test_identical_embeddings(self, service):
        """Test similarity with identical embeddings."""
        embedding = np.ones(10, dtype=np.float32)
        embeddings = [embedding, embedding.copy()]
        
        matrix = service._calculate_similarity_matrix(embeddings)
        
        # Should be identity matrix (or close to it)
        assert np.allclose(matrix[0, 1], 1.0)
        assert np.allclose(matrix[1, 0], 1.0)

    def test_orthogonal_embeddings(self, service):
        """Test similarity with orthogonal embeddings."""
        embedding1 = np.array([1, 0, 0, 0], dtype=np.float32)
        embedding2 = np.array([0, 1, 0, 0], dtype=np.float32)
        
        matrix = service._calculate_similarity_matrix([embedding1, embedding2])
        
        # Orthogonal vectors have zero similarity
        assert np.allclose(matrix[0, 1], 0.0, atol=1e-5)


class TestRelationshipExtraction:
    """Test relationship extraction from similarity matrix."""

    def test_extract_relationships_above_threshold(self, service, sample_memories):
        """Test that only relationships above threshold are extracted."""
        embeddings = [m["embedding"] for m in sample_memories]
        matrix = service._calculate_similarity_matrix(embeddings)
        
        relationships = service._extract_relationships(
            sample_memories,
            matrix,
            threshold=0.5,
            max_per_memory=10
        )
        
        # All relationships should be above threshold (except self)
        for rel in relationships:
            assert rel["similarity_score"] >= 0.5

    def test_self_relationships_excluded(self, service, sample_memories):
        """Test that self-relationships are excluded."""
        embeddings = [m["embedding"] for m in sample_memories]
        matrix = service._calculate_similarity_matrix(embeddings)
        
        relationships = service._extract_relationships(
            sample_memories,
            matrix,
            threshold=0.0,
            max_per_memory=10
        )
        
        # No memory should relate to itself
        for rel in relationships:
            assert rel["from_id"] != rel["to_id"]

    def test_max_relationships_per_memory(self, service, sample_memories):
        """Test that max_per_memory limit is enforced."""
        embeddings = [m["embedding"] for m in sample_memories]
        matrix = service._calculate_similarity_matrix(embeddings)
        max_rel = 2
        
        relationships = service._extract_relationships(
            sample_memories,
            matrix,
            threshold=0.0,
            max_per_memory=max_rel
        )
        
        # Count relationships per memory
        from_counts = {}
        for rel in relationships:
            from_id = rel["from_id"]
            from_counts[from_id] = from_counts.get(from_id, 0) + 1
        
        # Each memory should have at most max_per_memory relationships
        for count in from_counts.values():
            assert count <= max_rel

    def test_bidirectional_avoidance(self, service, sample_memories):
        """Test that only one direction is created for each pair."""
        embeddings = [m["embedding"] for m in sample_memories]
        matrix = service._calculate_similarity_matrix(embeddings)
        
        relationships = service._extract_relationships(
            sample_memories,
            matrix,
            threshold=0.0,
            max_per_memory=10
        )
        
        # Check no bidirectional relationships
        seen_pairs = set()
        for rel in relationships:
            pair = tuple(sorted([rel["from_id"], rel["to_id"]]))
            assert pair not in seen_pairs, "Bidirectional relationship found"
            seen_pairs.add(pair)

    def test_threshold_filters_weak_relationships(self, service, sample_memories):
        """Test that threshold properly filters relationships."""
        embeddings = [m["embedding"] for m in sample_memories]
        matrix = service._calculate_similarity_matrix(embeddings)
        
        # Extract with high threshold
        high_threshold_rels = service._extract_relationships(
            sample_memories,
            matrix,
            threshold=0.9,
            max_per_memory=10
        )
        
        # Extract with low threshold
        low_threshold_rels = service._extract_relationships(
            sample_memories,
            matrix,
            threshold=0.1,
            max_per_memory=10
        )
        
        # Should have more relationships with low threshold
        assert len(low_threshold_rels) >= len(high_threshold_rels)


class TestFalkorDBIntegration:
    """Test FalkorDB relationship creation."""

    @pytest.mark.asyncio
    async def test_create_falkordb_relationships_success(self, service, mock_redis):
        """Test successful creation of relationships in FalkorDB."""
        relationships = [
            {
                "from_id": "mem1",
                "to_id": "mem2",
                "similarity_score": 0.85,
                "relationship_type": "RELATES_TO"
            }
        ]
        
        # Mock Redis GRAPH.QUERY
        mock_redis.execute_command.return_value = True
        
        created = await service._create_falkordb_relationships(relationships)
        
        assert created == 1
        mock_redis.execute_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_relationships_list(self, service):
        """Test handling of empty relationships list."""
        created = await service._create_falkordb_relationships([])
        
        assert created == 0

    @pytest.mark.asyncio
    async def test_falkordb_query_format(self, service, mock_redis):
        """Test that FalkorDB queries are properly formatted."""
        relationships = [
            {
                "from_id": "abc123",
                "to_id": "def456",
                "similarity_score": 0.92,
                "relationship_type": "RELATES_TO"
            }
        ]
        
        mock_redis.execute_command.return_value = True
        
        await service._create_falkordb_relationships(relationships)
        
        # Check that query was made
        call_args = mock_redis.execute_command.call_args
        assert call_args[0][0] == "GRAPH.QUERY"
        assert call_args[0][1] == service.graph_name


class TestPostgresIntegration:
    """Test PostgreSQL persistence."""

    @pytest.mark.asyncio
    async def test_store_relationship_metadata(self, service, mock_db):
        """Test storing relationship metadata in PostgreSQL."""
        org_id = str(uuid4())
        relationships = [
            {
                "from_id": str(uuid4()),
                "to_id": str(uuid4()),
                "similarity_score": 0.80,
                "relationship_type": "RELATES_TO"
            }
        ]
        
        # Mock database operations
        mock_db.execute.return_value = MagicMock()
        mock_db.execute.return_value.rowcount = 1
        
        stored = await service._store_relationship_metadata(org_id, relationships)
        
        # Should have attempted to execute database operations
        assert mock_db.execute.called or mock_db.commit.called


class TestConfigurationManagement:
    """Test configuration storage and retrieval."""

    @pytest.mark.asyncio
    async def test_update_config(self, service, mock_redis):
        """Test updating relationship config."""
        org_id = str(uuid4())
        
        mock_redis.hset.return_value = True
        
        config = await service.update_config(
            org_id,
            similarity_threshold=0.8,
            max_relationships=10
        )
        
        assert mock_redis.hset.called

    @pytest.mark.asyncio
    async def test_get_config_defaults(self, service, mock_redis):
        """Test getting config with default values."""
        org_id = str(uuid4())
        mock_redis.hgetall.return_value = {}
        
        config = await service.get_config(org_id)
        
        assert config["similarity_threshold"] == 0.75  # Default
        assert config["max_relationships"] == 5  # Default


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_populate_with_no_memories(self, service, mock_db):
        """Test graceful handling when no memories exist."""
        org_id = str(uuid4())
        
        # Mock empty result
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        
        # Should handle gracefully, not crash
        result = await service.populate_relationships(org_id)
        
        assert result.get("relationships_found", 0) == 0

    @pytest.mark.asyncio
    async def test_populate_with_embedding_errors(self, service, mock_db):
        """Test handling of embedding calculation errors."""
        org_id = str(uuid4())
        
        # Mock exception during execution
        mock_db.execute.side_effect = Exception("Database error")
        
        result = await service.populate_relationships(org_id)
        
        assert "error" in result or result.get("created") == 0

    def test_similarity_with_nan_values(self, service):
        """Test handling of NaN values in embeddings."""
        embeddings = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([np.nan, 2.0, 3.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ]
        
        # Should handle NaN gracefully (typically with filtering or replacement)
        # This test verifies the function doesn't crash
        matrix = service._calculate_similarity_matrix(embeddings)
        assert matrix is not None


class TestPerformance:
    """Test performance with large datasets."""

    def test_similarity_matrix_with_large_dataset(self, service):
        """Test similarity calculation performance with 1000+ embeddings."""
        np.random.seed(42)
        embeddings = [
            np.random.randn(384).astype(np.float32)
            for _ in range(100)  # 100 embeddings
        ]
        
        # Should complete in reasonable time
        import time
        start = time.time()
        matrix = service._calculate_similarity_matrix(embeddings)
        elapsed = time.time() - start
        
        assert matrix.shape == (100, 100)
        assert elapsed < 5.0  # Should be fast with vectorized operations

    def test_relationship_extraction_performance(self, service):
        """Test relationship extraction with large dataset."""
        np.random.seed(42)
        embeddings = [
            np.random.randn(384).astype(np.float32)
            for _ in range(100)
        ]
        
        memories = [
            {
                "id": f"mem_{i}",
                "title": f"Memory {i}",
                "embedding": emb,
                "created_at": datetime.utcnow(),
                "org_id": "org_123"
            }
            for i, emb in enumerate(embeddings)
        ]
        
        matrix = service._calculate_similarity_matrix(embeddings)
        
        import time
        start = time.time()
        relationships = service._extract_relationships(
            memories,
            matrix,
            threshold=0.7,
            max_per_memory=5
        )
        elapsed = time.time() - start
        
        assert elapsed < 2.0  # Should complete quickly
        assert len(relationships) <= 100 * 5  # Max limit


class TestIntegration:
    """Integration tests with full service flow."""

    @pytest.mark.asyncio
    async def test_full_population_workflow(self, service, mock_db, mock_redis):
        """Test complete populate_relationships workflow."""
        org_id = str(uuid4())
        
        # Mock database queries
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db.execute.return_value = mock_result
        
        result = await service.populate_relationships(org_id)
        
        # Should return stats dict
        assert isinstance(result, dict)
        assert "error" in result or "relationships_found" in result
