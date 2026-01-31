"""
Tests for Memory Consolidation Service

Test Coverage:
- Finding consolidation candidates (similarity-based)
- Consolidating memories with conflict resolution
- Relationship remapping
- Metadata merging
- Audit trail creation
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryMetadata
from app.models.graph_relationship import GraphRelationship
from app.services.consolidation_service import ConsolidationService


@pytest.mark.asyncio
class TestFindConsolidationCandidates:
    """Test finding consolidation candidates."""
    
    async def test_find_candidates_by_similarity(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test finding similar memories."""
        # Create test memories
        mem1_id = str(uuid.uuid4())
        mem2_id = str(uuid.uuid4())
        mem3_id = str(uuid.uuid4())
        
        mem1 = MemoryMetadata(
            id=mem1_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Python programming tutorial for beginners",
            content_hash="hash1",
            tags=["python", "tutorial"],
            entities={"topic": ["programming"]},
            source_type="manual",
            vector_id="vec1",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        mem2 = MemoryMetadata(
            id=mem2_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Python programming basics for beginners",  # Very similar
            content_hash="hash2",
            tags=["python", "basics"],
            entities={"topic": ["programming"]},
            source_type="manual",
            vector_id="vec2",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        mem3 = MemoryMetadata(
            id=mem3_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="JavaScript best practices",  # Different topic
            content_hash="hash3",
            tags=["javascript"],
            entities={"topic": ["web"]},
            source_type="manual",
            vector_id="vec3",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add_all([mem1, mem2, mem3])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        # Should not find candidates (no vector search in test)
        # This test validates structure, actual similarity matching needs Qdrant
        candidates = await service.find_consolidation_candidates()
        assert isinstance(candidates, list)
    
    async def test_find_candidates_for_specific_memory(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test finding candidates for specific memory."""
        mem_id = str(uuid.uuid4())
        mem = MemoryMetadata(
            id=mem_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Test memory",
            content_hash="hash1",
            tags=["test"],
            entities={},
            source_type="manual",
            vector_id="vec1",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add(mem)
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        candidates = await service.find_consolidation_candidates(
            memory_id=mem_id
        )
        
        assert isinstance(candidates, list)
    
    async def test_find_candidates_with_scope_filter(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test finding candidates filtered by scope."""
        mem1 = MemoryMetadata(
            id=str(uuid.uuid4()),
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Personal note",
            content_hash="hash1",
            tags=["personal"],
            entities={},
            source_type="manual",
            vector_id="vec_personal",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        mem2 = MemoryMetadata(
            id=str(uuid.uuid4()),
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="team",
            classification="public",
            content_preview="Team note",
            content_hash="hash2",
            tags=["team"],
            entities={},
            source_type="manual",
            vector_id="vec_team",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add_all([mem1, mem2])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        # Find candidates in personal scope
        candidates = await service.find_consolidation_candidates(
            scope="personal"
        )
        
        assert isinstance(candidates, list)


@pytest.mark.asyncio
class TestConsolidateMemories:
    """Test consolidating memories."""
    
    async def test_consolidate_basic(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test basic memory consolidation."""
        primary_id = str(uuid.uuid4())
        dup_id = str(uuid.uuid4())
        
        primary = MemoryMetadata(
            id=primary_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Python tutorial",
            content_hash="hash1",
            tags=["python", "tutorial"],
            entities={"topic": ["programming"]},
            extra_metadata={"difficulty": "beginner"},
            source_type="manual",
            vector_id="vec_primary",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        duplicate = MemoryMetadata(
            id=dup_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Python basics",
            content_hash="hash2",
            tags=["python", "basics"],
            entities={"topic": ["programming"]},
            extra_metadata={"difficulty": "beginner"},
            source_type="manual",
            vector_id="vec_duplicate",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add_all([primary, duplicate])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        result = await service.consolidate(
            primary_id=primary_id,
            duplicate_ids=[dup_id]
        )
        
        assert result["primary_id"] == primary_id
        assert dup_id in result["consolidated_ids"]
        assert len(result["merged_metadata"]["tags"]) >= 2
    
    async def test_consolidate_merges_tags(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test that consolidation merges tags correctly."""
        primary_id = str(uuid.uuid4())
        dup_id = str(uuid.uuid4())
        
        primary = MemoryMetadata(
            id=primary_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Memory A",
            content_hash="hash1",
            tags=["tag1", "tag2"],
            entities={},
            source_type="manual",
            vector_id="vec_mem_a",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        duplicate = MemoryMetadata(
            id=dup_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Memory B",
            content_hash="hash2",
            tags=["tag2", "tag3"],  # tag2 is duplicate
            entities={},
            source_type="manual",
            vector_id="vec_mem_b",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add_all([primary, duplicate])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        result = await service.consolidate(
            primary_id=primary_id,
            duplicate_ids=[dup_id]
        )
        
        merged_tags = result["merged_metadata"]["tags"]
        assert "tag1" in merged_tags
        assert "tag2" in merged_tags
        assert "tag3" in merged_tags
        assert len(merged_tags) == 3  # No duplicates
    
    async def test_consolidate_merges_entities(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test that consolidation merges entities correctly."""
        primary_id = str(uuid.uuid4())
        dup_id = str(uuid.uuid4())
        
        primary = MemoryMetadata(
            id=primary_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Memory A",
            content_hash="hash1",
            tags=[],
            entities={"person": ["Alice", "Bob"], "place": ["NYC"]},
            source_type="manual",
            vector_id="vec_entities_a",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        duplicate = MemoryMetadata(
            id=dup_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Memory B",
            content_hash="hash2",
            tags=[],
            entities={"person": ["Bob", "Charlie"], "place": ["LA"]},
            source_type="manual",
            vector_id="vec_entities_b",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add_all([primary, duplicate])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        result = await service.consolidate(
            primary_id=primary_id,
            duplicate_ids=[dup_id]
        )
        
        merged_entities = result["merged_metadata"]["entities"]
        assert "person" in merged_entities
        assert "place" in merged_entities
        assert len(merged_entities["person"]) >= 3  # Alice, Bob, Charlie
    
    async def test_consolidate_archives_duplicates(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test that duplicates are archived."""
        primary_id = str(uuid.uuid4())
        dup_id = str(uuid.uuid4())
        
        primary = MemoryMetadata(
            id=primary_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Primary",
            content_hash="hash1",
            tags=[],
            entities={},
            source_type="manual",
            vector_id="vec_archive_primary",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        duplicate = MemoryMetadata(
            id=dup_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Duplicate",
            content_hash="hash2",
            tags=[],
            entities={},
            source_type="manual",
            vector_id="vec_archive_duplicate",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add_all([primary, duplicate])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        await service.consolidate(
            primary_id=primary_id,
            duplicate_ids=[dup_id]
        )
        
        # Verify duplicate is archived
        from sqlalchemy import select
        stmt = select(MemoryMetadata).where(
            MemoryMetadata.id == dup_id
        )
        dup = (await db_session.execute(stmt)).scalar_one()
        # MemoryMetadata uses soft-delete via is_active.
        assert dup.is_active == False


@pytest.mark.asyncio
class TestRelationshipRemapping:
    """Test relationship remapping during consolidation."""
    
    async def test_remap_relationships_from_duplicate(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test remapping relationships from duplicate memory."""
        primary_id = str(uuid.uuid4())
        dup_id = str(uuid.uuid4())
        target_id = str(uuid.uuid4())
        
        # Create memories
        primary = MemoryMetadata(
            id=primary_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Primary",
            content_hash="hash1",
            tags=[],
            entities={},
            source_type="manual",
            vector_id="vec_rel_primary",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        duplicate = MemoryMetadata(
            id=dup_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Duplicate",
            content_hash="hash2",
            tags=[],
            entities={},
            source_type="manual",
            vector_id="vec_rel_duplicate",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        target = MemoryMetadata(
            id=target_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Target",
            content_hash="hash3",
            tags=[],
            entities={},
            source_type="manual",
            vector_id="vec_rel_target",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        # Create relationship from duplicate to target
        rel = GraphRelationship(
            id=str(uuid.uuid4()),
            organization_id=test_org_id,
            from_memory_id=dup_id,
            to_memory_id=target_id,
            relationship_type="RELATES_TO",
            similarity_score=0.85
        )
        
        db_session.add_all([primary, duplicate, target, rel])
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        await service.consolidate(
            primary_id=primary_id,
            duplicate_ids=[dup_id]
        )
        
        # Verify relationship was remapped
        from sqlalchemy import select
        stmt = select(GraphRelationship).where(
            GraphRelationship.from_memory_id == primary_id,
            GraphRelationship.to_memory_id == target_id
        )
        updated_rel = (await db_session.execute(stmt)).scalar_one_or_none()
        assert updated_rel is not None


@pytest.mark.asyncio
class TestConsolidationStatus:
    """Test getting consolidation status."""
    
    async def test_get_status_not_consolidated(
        self,
        db_session: AsyncSession,
        test_org_id: str,
        test_user_id: str,
    ):
        """Test status of non-consolidated memory."""
        mem_id = str(uuid.uuid4())
        
        mem = MemoryMetadata(
            id=mem_id,
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="public",
            content_preview="Test",
            content_hash="hash1",
            tags=["tag1"],
            entities={"type": ["entity"]},
            source_type="manual",
            vector_id="vec_status_test",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )
        
        db_session.add(mem)
        await db_session.commit()
        
        service = ConsolidationService(db_session, test_org_id)
        
        status = await service.get_consolidation_status(mem_id)
        
        assert status["memory_id"] == mem_id
        assert status["is_consolidated"] == False
        assert len(status["tags"]) >= 1


@pytest.mark.asyncio
class TestTextSimilarity:
    """Test text similarity calculation."""
    
    async def test_text_similarity_identical(
        self,
        db_session: AsyncSession,
        test_org_id: str
    ):
        """Test similarity of identical texts."""
        service = ConsolidationService(db_session, test_org_id)
        
        text = "Python programming tutorial"
        sim = service._calculate_text_similarity(text, text)
        
        assert sim == 1.0
    
    async def test_text_similarity_different(
        self,
        db_session: AsyncSession,
        test_org_id: str
    ):
        """Test similarity of different texts."""
        service = ConsolidationService(db_session, test_org_id)
        
        text1 = "Python programming"
        text2 = "JavaScript development"
        sim = service._calculate_text_similarity(text1, text2)
        
        assert sim == 0.0  # No common tokens
    
    async def test_text_similarity_partial(
        self,
        db_session: AsyncSession,
        test_org_id: str
    ):
        """Test similarity of partially matching texts."""
        service = ConsolidationService(db_session, test_org_id)
        
        text1 = "Python programming tutorial"
        text2 = "Python programming basics"
        sim = service._calculate_text_similarity(text1, text2)
        
        assert 0 < sim < 1  # Partial match
