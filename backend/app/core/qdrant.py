"""
Qdrant Vector Database Client
=============================

Client for Qdrant vector database operations with built-in
organization filtering for multi-tenant security.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    PointStruct,
    VectorParams,
    Distance,
)

from app.core.config import settings


class QdrantService:
    """
    Qdrant vector database service.
    
    Provides methods for vector operations with automatic
    organization filtering for multi-tenant isolation.
    
    SECURITY: All search operations MUST include organization_id filter.
    """
    
    _client: Optional[QdrantClient] = None
    
    @classmethod
    def get_client(cls) -> QdrantClient:
        """
        Get or create Qdrant client.
        
        Returns:
            QdrantClient: Configured Qdrant client
        """
        if cls._client is None:
            cls._client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            )
        return cls._client
    
    @classmethod
    async def ensure_collection(cls) -> None:
        """
        Ensure the memories collection exists with proper configuration.
        
        Creates the collection if it doesn't exist with appropriate
        vector dimensions and distance metric.
        """
        client = cls.get_client()
        collection_name = settings.QDRANT_COLLECTION_NAME
        
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )
    
    @classmethod
    def build_org_filter(
        cls,
        org_id: str,
        additional_filters: Optional[List[FieldCondition]] = None,
    ) -> Filter:
        """
        Build a Qdrant filter that includes organization isolation.
        
        CRITICAL: This method ensures all vector searches are scoped
        to the requesting organization. Never bypass this!
        
        Args:
            org_id: Organization UUID to filter by
            additional_filters: Optional additional filter conditions
        
        Returns:
            Filter: Qdrant filter with org isolation (or just additional filters if no org)
        """
        must_conditions = []
        
        # Only add org filter if org_id is provided
        if org_id:
            must_conditions.append(
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=org_id),
                )
            )
        
        if additional_filters:
            must_conditions.extend(additional_filters)
        
        # Return filter only if there are conditions
        if must_conditions:
            return Filter(must=must_conditions)
        return None
    
    @classmethod
    async def upsert_memory(
        cls,
        memory_id: str,
        org_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> bool:
        """
        Upsert a memory vector with payload.
        
        Args:
            memory_id: Unique memory identifier
            org_id: Organization UUID (stored in payload for filtering)
            vector: Embedding vector
            payload: Additional metadata to store
        
        Returns:
            bool: True if operation successful
        """
        await cls.ensure_collection()
        client = cls.get_client()
        
        # Always include organization_id in payload for filtering
        payload["organization_id"] = org_id
        
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload=payload,
                ),
            ],
        )
        return True
    
    @classmethod
    async def search(
        cls,
        org_id: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        scope_filter: Optional[str] = None,
        team_id: Optional[str] = None,
        classification_max: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories with organization filtering.
        
        SECURITY: Always filters by organization_id. Results must still
        be verified against Postgres RLS before returning to user.
        
        Args:
            org_id: Organization UUID (required)
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            scope_filter: Optional scope filter (personal/team/org)
            team_id: Optional team filter
            classification_max: Optional max classification level
        
        Returns:
            List of search results with scores and payloads
        """
        client = cls.get_client()
        
        # Build filter conditions
        filter_conditions = []
        
        if scope_filter:
            filter_conditions.append(
                FieldCondition(
                    key="scope",
                    match=MatchValue(value=scope_filter),
                )
            )
        
        if team_id:
            filter_conditions.append(
                FieldCondition(
                    key="team_id",
                    match=MatchValue(value=team_id),
                )
            )
        
        # Build filter with org isolation
        search_filter = cls.build_org_filter(org_id, filter_conditions)
        
        # Perform search
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": str(result.id),
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

    @classmethod
    async def recommend_by_point_id(
        cls,
        org_id: str,
        positive_point_id: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """Recommend similar vectors using an existing point id.

        This avoids needing raw embeddings in Postgres.

        Args:
            org_id: Organization UUID (required for tenant isolation)
            positive_point_id: Existing Qdrant point id to use as the seed
            limit: Maximum results
            score_threshold: Minimum similarity score
            with_payload: Include payload in results
        """
        client = cls.get_client()

        recommend_filter = cls.build_org_filter(org_id)

        results = client.recommend(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            positive=[positive_point_id],
            negative=None,
            query_filter=recommend_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=False,
        )

        return [
            {
                "id": str(result.id),
                "score": result.score,
                "payload": getattr(result, "payload", None),
            }
            for result in results
        ]
    
    @classmethod
    async def delete_memory(
        cls,
        memory_id: str,
        org_id: str,
    ) -> bool:
        """
        Delete a memory vector.
        
        Args:
            memory_id: Memory UUID to delete
            org_id: Organization UUID (for verification)
        
        Returns:
            bool: True if deleted
        """
        client = cls.get_client()
        
        # Delete with org filter for safety
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.PointIdsList(
                points=[memory_id],
            ),
        )
        return True
    
    @classmethod
    async def delete_by_org(cls, org_id: str) -> bool:
        """
        Delete all memories for an organization.
        
        Use with caution! This is for org deletion/cleanup.
        
        Args:
            org_id: Organization UUID
        """
        client = cls.get_client()
        
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="organization_id",
                            match=MatchValue(value=org_id),
                        ),
                    ],
                ),
            ),
        )
        return True

    @classmethod
    async def delete_point(cls, point_id: str) -> bool:
        """Delete a single point by id (memory vector or attachment vector)."""
        client = cls.get_client()
        client.delete(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points_selector=qdrant_models.PointIdsList(points=[point_id]),
        )
        return True
