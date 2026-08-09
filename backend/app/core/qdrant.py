"""
Qdrant Vector Database Client
=============================

Client for Qdrant vector database operations with built-in
organization filtering for multi-tenant security.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    Range,
    PointStruct,
    VectorParams,
    Distance,
)

from app.core.config import settings


logger = logging.getLogger(__name__)


class QdrantService:
    """
    Qdrant vector database service.
    
    Provides methods for vector operations with automatic
    organization filtering for multi-tenant isolation.
    
    SECURITY: All search operations MUST include organization_id filter.
    """
    
    _client: Optional[QdrantClient] = None
    _collection_ready: bool = False
    
    @classmethod
    def get_client(cls) -> QdrantClient:
        """
        Get or create Qdrant client.
        
        Returns:
            QdrantClient: Configured Qdrant client
        """
        if cls._client is None:
            api_key_raw = settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
            api_key = str(api_key_raw).strip() if api_key_raw else None
            if api_key == '':
                api_key = None
            timeout = float(getattr(settings, "QDRANT_TIMEOUT_SECONDS", 2.5) or 2.5)
            qdrant_url_raw = getattr(settings, "QDRANT_URL", None)
            qdrant_url = str(qdrant_url_raw).strip() if qdrant_url_raw else None
            if qdrant_url == '':
                qdrant_url = None
            if qdrant_url:
                cls._client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=timeout)
            else:
                cls._client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    api_key=api_key,
                    timeout=timeout,
                )
        return cls._client

    @classmethod
    async def _run_sync_with_timeout(cls, fn, *args, timeout: float = 2.5, **kwargs):
        """Run blocking qdrant-client calls off the event loop with a hard timeout."""
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args, **kwargs),
            timeout=timeout,
        )
    
    @classmethod
    async def ensure_collection(cls) -> bool:
        """Ensure the memories collection exists with the configured vector dimensions.

        If the collection exists but was created with a different dimension (e.g. switching
        from OpenAI 1536-dim to vllm-embed bge-base-en-v1.5 768-dim), the old collection is
        deleted and recreated automatically so embeddings remain consistent.
        """
        if cls._collection_ready:
            return True

        client = cls.get_client()
        collection_name = settings.QDRANT_COLLECTION_NAME
        target_dim = settings.EMBEDDING_DIMENSIONS

        try:
            collections = await cls._run_sync_with_timeout(client.get_collections)
            existing = {c.name for c in collections.collections}

            if collection_name in existing:
                info = await cls._run_sync_with_timeout(client.get_collection, collection_name)
                current_dim = info.config.params.vectors.size
                if current_dim != target_dim:
                    logger.warning(
                        "Qdrant collection '%s' has dim=%d but config wants dim=%d — recreating",
                        collection_name, current_dim, target_dim,
                    )
                    await cls._run_sync_with_timeout(client.delete_collection, collection_name)
                    existing.discard(collection_name)

            if collection_name not in existing:
                await cls._run_sync_with_timeout(
                    client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=target_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection '%s' with dim=%d", collection_name, target_dim)

            cls._collection_ready = True
            return True
        except Exception as exc:
            logger.warning("Qdrant unavailable during collection ensure; continuing in degraded mode: %s", exc)
            cls._collection_ready = False
            return False
    
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
        # Skip upsert for zero vectors — cosine distance is undefined for the zero vector,
        # and Qdrant will reject or silently corrupt such entries.
        if not any(vector):
            logger.debug("Skipping Qdrant upsert for memory_id=%s: zero embedding vector.", memory_id)
            return False

        if not await cls.ensure_collection():
            return False
        client = cls.get_client()
        
        # Always include organization_id in payload for filtering
        payload["organization_id"] = org_id
        
        try:
            await cls._run_sync_with_timeout(
                client.upsert,
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
        except Exception as exc:
            logger.warning("Qdrant upsert failed for memory_id=%s: %s", memory_id, exc)
            return False
    
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
        tags: Optional[List[str]] = None,
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
        timeout = float(getattr(settings, "QDRANT_TIMEOUT_SECONDS", 2.5) or 2.5)

        # If the query vector is all zeros (no embedding available), skip vector search.
        if not any(query_vector):
            return []

        # Ensure the collection exists before searching.
        # If Qdrant is configured but unavailable, surface the failure instead of
        # pretending the query legitimately returned no vector matches.
        if not await cls.ensure_collection():
            if settings.QDRANT_URL or (settings.QDRANT_HOST and settings.QDRANT_PORT):
                raise RuntimeError("Qdrant collection is unavailable for vector search")
            return []

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

        if tags:
            for tag in tags:
                filter_conditions.append(
                    FieldCondition(
                        key="tags",
                        match=MatchAny(any=[tag]),
                    )
                )
        
        # Build filter with org isolation
        search_filter = cls.build_org_filter(org_id, filter_conditions)
        
        # Perform search
        try:
            results = await cls._run_sync_with_timeout(
                client.search,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning(
                "Qdrant search failed for org_id=%s scope=%s team_id=%s tags=%s collection=%s: %s",
                org_id,
                scope_filter,
                team_id,
                tags,
                settings.QDRANT_COLLECTION_NAME,
                exc,
            )
            raise RuntimeError("Qdrant vector search failed") from exc

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

        if not await cls.ensure_collection():
            return []

        recommend_filter = cls.build_org_filter(org_id)

        try:
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
        except Exception as exc:
            logger.warning("Qdrant recommend failed for point_id=%s: %s", positive_point_id, exc)
            return []

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
