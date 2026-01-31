"""
Graph Relationship Service - Auto-populate memory relationships via semantic similarity

Finds similar memories using embeddings and creates relationships in FalkorDB.
Runs periodically or on-demand to keep the knowledge graph up-to-date.
"""

from __future__ import annotations

import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, delete
import redis

from app.models.memory import MemoryMetadata
from app.models.graph_relationship import GraphRelationship
from app.core.config import settings
from app.core.qdrant import QdrantService

logger = logging.getLogger(__name__)


class GraphRelationshipService:
    """
    Auto-populate graph relationships based on semantic similarity.
    
    Process:
    1. Get all memories for organization
    2. Calculate embedding similarity matrix
    3. Filter pairs above threshold
    4. Create RELATES_TO relationships in FalkorDB
    5. Store metadata in PostgreSQL for tracking
    """

    def __init__(self, db: AsyncSession, redis_client: redis.Redis):
        self.db = db
        self.redis = redis_client
        self.graph_name = "ninai_graph"

    async def populate_relationships(
        self,
        org_id: str,
        similarity_threshold: float = 0.75,
        batch_size: int = 100,
        max_relationships_per_memory: int = 5
    ) -> Dict[str, Any]:
        """
        Auto-populate relationships based on embeddings similarity.
        
        Args:
            org_id: Organization ID
            similarity_threshold: Min similarity (0.0-1.0) to create relationship
            batch_size: Process memories in batches
            max_relationships_per_memory: Limit relationships per memory
            
        Returns:
            Stats dict with created/updated/skipped counts
        """
        logger.info(
            f"Starting relationship population for org {org_id} "
            f"(threshold={similarity_threshold})"
        )

        try:
            # Get all active memories with vectors
            memories = await self._get_memories_with_vectors(org_id)
            
            if not memories:
                logger.warning(f"No memories found for org {org_id}")
                return {"created": 0, "updated": 0, "skipped": 0, "errors": 0}

            logger.info(f"Processing {len(memories)} memories")

            # Extract relationships via Qdrant similarity (recommend by point id)
            relationships = await self._extract_relationships_via_qdrant(
                org_id=org_id,
                memories=memories,
                threshold=similarity_threshold,
                max_per_memory=max_relationships_per_memory,
            )

            logger.info(f"Found {len(relationships)} potential relationships")

            # Create relationships in FalkorDB
            created = await self._create_falkordb_relationships(relationships)

            # Store metadata in PostgreSQL
            stored = await self._store_relationship_metadata(org_id, relationships)

            stats = {
                "memories_processed": len(memories),
                "relationships_found": len(relationships),
                "relationships_created": created,
                "relationships_stored": stored,
                "similarity_threshold": similarity_threshold,
                "max_per_memory": max_relationships_per_memory
            }

            logger.info(f"Relationship population complete: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error populating relationships: {e}", exc_info=True)
            return {
                "created": 0,
                "error": str(e),
                "success": False
            }

    async def _get_memories_with_vectors(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Get all active memories with vector references for organization.
        
        Returns:
            List of {"id", "title", "vector_id", "created_at", ...}
        """
        stmt = (
            select(MemoryMetadata)
            .where(
                MemoryMetadata.organization_id == org_id,
                MemoryMetadata.is_active.is_(True),
            )
            .order_by(MemoryMetadata.created_at.desc())
        )

        result = await self.db.execute(stmt)
        memories = result.scalars().all()

        return [
            {
                "id": str(m.id),
                "title": m.title,
                "vector_id": m.vector_id,
                "created_at": m.created_at,
                "org_id": org_id
            }
            for m in memories
        ]

    async def _extract_relationships_via_qdrant(
        self,
        org_id: str,
        memories: List[Dict[str, Any]],
        threshold: float,
        max_per_memory: int,
    ) -> List[Dict[str, Any]]:
        """Extract relationships using Qdrant similarity, scoped to org.

        Uses Qdrant "recommend" by point id (vector_id) so we do not need
        raw embeddings in Postgres.
        """
        relationships: List[Dict[str, Any]] = []
        if not memories:
            return relationships

        # Map vector_id -> memory_id for fast lookup
        vector_to_memory: Dict[str, str] = {
            str(m["vector_id"]): str(m["id"]) for m in memories if m.get("vector_id")
        }

        for memory in memories:
            vector_id = str(memory.get("vector_id") or "")
            if not vector_id:
                continue

            try:
                candidates = await QdrantService.recommend_by_point_id(
                    org_id=org_id,
                    positive_point_id=vector_id,
                    limit=max_per_memory,
                    score_threshold=threshold,
                    with_payload=True,
                )
            except Exception as e:
                logger.warning(f"Qdrant recommend failed for vector {vector_id}: {e}")
                continue

            for candidate in candidates:
                candidate_vector_id = str(candidate.get("id"))
                candidate_memory_id = vector_to_memory.get(candidate_vector_id)
                if not candidate_memory_id:
                    # Candidate might be a memory outside current DB snapshot; ignore.
                    continue
                if candidate_memory_id == memory["id"]:
                    continue

                from_id = str(memory["id"])
                to_id = str(candidate_memory_id)
                similarity_score = float(candidate.get("score") or 0.0)
                if similarity_score < threshold:
                    continue

                # De-dup directionally using lexicographic ordering
                if from_id < to_id:
                    relationships.append(
                        {
                            "from_id": from_id,
                            "to_id": to_id,
                            "org_id": org_id,
                            "similarity_score": similarity_score,
                            "relationship_type": "RELATES_TO",
                        }
                    )

        logger.info(
            f"Extracted {len(relationships)} relationships via Qdrant above threshold {threshold}"
        )
        return relationships

    async def _create_falkordb_relationships(
        self,
        relationships: List[Dict[str, Any]]
    ) -> int:
        """
        Create relationships in FalkorDB.
        
        Returns:
            Number of relationships created
        """
        if not relationships:
            return 0

        created = 0

        for rel in relationships:
            try:
                # Ensure nodes exist; include org_id for tenant isolation
                query = f"""
                MERGE (a:Memory {{id: '{rel['from_id']}', org_id: '{rel.get('org_id', '')}'}})
                MERGE (b:Memory {{id: '{rel['to_id']}', org_id: '{rel.get('org_id', '')}'}})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r.similarity = {rel['similarity_score']},
                    r.auto_created = true,
                    r.created_at = timestamp()
                RETURN r
                """

                self.redis.execute_command("GRAPH.QUERY", self.graph_name, query)
                created += 1

                if created % 100 == 0:
                    logger.info(f"Created {created} relationships in FalkorDB")

            except Exception as e:
                logger.warning(f"Failed to create relationship {rel['from_id']} → {rel['to_id']}: {e}")
                continue

        logger.info(f"Created {created} relationships in FalkorDB")
        return created

    async def _store_relationship_metadata(
        self,
        org_id: str,
        relationships: List[Dict[str, Any]]
    ) -> int:
        """
        Store relationship metadata in PostgreSQL for tracking.
        
        Returns:
            Number stored
        """
        if not relationships:
            return 0

        # Delete old auto-created relationships to avoid duplicates
        await self.db.execute(
            delete(GraphRelationship).where(
                GraphRelationship.organization_id == uuid.UUID(org_id),
                GraphRelationship.auto_created == True
            )
        )

        # Insert new relationships
        org_uuid = uuid.UUID(org_id)

        stmt = insert(GraphRelationship).values([
            {
                "id": uuid.uuid4(),
                "organization_id": org_uuid,
                "from_memory_id": rel["from_id"],
                "to_memory_id": rel["to_id"],
                "relationship_type": rel["relationship_type"],
                "similarity_score": rel["similarity_score"],
                "auto_created": True,
                "created_at": datetime.utcnow(),
                "metadata_": {
                    "algorithm": "cosine_similarity",
                    "version": "1.0"
                }
            }
            for rel in relationships
        ])

        result = await self.db.execute(stmt)
        await self.db.commit()

        logger.info(f"Stored {result.rowcount} relationships in PostgreSQL")
        return result.rowcount

    async def get_relationship_stats(self, org_id: str) -> Dict[str, Any]:
        """
        Get statistics about relationships for an organization.
        
        Returns:
            Stats dict with totals and averages
        """
        stmt = select(GraphRelationship).where(
            GraphRelationship.organization_id == uuid.UUID(org_id)
        )

        result = await self.db.execute(stmt)
        relationships = result.scalars().all()

        if not relationships:
            return {"total": 0, "avg_similarity": 0, "auto_created": 0}

        similarities = [float(r.similarity_score) for r in relationships if r.similarity_score is not None]
        auto_created = sum(1 for r in relationships if r.auto_created)

        return {
            "total": len(relationships),
            "auto_created": auto_created,
            "manually_created": len(relationships) - auto_created,
            "avg_similarity": (sum(similarities) / len(similarities)) if similarities else 0,
            "min_similarity": min(similarities) if similarities else 0,
            "max_similarity": max(similarities) if similarities else 0,
        }

    async def update_config(
        self,
        org_id: str,
        similarity_threshold: Optional[float] = None,
        max_relationships: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update relationship generation config for organization.
        
        Stores in Redis for fast access.
        """
        config_key = f"graph_config:{org_id}"
        config = self.redis.hgetall(config_key) or {}

        if similarity_threshold is not None:
            config["similarity_threshold"] = str(similarity_threshold)
        if max_relationships is not None:
            config["max_relationships"] = str(max_relationships)

        self.redis.hset(config_key, mapping=config)

        logger.info(f"Updated graph config for {org_id}: {config}")
        return config

    async def get_config(self, org_id: str) -> Dict[str, Any]:
        """Get relationship generation config for organization."""
        config_key = f"graph_config:{org_id}"
        config = self.redis.hgetall(config_key) or {}

        return {
            "similarity_threshold": float(config.get("similarity_threshold", 0.75)),
            "max_relationships": int(config.get("max_relationships", 5)),
        }


async def get_graph_relationship_service(
    db: AsyncSession,
    redis_client: redis.Redis
) -> GraphRelationshipService:
    """Dependency to get service instance."""
    return GraphRelationshipService(db, redis_client)
