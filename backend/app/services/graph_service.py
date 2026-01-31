"""
FalkorDB Graph Service - Relationship Queries and Graph Analysis

Provides graph-based relationship queries for memories, knowledge, and goals.
Uses FalkorDB (Redis-based graph database) - BSD-3-Clause license, no GPL issues.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.core.config import settings

logger = logging.getLogger(__name__)


class FalkorDBGraphService:
    """
    FalkorDB graph database service for relationship queries.
    
    Uses Redis-based FalkorDB (fork of RedisGraph) with Cypher queries.
    BSD-3-Clause license - compatible with commercial use, no GPL contamination.
    
    Provides:
    - Memory relationship mapping
    - Knowledge graph traversal
    - Goal dependency analysis  
    - Entity relationship queries
    - Path finding between nodes
    """
    
    def __init__(self, redis_url: str | None = None, graph_name: str = "ninai_graph"):
        """
        Initialize FalkorDB connection via Redis.
        
        Args:
            redis_url: Redis connection URL (default from settings)
            graph_name: Graph database name (default: "ninai_graph")
        """
        if not REDIS_AVAILABLE:
            logger.warning("Redis library not available - graph features disabled")
            self.redis = None
            self.graph_name = graph_name
            return
        
        self.redis_url = redis_url or getattr(settings, "REDIS_URL", "redis://localhost:6379/0")
        self.graph_name = graph_name
        
        try:
            self.redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            self.redis.ping()
            logger.info(f"✓ FalkorDB graph service initialized: {self.redis_url} (graph: {graph_name})")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for FalkorDB: {e}")
            self.redis = None
    
    def close(self):
        """Close Redis connection."""
        if self.redis:
            self.redis.close()
    
    def _execute_query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        """
        Execute Cypher query via FalkorDB.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            
        Returns:
            List of result records
        """
        if not self.redis:
            return []
        
        try:
            # Build parameterized query (simple substitution for FalkorDB)
            query = cypher
            if params:
                for key, value in params.items():
                    if isinstance(value, str):
                        query = query.replace(f"${key}", f"'{value}'")
                    elif isinstance(value, list):
                        query = query.replace(f"${key}", json.dumps(value))
                    else:
                        query = query.replace(f"${key}", str(value))
            
            # Execute via Redis GRAPH.QUERY command
            result = self.redis.execute_command("GRAPH.QUERY", self.graph_name, query)
            
            # Parse results (FalkorDB returns [header, rows, stats])
            if not result or len(result) < 2:
                return []
            
            headers = result[0] if result[0] else []
            rows = result[1] if len(result) > 1 else []
            
            return [
                {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                for row in rows
            ]
        except Exception as e:
            logger.error(f"FalkorDB query failed: {e}")
            return []
    
    async def create_memory_node(
        self,
        memory_id: str,
        org_id: str,
        user_id: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create a memory node in the graph.
        
        Args:
            memory_id: Unique memory identifier
            org_id: Organization ID
            user_id: User ID who created the memory
            title: Memory title
            content: Memory content
            tags: Optional tags
            metadata: Optional metadata
            
        Returns:
            Created node properties
        """
        if not self.redis:
            logger.warning("FalkorDB not available - skipping node creation")
            return {}
        
        query = """
        MERGE (m:Memory {id: $memory_id})
        SET m.org_id = $org_id,
            m.user_id = $user_id,
            m.title = $title,
            m.content = $content,
            m.tags = $tags,
            m.updated_at = timestamp()
        RETURN m
        """
        
        params = {
            "memory_id": memory_id,
            "org_id": org_id,
            "user_id": user_id,
            "title": title,
            "content": content[:1000],
            "tags": json.dumps(tags or [])
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._execute_query, query, params)
        return result[0] if result else {}
    
    async def create_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            relationship_type: Type of relationship (e.g., "RELATES_TO", "DEPENDS_ON")
            properties: Optional relationship properties
            
        Returns:
            Created relationship properties
        """
        if not self.redis:
            return {}
        
        query = f"""
        MATCH (a {{id: $from_id}})
        MATCH (b {{id: $to_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r.created_at = timestamp()
        RETURN r
        """
        
        params = {
            "from_id": from_id,
            "to_id": to_id
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._execute_query, query, params)
        return result[0] if result else {}
    
    async def find_related_memories(
        self,
        memory_id: str,
        org_id: str,
        relationship_types: list[str] | None = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Find memories related to a given memory.
        
        Args:
            memory_id: Source memory ID
            org_id: Organization ID for filtering
            relationship_types: Optional relationship type filter
            max_depth: Maximum traversal depth
            limit: Maximum results
            
        Returns:
            List of related memory nodes with relationship info
        """
        if not self.redis:
            return []
        
        rel_filter = f"[*1..{max_depth}]"
        
        query = f"""
        MATCH path = (m:Memory {{id: $memory_id, org_id: $org_id}})-{rel_filter}-(related:Memory)
        WHERE related.org_id = $org_id
        RETURN DISTINCT related, length(path) as depth
        ORDER BY depth ASC
        LIMIT $limit
        """
        
        params = {
            "memory_id": memory_id,
            "org_id": org_id,
            "limit": limit
        }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute_query, query, params)
    
    async def find_shortest_path(
        self,
        from_id: str,
        to_id: str,
        org_id: str,
        max_depth: int = 5
    ) -> dict[str, Any]:
        """
        Find shortest path between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            org_id: Organization ID
            max_depth: Maximum path length
            
        Returns:
            Path information with nodes and relationships
        """
        if not self.redis:
            return {}
        
        query = """
        MATCH path = shortestPath(
            (a {id: $from_id, org_id: $org_id})-[*1..5]-(b {id: $to_id, org_id: $org_id})
        )
        RETURN path, length(path) as pathLength
        """
        
        params = {
            "from_id": from_id,
            "to_id": to_id,
            "org_id": org_id
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._execute_query, query, params)
        return result[0] if result else {}
    
    async def get_node_degree(
        self,
        node_id: str,
        org_id: str,
        direction: str = "both"
    ) -> dict[str, int]:
        """
        Get degree (number of connections) for a node.
        
        Args:
            node_id: Node ID
            org_id: Organization ID
            direction: "in", "out", or "both"
            
        Returns:
            Degree counts
        """
        if not self.redis:
            return {"in": 0, "out": 0, "total": 0}
        
        query = """
        MATCH (n {id: $node_id, org_id: $org_id})
        OPTIONAL MATCH (n)<-[inRel]-()
        OPTIONAL MATCH (n)-[outRel]->()
        RETURN count(DISTINCT inRel) as in_degree,
               count(DISTINCT outRel) as out_degree
        """
        
        params = {
            "node_id": node_id,
            "org_id": org_id
        }
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._execute_query, query, params)
        
        if not result:
            return {"in": 0, "out": 0, "total": 0}
        
        in_deg = result[0].get("in_degree", 0)
        out_deg = result[0].get("out_degree", 0)
        
        return {
            "in": in_deg,
            "out": out_deg,
            "total": in_deg + out_deg
        }
    
    async def find_communities(
        self,
        org_id: str,
        algorithm: str = "louvain",
        min_size: int = 3
    ) -> list[dict[str, Any]]:
        """
        Detect communities/clusters in the memory graph.
        
        Args:
            org_id: Organization ID
            algorithm: Community detection algorithm
            min_size: Minimum community size
            
        Returns:
            List of communities with member nodes
        """
        if not self.redis:
            return []
        
        # FalkorDB doesn't have built-in community detection algorithms
        logger.info("Community detection not supported in FalkorDB - requires external algorithm")
        return []
    
    async def get_graph_statistics(self, org_id: str) -> dict[str, Any]:
        """
        Get overall graph statistics for an organization.
        
        Args:
            org_id: Organization ID
            
        Returns:
            Graph statistics
        """
        if not self.redis:
            return {
                "enabled": False,
                "total_nodes": 0,
                "total_relationships": 0
            }
        
        query = """
        MATCH (n {org_id: $org_id})
        OPTIONAL MATCH (n)-[r]-()
        RETURN 
            count(DISTINCT n) as node_count,
            count(DISTINCT r) as rel_count
        """
        
        params = {"org_id": org_id}
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._execute_query, query, params)
        
        if not result:
            return {"enabled": True, "total_nodes": 0, "total_relationships": 0}
        
        return {
            "enabled": True,
            "total_nodes": result[0].get("node_count", 0),
            "total_relationships": result[0].get("rel_count", 0)
        }


# Global instance (initialized on demand)
_graph_service: FalkorDBGraphService | None = None


def get_graph_service() -> FalkorDBGraphService:
    """Get or create FalkorDB graph service instance."""
    global _graph_service
    if _graph_service is None:
        _graph_service = FalkorDBGraphService()
    return _graph_service
