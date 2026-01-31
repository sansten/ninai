"""
LlamaIndex Vector Store Adapter

Integrates Ninai Memory OS with LlamaIndex's VectorStore interface.
Supports document indexing, retrieval, and hybrid search.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime

try:
    from llama_index.core.vector_stores import VectorStore, VectorStoreQuery, VectorStoreQueryResult
    from llama_index.core.schema import BaseNode, TextNode, NodeRelationship
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    VectorStore = object  # Fallback
    BaseNode = object
    TextNode = object
    NodeRelationship = object
    VectorStoreQuery = object
    VectorStoreQueryResult = object

logger = logging.getLogger(__name__)


class NinaiLlamaIndexVectorStore(VectorStore):
    """
    LlamaIndex vector store adapter using Ninai Memory OS.
    
    Features:
    - Document storage with embeddings
    - Hybrid search (vector + BM25)
    - Metadata filtering
    - Organization scoping
    - Automatic memory promotion
    
    Usage:
        ```python
        from integrations.llamaindex_adapter import NinaiLlamaIndexVectorStore
        from llama_index.core import VectorStoreIndex, Document
        
        vector_store = NinaiLlamaIndexVectorStore(
            api_base_url="http://localhost:8002/api/v1",
            api_key="your-capability-token"
        )
        
        index = VectorStoreIndex.from_vector_store(vector_store)
        index.insert(Document(text="Your content here"))
        
        query_engine = index.as_query_engine()
        response = query_engine.query("What is...")
        ```
    """
    
    stores_text: bool = True
    is_embedding_query: bool = True
    
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        scope: str = "private",
        **kwargs
    ):
        """
        Initialize Ninai LlamaIndex Vector Store.
        
        Args:
            api_base_url: Base URL for Ninai API
            api_key: Capability token for authentication
            user_id: User identifier (optional)
            org_id: Organization identifier (optional)
            scope: Memory scope (private/team/organization/public)
        """
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is not installed. Install with: pip install llama-index"
            )
        
        super().__init__(**kwargs)
        
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.user_id = user_id
        self.org_id = org_id
        self.scope = scope
        
        import httpx
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    
    @property
    def client(self) -> Any:
        """Return HTTP client."""
        return self._client
    
    @client.setter
    def client(self, value: Any) -> None:
        """Set HTTP client."""
        self._client = value
    
    def add(self, nodes: List[BaseNode], **kwargs) -> List[str]:
        """
        Add nodes to vector store.
        
        Args:
            nodes: List of nodes to add
            
        Returns:
            List of node IDs
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            return asyncio.run(self._aadd(nodes, **kwargs))
        
        except Exception as e:
            logger.error(f"Error adding nodes: {e}")
            return []
    
    async def _aadd(self, nodes: List[BaseNode], **kwargs) -> List[str]:
        """Async add nodes."""
        node_ids = []
        
        for node in nodes:
            try:
                # Extract node data
                text = node.get_content()
                embedding = node.get_embedding() if hasattr(node, 'get_embedding') else None
                metadata = node.metadata or {}
                
                # Create memory
                payload = {
                    "title": metadata.get("title", f"Document: {text[:50]}..."),
                    "content": text,
                    "scope": self.scope,
                    "tags": metadata.get("tags", ["llamaindex", "document"]),
                    "metadata": {
                        **metadata,
                        "node_id": node.node_id,
                        "framework": "llamaindex",
                        "doc_id": node.ref_doc_id if hasattr(node, 'ref_doc_id') else None,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                if embedding:
                    payload["embedding"] = embedding
                
                response = await self.client.post(
                    f"{self.api_base_url}/memories",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                node_ids.append(data["id"])
            
            except Exception as e:
                logger.error(f"Error adding node {node.node_id}: {e}")
        
        return node_ids
    
    async def adelete(self, ref_doc_id: str, **kwargs) -> None:
        """
        Delete document by reference ID.
        
        Args:
            ref_doc_id: Reference document ID
        """
        try:
            # Find all memories with this doc_id
            response = await self.client.get(
                f"{self.api_base_url}/memories",
                params={"limit": 1000}
            )
            response.raise_for_status()
            
            data = response.json()
            memory_ids = [
                m["id"] for m in data.get("items", [])
                if m.get("metadata", {}).get("doc_id") == ref_doc_id
            ]
            
            if memory_ids:
                # Batch delete
                await self.client.post(
                    f"{self.api_base_url}/memories/batch/delete",
                    json={"memory_ids": memory_ids}
                )
        
        except Exception as e:
            logger.error(f"Error deleting document {ref_doc_id}: {e}")
    
    def delete(self, ref_doc_id: str, **kwargs) -> None:
        """Delete document (sync wrapper)."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            asyncio.run(self.adelete(ref_doc_id, **kwargs))
        
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
    
    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        """
        Query vector store.
        
        Args:
            query: Vector store query
            
        Returns:
            Query results with nodes and scores
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            return asyncio.run(self._aquery(query, **kwargs))
        
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    
    async def _aquery(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        """Async query."""
        try:
            # Build search payload
            payload = {
                "limit": query.similarity_top_k,
                "hnms_mode": "AUTO"  # Hybrid search
            }
            
            # Add query
            if query.query_str:
                payload["query"] = query.query_str
            elif query.query_embedding:
                payload["embedding"] = query.query_embedding
            
            # Add filters
            if query.filters:
                payload["filters"] = query.filters.filters if hasattr(query.filters, 'filters') else {}
            
            # Execute search
            response = await self.client.post(
                f"{self.api_base_url}/memories/search",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Convert to LlamaIndex format
            nodes = []
            similarities = []
            ids = []
            
            for result in results:
                # Create TextNode
                node = TextNode(
                    text=result["content"],
                    id_=result["id"],
                    metadata=result.get("metadata", {}),
                    relationships={
                        NodeRelationship.SOURCE: result.get("metadata", {}).get("doc_id")
                    }
                )
                
                nodes.append(node)
                similarities.append(result.get("score", 0.0))
                ids.append(result["id"])
            
            return VectorStoreQueryResult(
                nodes=nodes,
                similarities=similarities,
                ids=ids
            )
        
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
    
    async def aget_nodes(self, node_ids: List[str], **kwargs) -> List[BaseNode]:
        """
        Get nodes by IDs.
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            List of nodes
        """
        nodes = []
        
        for node_id in node_ids:
            try:
                response = await self.client.get(
                    f"{self.api_base_url}/memories/{node_id}"
                )
                response.raise_for_status()
                
                data = response.json()
                
                node = TextNode(
                    text=data["content"],
                    id_=data["id"],
                    metadata=data.get("metadata", {})
                )
                
                nodes.append(node)
            
            except Exception as e:
                logger.error(f"Error getting node {node_id}: {e}")
        
        return nodes
    
    def __del__(self):
        """Cleanup HTTP client."""
        try:
            import asyncio
            asyncio.run(self.client.aclose())
        except:
            pass
