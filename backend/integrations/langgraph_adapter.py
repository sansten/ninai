"""
LangGraph Checkpoint Store Adapter

Integrates Ninai Memory OS with LangGraph's checkpoint/state persistence system.
Supports agent state snapshots, graph execution history, and resumable workflows.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import httpx

try:
    from langgraph.checkpoint import BaseCheckpointSaver, Checkpoint
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    BaseCheckpointSaver = object  # Fallback
    Checkpoint = Dict[str, Any]  # Fallback

logger = logging.getLogger(__name__)


class NinaiLangGraphCheckpointSaver(BaseCheckpointSaver if LANGGRAPH_AVAILABLE else object):
    """
    LangGraph checkpoint saver using Ninai Memory OS backend.
    
    Features:
    - Persistent graph state storage
    - Checkpoint versioning
    - Parent/child checkpoint relationships
    - Resumable workflows
    - Multi-tenant isolation
    
    Usage:
        ```python
        from integrations.langgraph_adapter import NinaiLangGraphCheckpointSaver
        from langgraph.graph import StateGraph
        
        checkpoint_saver = NinaiLangGraphCheckpointSaver(
            api_base_url="http://localhost:8080/api/v1",
            api_key="your-capability-token",
            graph_id="workflow-001"
        )
        
        graph = StateGraph(State)
        graph.add_node("step1", step1_func)
        graph.add_edge("step1", "step2")
        
        app = graph.compile(checkpointer=checkpoint_saver)
        
        # Execute with checkpointing
        result = app.invoke(
            {"input": "data"},
            config={"configurable": {"thread_id": "conversation-123"}}
        )
        ```
    """
    
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        graph_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Ninai LangGraph Checkpoint Saver.
        
        Args:
            api_base_url: Base URL for Ninai API (e.g., http://localhost:8080/api/v1)
            api_key: Capability token for authentication
            graph_id: Graph identifier for grouping checkpoints
            user_id: User identifier (optional)
            org_id: Organization identifier (optional)
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )
        
        super().__init__(**kwargs) if LANGGRAPH_AVAILABLE else None
        
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.graph_id = graph_id or f"graph_{datetime.utcnow().timestamp()}"
        self.user_id = user_id
        self.org_id = org_id
        
        self._client = httpx.Client(
            base_url=self.api_base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0
        )
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save a checkpoint to Ninai.
        
        Args:
            config: Checkpoint configuration (includes thread_id, etc.)
            checkpoint: Checkpoint data to save
            metadata: Additional metadata
            
        Returns:
            Saved checkpoint configuration
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            checkpoint_id = checkpoint.get("id", f"checkpoint_{datetime.utcnow().timestamp()}")
            
            # Serialize checkpoint data
            checkpoint_data = {
                "content": json.dumps({
                    "checkpoint": checkpoint,
                    "config": config,
                    "timestamp": datetime.utcnow().isoformat()
                }),
                "title": f"Checkpoint {checkpoint_id}",
                "tags": [
                    "langgraph",
                    f"graph:{self.graph_id}",
                    f"thread:{thread_id}",
                    f"checkpoint:{checkpoint_id}"
                ],
                "metadata": {
                    **(metadata or {}),
                    "framework": "langgraph",
                    "graph_id": self.graph_id,
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "parent_checkpoint_id": checkpoint.get("parent_config", {}).get("configurable", {}).get("checkpoint_id")
                },
                "scope": "session"  # Use session scope for workflow state
            }
            
            response = self._client.post("/memories", json=checkpoint_data)
            response.raise_for_status()
            
            # Return updated config with memory ID
            result_data = response.json()
            config["configurable"]["checkpoint_id"] = checkpoint_id
            config["configurable"]["memory_id"] = result_data.get("id")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to save LangGraph checkpoint: {e}")
            raise
    
    def get(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint from Ninai.
        
        Args:
            config: Checkpoint configuration (includes thread_id, checkpoint_id)
            
        Returns:
            Checkpoint data or None if not found
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            
            if not checkpoint_id:
                # Get latest checkpoint for thread
                return self.get_latest(config)
            
            # Search for specific checkpoint
            params = {
                "tags": f"langgraph,graph:{self.graph_id},thread:{thread_id},checkpoint:{checkpoint_id}",
                "limit": 1
            }
            
            response = self._client.get("/memories", params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("items"):
                return None
            
            # Deserialize checkpoint
            memory = data["items"][0]
            checkpoint_data = json.loads(memory.get("content", "{}"))
            
            return checkpoint_data.get("checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to retrieve LangGraph checkpoint: {e}")
            return None
    
    def get_latest(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for a thread.
        
        Args:
            config: Configuration with thread_id
            
        Returns:
            Latest checkpoint or None
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            params = {
                "tags": f"langgraph,graph:{self.graph_id},thread:{thread_id}",
                "limit": 1,
                "sort": "created_at:desc"  # Get most recent
            }
            
            response = self._client.get("/memories", params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("items"):
                return None
            
            # Deserialize checkpoint
            memory = data["items"][0]
            checkpoint_data = json.loads(memory.get("content", "{}"))
            
            # Update config with checkpoint_id
            checkpoint = checkpoint_data.get("checkpoint")
            if checkpoint:
                config["configurable"]["checkpoint_id"] = checkpoint.get("id")
                config["configurable"]["memory_id"] = memory.get("id")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to retrieve latest checkpoint: {e}")
            return None
    
    def list(
        self,
        config: Dict[str, Any],
        limit: int = 10,
        before: Optional[str] = None
    ) -> List[Tuple[Dict[str, Any], Checkpoint]]:
        """
        List checkpoints for a thread.
        
        Args:
            config: Configuration with thread_id
            limit: Max number of checkpoints to return
            before: Return checkpoints before this checkpoint_id
            
        Returns:
            List of (config, checkpoint) tuples
        """
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            params = {
                "tags": f"langgraph,graph:{self.graph_id},thread:{thread_id}",
                "limit": limit,
                "sort": "created_at:desc"
            }
            
            response = self._client.get("/memories", params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for memory in data.get("items", []):
                try:
                    checkpoint_data = json.loads(memory.get("content", "{}"))
                    checkpoint = checkpoint_data.get("checkpoint")
                    saved_config = checkpoint_data.get("config", {})
                    
                    # Update config with IDs
                    saved_config["configurable"]["checkpoint_id"] = checkpoint.get("id")
                    saved_config["configurable"]["memory_id"] = memory.get("id")
                    
                    results.append((saved_config, checkpoint))
                except Exception as e:
                    logger.warning(f"Failed to parse checkpoint: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def delete(self, config: Dict[str, Any]) -> None:
        """
        Delete a checkpoint (not typically needed).
        
        Args:
            config: Configuration with checkpoint_id
        """
        try:
            memory_id = config.get("configurable", {}).get("memory_id")
            
            if not memory_id:
                logger.warning("No memory_id in config, cannot delete checkpoint")
                return
            
            response = self._client.delete(f"/memories/{memory_id}")
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
    
    def close(self) -> None:
        """Close HTTP client connection."""
        self._client.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore cleanup errors


# Convenience functions for quick setup

def create_checkpoint_saver(
    api_base_url: str = "http://localhost:8080/api/v1",
    api_key: Optional[str] = None,
    graph_id: Optional[str] = None
) -> NinaiLangGraphCheckpointSaver:
    """
    Create a LangGraph checkpoint saver with sensible defaults.
    
    Args:
        api_base_url: Ninai API base URL
        api_key: Capability token (required)
        graph_id: Graph identifier
        
    Returns:
        Configured NinaiLangGraphCheckpointSaver instance
    """
    if not api_key:
        raise ValueError("api_key is required")
    
    return NinaiLangGraphCheckpointSaver(
        api_base_url=api_base_url,
        api_key=api_key,
        graph_id=graph_id
    )
