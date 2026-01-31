"""
CrewAI Memory Adapter

Integrates Ninai Memory OS with CrewAI's memory system.
Supports agent memory, task context, and crew-level knowledge sharing.
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import httpx

try:
    from crewai.memory import Memory
    CREWAI_AVAILABLE = True
    _CREWAI_IMPORT_ERROR: Exception | None = None
except Exception as e:
    CREWAI_AVAILABLE = False
    Memory = object  # Fallback for type checking
    _CREWAI_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


class NinaiCrewAIMemory(Memory if CREWAI_AVAILABLE else object):
    """
    CrewAI memory adapter using Ninai Memory OS backend.
    
    Features:
    - Agent-level memory (scoped to individual agents)
    - Task context memory (scoped to tasks)
    - Crew-level shared knowledge
    - Vector-backed semantic search
    - Automatic memory consolidation
    
    Usage:
        ```python
        from integrations.crewai_adapter import NinaiCrewAIMemory
        from crewai import Agent, Task, Crew
        
        memory = NinaiCrewAIMemory(
            api_base_url="http://localhost:8080/api/v1",
            api_key="your-capability-token",
            crew_id="research-crew-001"
        )
        
        agent = Agent(
            role="Researcher",
            goal="Find information",
            backstory="Expert researcher",
            memory=memory
        )
        
        crew = Crew(agents=[agent], tasks=[task], memory=memory)
        result = crew.kickoff()
        ```
    """
    
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        crew_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        enable_long_term: bool = True,
        enable_short_term: bool = True,
        enable_entity_memory: bool = True,
        **kwargs
    ):
        """
        Initialize Ninai CrewAI Memory.
        
        Args:
            api_base_url: Base URL for Ninai API (e.g., http://localhost:8080/api/v1)
            api_key: Capability token for authentication
            crew_id: Crew identifier for grouping memories
            user_id: User identifier (optional)
            org_id: Organization identifier (optional)
            enable_long_term: Enable long-term memory storage
            enable_short_term: Enable short-term (working) memory
            enable_entity_memory: Enable entity tracking
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "crewai integration unavailable (optional dependency missing or incompatible)."
            ) from _CREWAI_IMPORT_ERROR
        
        super().__init__(**kwargs) if CREWAI_AVAILABLE else None
        
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.crew_id = crew_id or f"crew_{datetime.utcnow().timestamp()}"
        self.user_id = user_id
        self.org_id = org_id
        self.enable_long_term = enable_long_term
        self.enable_short_term = enable_short_term
        self.enable_entity_memory = enable_entity_memory
        
        self._client = httpx.AsyncClient(
            base_url=self.api_base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0
        )
    
    def save(self, agent_name: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save memory for an agent (synchronous wrapper).
        
        Args:
            agent_name: Name of the agent
            content: Memory content to save
            metadata: Additional metadata
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        loop.run_until_complete(self.save_async(agent_name, content, metadata))
    
    async def save_async(
        self,
        agent_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Save memory for an agent (async).
        
        Args:
            agent_name: Name of the agent
            content: Memory content to save
            metadata: Additional metadata
            
        Returns:
            Created memory data
        """
        try:
            memory_data = {
                "content": content,
                "tags": [
                    "crewai",
                    f"crew:{self.crew_id}",
                    f"agent:{agent_name}"
                ],
                "metadata": {
                    **(metadata or {}),
                    "framework": "crewai",
                    "crew_id": self.crew_id,
                    "agent_name": agent_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Use short-term or long-term memory endpoint
            if self.enable_short_term:
                endpoint = "/memory/short-term"
            else:
                endpoint = "/memories"
            
            response = await self._client.post(endpoint, json=memory_data)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to save CrewAI memory: {e}")
            raise
    
    def search(
        self,
        query: str,
        agent_name: Optional[str] = None,
        limit: int = 10
    ) -> List[str]:
        """
        Search memories (synchronous wrapper).
        
        Args:
            query: Search query
            agent_name: Filter by agent name (optional)
            limit: Max results to return
            
        Returns:
            List of matching memory contents
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.search_async(query, agent_name, limit)
        )
    
    async def search_async(
        self,
        query: str,
        agent_name: Optional[str] = None,
        limit: int = 10
    ) -> List[str]:
        """
        Search memories using vector similarity (async).
        
        Args:
            query: Search query
            agent_name: Filter by agent name (optional)
            limit: Max results to return
            
        Returns:
            List of matching memory contents
        """
        try:
            params = {
                "q": query,
                "limit": limit,
                "tags": f"crew:{self.crew_id}"
            }
            
            if agent_name:
                params["tags"] = f"{params['tags']},agent:{agent_name}"
            
            response = await self._client.get("/memories/search", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract content from results
            return [item.get("content", "") for item in data.get("results", [])]
            
        except Exception as e:
            logger.error(f"Failed to search CrewAI memories: {e}")
            return []
    
    def get_context(self, agent_name: str, task_description: str) -> str:
        """
        Get relevant context for an agent's task (synchronous wrapper).
        
        Args:
            agent_name: Name of the agent
            task_description: Description of the task
            
        Returns:
            Relevant context as a string
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.get_context_async(agent_name, task_description)
        )
    
    async def get_context_async(
        self,
        agent_name: str,
        task_description: str
    ) -> str:
        """
        Get relevant context for an agent's task (async).
        
        Args:
            agent_name: Name of the agent
            task_description: Description of the task
            
        Returns:
            Relevant context as a string
        """
        try:
            # Search for relevant memories
            memories = await self.search_async(
                query=task_description,
                agent_name=agent_name,
                limit=5
            )
            
            if not memories:
                return "No relevant context found."
            
            # Format as context
            context_parts = [f"Relevant memory {i+1}: {mem}" for i, mem in enumerate(memories)]
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return "Error retrieving context."
    
    def reset(self, agent_name: Optional[str] = None) -> None:
        """
        Reset memory for an agent or entire crew.
        
        Args:
            agent_name: Specific agent to reset, or None for entire crew
        """
        logger.warning(f"Memory reset not implemented for CrewAI adapter")
        # Note: Ninai doesn't support bulk delete by tag yet
        # This would require DELETE /memories?tags=crew:{crew_id}
    
    async def close(self) -> None:
        """Close HTTP client connection."""
        await self._client.aclose()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())
        except Exception:
            pass  # Ignore cleanup errors


# Convenience functions for quick setup

def create_crew_memory(
    api_base_url: str = "http://localhost:8080/api/v1",
    api_key: Optional[str] = None,
    crew_id: Optional[str] = None
) -> NinaiCrewAIMemory:
    """
    Create a CrewAI memory instance with sensible defaults.
    
    Args:
        api_base_url: Ninai API base URL
        api_key: Capability token (required)
        crew_id: Crew identifier
        
    Returns:
        Configured NinaiCrewAIMemory instance
    """
    if not api_key:
        raise ValueError("api_key is required")
    
    return NinaiCrewAIMemory(
        api_base_url=api_base_url,
        api_key=api_key,
        crew_id=crew_id
    )
