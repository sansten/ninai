"""
LangChain Memory Adapter

Integrates Ninai Memory OS with LangChain's BaseMemory interface.
Supports conversation history, entity memory, and vector-backed retrieval.
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

try:
    from langchain.schema import BaseMemory
    from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseMemory = object  # Fallback for type checking
    BaseMessage = object  # Fallback for type checking
    HumanMessage = object
    AIMessage = object

logger = logging.getLogger(__name__)


class NinaiLangChainMemory(BaseMemory):
    """
    LangChain memory adapter using Ninai Memory OS backend.
    
    Features:
    - Conversation history stored in Ninai
    - Entity extraction and tracking
    - Vector-backed semantic search
    - Organization and user scoping
    - Automatic memory promotion (short-term → long-term)
    
    Usage:
        ```python
        from integrations.langchain_adapter import NinaiLangChainMemory
        from langchain.chains import ConversationChain
        
        memory = NinaiLangChainMemory(
            api_base_url="http://localhost:8002/api/v1",
            api_key="your-capability-token",
            session_id="conversation-123"
        )
        
        chain = ConversationChain(llm=llm, memory=memory)
        response = chain.run("Hello!")
        ```
    """
    
    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        memory_key: str = "history",
        return_messages: bool = False,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        **kwargs
    ):
        """
        Initialize Ninai LangChain Memory.
        
        Args:
            api_base_url: Base URL for Ninai API (e.g., http://localhost:8002/api/v1)
            api_key: Capability token for authentication
            session_id: Session identifier for grouping memories
            user_id: User identifier (optional, from token if not provided)
            org_id: Organization identifier (optional, from token if not provided)
            memory_key: Key to use for memory in chain inputs
            return_messages: Whether to return Message objects or strings
            input_key: Key to use for input in chain (auto-detect if None)
            output_key: Key to use for output in chain (auto-detect if None)
            human_prefix: Prefix for human messages
            ai_prefix: Prefix for AI messages
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with: pip install langchain"
            )
        
        super().__init__(**kwargs)
        
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.session_id = session_id or f"langchain_{datetime.utcnow().timestamp()}"
        self.user_id = user_id
        self.org_id = org_id
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.input_key = input_key
        self.output_key = output_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        
        # Import here to avoid circular dependency
        import httpx
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    
    @property
    def memory_variables(self) -> List[str]:
        """Return memory variables (used by LangChain)."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load conversation history from Ninai Memory OS.
        
        Returns dict with memory_key → conversation history.
        """
        import asyncio
        
        try:
            # Run async function in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create task
                import nest_asyncio
                nest_asyncio.apply()
            
            messages = asyncio.run(self._load_messages())
            
            if self.return_messages:
                return {self.memory_key: messages}
            else:
                # Convert to string format
                buffer = ""
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        buffer += f"{self.human_prefix}: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        buffer += f"{self.ai_prefix}: {msg.content}\n"
                return {self.memory_key: buffer.strip()}
        
        except Exception as e:
            logger.error(f"Error loading memory from Ninai: {e}")
            return {self.memory_key: [] if self.return_messages else ""}
    
    async def _load_messages(self) -> List[BaseMessage]:
        """Load messages from Ninai API."""
        try:
            # Search for conversation history
            params = {
                "session_id": self.session_id,
                "scope": "private",
                "limit": 50,
                "sort": "created_at:asc"
            }
            
            response = await self.client.get(
                f"{self.api_base_url}/memories",
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            messages = []
            
            for memory in data.get("items", []):
                metadata = memory.get("metadata", {})
                role = metadata.get("role", "human")
                content = memory.get("content", "")
                
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))
            
            return messages
        
        except Exception as e:
            logger.error(f"Error fetching memories: {e}")
            return []
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """
        Save conversation turn to Ninai Memory OS.
        
        Stores both human input and AI output as separate memories.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            asyncio.run(self._save_context(inputs, outputs))
        
        except Exception as e:
            logger.error(f"Error saving memory to Ninai: {e}")
    
    async def _save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to Ninai API."""
        try:
            # Extract input and output
            input_key = self.input_key or list(inputs.keys())[0]
            output_key = self.output_key or list(outputs.keys())[0]
            
            human_message = inputs.get(input_key, "")
            ai_message = outputs.get(output_key, "")
            
            # Save human message
            await self._create_memory(
                content=str(human_message),
                role="human",
                title=f"Human: {str(human_message)[:50]}..."
            )
            
            # Save AI message
            await self._create_memory(
                content=str(ai_message),
                role="ai",
                title=f"AI: {str(ai_message)[:50]}..."
            )
        
        except Exception as e:
            logger.error(f"Error saving context: {e}")
    
    async def _create_memory(self, content: str, role: str, title: str) -> None:
        """Create a memory in Ninai."""
        try:
            payload = {
                "title": title,
                "content": content,
                "scope": "private",
                "session_id": self.session_id,
                "tags": ["langchain", "conversation", role],
                "metadata": {
                    "role": role,
                    "framework": "langchain",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            response = await self.client.post(
                f"{self.api_base_url}/memories",
                json=payload
            )
            response.raise_for_status()
        
        except Exception as e:
            logger.error(f"Error creating memory: {e}")
    
    def clear(self) -> None:
        """Clear conversation history (deletes memories for this session)."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            
            asyncio.run(self._clear())
        
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
    
    async def _clear(self) -> None:
        """Clear memories from Ninai API."""
        try:
            # Get all memories for this session
            response = await self.client.get(
                f"{self.api_base_url}/memories",
                params={"session_id": self.session_id, "limit": 1000}
            )
            response.raise_for_status()
            
            data = response.json()
            memory_ids = [m["id"] for m in data.get("items", [])]
            
            if memory_ids:
                # Batch delete
                await self.client.post(
                    f"{self.api_base_url}/memories/batch/delete",
                    json={"memory_ids": memory_ids}
                )
        
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
    
    async def asearch(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Semantic search across memories.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant memories
        """
        try:
            payload = {
                "query": query,
                "limit": k,
                "hnms_mode": "AUTO",
                "filters": {
                    "session_id": self.session_id
                }
            }
            
            response = await self.client.post(
                f"{self.api_base_url}/memories/search",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("results", [])
        
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def __del__(self):
        """Cleanup HTTP client."""
        try:
            import asyncio
            asyncio.run(self.client.aclose())
        except:
            pass
