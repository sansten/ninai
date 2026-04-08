"""
Example 3: Memory Workflow

Demonstrates Ninai2 memory management:
- Creating and retrieving memories
- Memory tagging and semantic search
- Working memory and attention
- Memory lifecycle (creation, update, deletion)
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import List, Dict


class MemoryManager:
    """Manage agent memories in Ninai2."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}
    
    async def create_memory(self, content: str, memory_type: str = "observation", tags: List[str] = None) -> dict:
        """Create a new memory."""
        payload = {
            "content": content,
            "memory_type": memory_type,  # observation, fact, insight, decision, error
            "tags": tags or [],
            "metadata": {
                "source": "example-script",
                "importance": "normal"
            }
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/memories",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Failed to create memory: {resp.status} {await resp.text()}")
            return await resp.json()
    
    async def search_memories(self, query: str, limit: int = 10) -> List[dict]:
        """Search memories using semantic similarity."""
        payload = {"query": query, "limit": limit}
        
        async with self.session.post(
            f"{self.base_url}/api/v1/memories/search",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Search failed: {resp.status} {await resp.text()}")
            data = await resp.json()
            return data.get("results", [])
    
    async def get_memory(self, memory_id: str) -> dict:
        """Retrieve a specific memory."""
        async with self.session.get(
            f"{self.base_url}/api/v1/memories/{memory_id}",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Retrieval failed: {resp.status}")
            return await resp.json()
    
    async def update_memory(self, memory_id: str, content: str = None, tags: List[str] = None) -> dict:
        """Update memory content or tags."""
        payload = {}
        if content:
            payload["content"] = content
        if tags:
            payload["tags"] = tags
        
        async with self.session.put(
            f"{self.base_url}/api/v1/memories/{memory_id}",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Update failed: {resp.status}")
            return await resp.json()
    
    async def list_memories(self, tags: List[str] = None, limit: int = 100) -> List[dict]:
        """List memories, optionally filtered by tags."""
        params = {"limit": limit}
        if tags:
            params["tags"] = ",".join(tags)
        
        async with self.session.get(
            f"{self.base_url}/api/v1/memories",
            params=params,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"List failed: {resp.status}")
            data = await resp.json()
            return data.get("memories", [])
    
    async def delete_memory(self, memory_id: str) -> None:
        """Delete a memory."""
        async with self.session.delete(
            f"{self.base_url}/api/v1/memories/{memory_id}",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Deletion failed: {resp.status}")


async def main():
    """Run memory workflow example."""
    
    base_url = "http://localhost:8000"
    api_key = "test-api-key"  # From /api-keys endpoint
    
    print("Ninai2 Memory Workflow Example")
    print("=" * 50)
    print()
    
    try:
        async with MemoryManager(base_url, api_key) as memory_mgr:
            
            # 1. Create diverse memories
            print("1. Creating memories...")
            memories = []
            
            mem1 = await memory_mgr.create_memory(
                "User prefers detailed responses with code examples",
                memory_type="fact",
                tags=["user_preference", "style"]
            )
            memories.append(mem1)
            print(f"   Fact: {mem1['id']}")
            
            mem2 = await memory_mgr.create_memory(
                "Attempted REST API integration; returned 429 rate limit",
                memory_type="observation",
                tags=["error", "api", "debugging"]
            )
            memories.append(mem2)
            print(f"   Observation: {mem2['id']}")
            
            mem3 = await memory_mgr.create_memory(
                "Implementing vector search significantly improved relevance",
                memory_type="insight",
                tags=["vector_search", "improvement", "architecture"]
            )
            memories.append(mem3)
            print(f"   Insight: {mem3['id']}")
            
            mem4 = await memory_mgr.create_memory(
                "Decision: Use async/await pattern for all I/O operations",
                memory_type="decision",
                tags=["architecture", "async", "best_practices"]
            )
            memories.append(mem4)
            print(f"   Decision: {mem4['id']}")
            print()
            
            # 2. Semantic search
            print("2. Searching memories...")
            search_results = await memory_mgr.search_memories("API rate limiting and error handling")
            print(f"   Found {len(search_results)} relevant memories:")
            for i, result in enumerate(search_results[:3], 1):
                relevance = result.get("relevance_score", 0)
                print(f"   {i}. [{relevance:.2f}] {result['content'][:60]}...")
            print()
            
            # 3. Retrieve by tag
            print("3. Filtering by tags...")
            arch_memories = await memory_mgr.list_memories(tags=["architecture"])
            print(f"   Found {len(arch_memories)} architecture-related memories:")
            for mem in arch_memories:
                print(f"   - {mem['memory_type']}: {mem['content'][:50]}...")
            print()
            
            # 4. Update memory
            print("4. Updating memory...")
            updated = await memory_mgr.update_memory(
                mem4["id"],
                tags=["architecture", "async", "best_practices", "critical"]
            )
            print(f"   Added 'critical' tag to decision memory")
            print(f"   New tags: {updated['tags']}")
            print()
            
            # 5. Show memory statistics
            print("5. Memory Statistics:")
            all_memories = await memory_mgr.list_memories()
            types_count = {}
            for mem in all_memories:
                mem_type = mem.get("memory_type", "unknown")
                types_count[mem_type] = types_count.get(mem_type, 0) + 1
            
            for mem_type, count in sorted(types_count.items()):
                print(f"   {mem_type}: {count}")
            print(f"   Total: {len(all_memories)}")
            print()
            
            # 6. Memory lifecycle
            print("6. Memory Lifecycle Demo:")
            lifecycle_mem = await memory_mgr.create_memory(
                "Temporary memory for demonstration",
                memory_type="observation",
                tags=["demo", "temporary"]
            )
            print(f"   Created: {lifecycle_mem['id']}")
            
            # Retrieve
            retrieved = await memory_mgr.get_memory(lifecycle_mem["id"])
            print(f"   Retrieved: created {retrieved['created_at']}")
            
            # Delete
            await memory_mgr.delete_memory(lifecycle_mem["id"])
            print(f"   Deleted")
            print()
            
            print("✓ Memory workflow example completed successfully!")
            print("\nKey Takeaways:")
            print("  - Memories are semantically indexed for fast retrieval")
            print("  - Tags enable filtering and organization")
            print("  - Memory types (fact, insight, decision) aid reasoning")
            print("  - Memories persist across sessions (conversation history)")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is the backend running?")
        print("  2. Is Qdrant vector DB running? (required for semantic search)")
        print("  3. Check API_KEY is valid")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
