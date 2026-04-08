"""
Example 5: Batch Operations

Demonstrates efficient batch operations in Ninai2:
- Batch memory creation
- Bulk user operations
- Async task submission (Celery)
- Error handling and retries
"""

import asyncio
import aiohttp
from typing import List, Dict
import time


class BatchOperator:
    """Handle batch operations efficiently."""
    
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
    
    async def batch_create_memories(self, memories: List[Dict]) -> dict:
        """Create multiple memories in a single batch request."""
        payload = {"memories": memories}
        async with self.session.post(
            f"{self.base_url}/api/v1/memories/batch",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Batch create failed: {resp.status} {await resp.text()}")
            return await resp.json()
    
    async def batch_update_memories(self, updates: List[Dict]) -> dict:
        """Update multiple memories in a single batch request."""
        payload = {"updates": updates}
        async with self.session.put(
            f"{self.base_url}/api/v1/memories/batch",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Batch update failed: {resp.status}")
            return await resp.json()
    
    async def batch_delete_memories(self, memory_ids: List[str]) -> dict:
        """Delete multiple memories in a single batch request."""
        payload = {"ids": memory_ids}
        async with self.session.delete(
            f"{self.base_url}/api/v1/memories/batch",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Batch delete failed: {resp.status}")
            return await resp.json()
    
    async def submit_async_task(self, task_type: str, payload: dict) -> dict:
        """Submit an async task via Celery (returns task_id for polling)."""
        request_payload = {
            "task_type": task_type,
            "payload": payload
        }
        async with self.session.post(
            f"{self.base_url}/api/v1/tasks",
            json=request_payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Task submission failed: {resp.status}")
            return await resp.json()
    
    async def get_task_status(self, task_id: str) -> dict:
        """Poll task status."""
        async with self.session.get(
            f"{self.base_url}/api/v1/tasks/{task_id}",
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Status check failed: {resp.status}")
            return await resp.json()
    
    async def wait_for_task(self, task_id: str, timeout: int = 60, poll_interval: int = 2) -> dict:
        """Wait for an async task to complete with polling."""
        start = time.time()
        while time.time() - start < timeout:
            status = await self.get_task_status(task_id)
            if status["state"] in ["success", "failure"]:
                return status
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")


async def main():
    """Run batch operations example."""
    
    base_url = "http://localhost:8000"
    api_key = "test-api-key"
    
    print("Ninai2 Batch Operations Example")
    print("=" * 50)
    print()
    
    try:
        async with BatchOperator(base_url, api_key) as operator:
            
            # 1. Batch create memories
            print("1. Creating memories in batch...")
            memories_to_create = [
                {
                    "content": "Meeting with Product team on Q1 roadmap",
                    "memory_type": "observation",
                    "tags": ["meetings", "product", "roadmap"]
                },
                {
                    "content": "API rate limit should be increased to 1000/min",
                    "memory_type": "insight",
                    "tags": ["api", "performance", "improvement"]
                },
                {
                    "content": "Customer reported latency in vector search > 2s",
                    "memory_type": "observation",
                    "tags": ["customer", "performance", "urgent"]
                },
                {
                    "content": "Migration to Qdrant v1.8 improved search latency by 40%",
                    "memory_type": "fact",
                    "tags": ["qdrant", "performance", "verified"]
                },
                {
                    "content": "Implement request timeout of 30s for LLM calls",
                    "memory_type": "decision",
                    "tags": ["llm", "reliability", "decision"]
                }
            ]
            
            batch_result = await operator.batch_create_memories(memories_to_create)
            created_count = batch_result.get("created", 0)
            memory_ids = [m.get("id") for m in batch_result.get("memories", []) if m.get("id")]
            print(f"✓ Created {created_count} memories in batch")
            print(f"  Memory IDs: {memory_ids[:2]}...")
            print()
            
            # 2. Batch update memories
            print("2. Updating memories in batch...")
            updates = [
                {
                    "id": memory_ids[0],
                    "tags": ["meetings", "product", "roadmap", "urgent"]
                },
                {
                    "id": memory_ids[1],
                    "tags": ["api", "performance", "improvement", "backlog"]
                }
            ]
            
            update_result = await operator.batch_update_memories(updates)
            updated_count = update_result.get("updated", 0)
            print(f"✓ Updated {updated_count} memories (added priority tags)")
            print()
            
            # 3. Demonstrate async tasks
            print("3. Submitting async tasks...")
            
            # Long-running task (e.g., bulk export)
            task1 = await operator.submit_async_task(
                "export_org_data",
                {
                    "org_id": "current",
                    "format": "json",
                    "include_audit_logs": True
                }
            )
            task1_id = task1["task_id"]
            print(f"✓ Submitted export task: {task1_id}")
            print(f"  Status: {task1['status']}")
            
            # Another async task
            task2 = await operator.submit_async_task(
                "vectorize_memories",
                {
                    "org_id": "current",
                    "batch_size": 100
                }
            )
            task2_id = task2["task_id"]
            print(f"✓ Submitted vectorize task: {task2_id}")
            print()
            
            # 4. Poll task status
            print("4. Polling task status...")
            for i in range(3):
                status1 = await operator.get_task_status(task1_id)
                status2 = await operator.get_task_status(task2_id)
                print(f"  Poll {i+1}:")
                print(f"    Export task: {status1['state']} ({status1.get('progress', 'N/A')}%)")
                print(f"    Vectorize task: {status2['state']} ({status2.get('progress', 'N/A')}%)")
                if i < 2:
                    await asyncio.sleep(1)
            print()
            
            # 5. Batch delete (optional, for cleanup)
            print("5. Batch cleanup...")
            if len(memory_ids) > 2:
                delete_result = await operator.batch_delete_memories(memory_ids[2:])
                deleted_count = delete_result.get("deleted", 0)
                print(f"✓ Deleted {deleted_count} temporary test memories")
            print()
            
            print("✓ Batch operations example completed successfully!")
            print("\nBest Practices:")
            print("  - Use batch endpoints for 10+ items (more efficient)")
            print("  - Implement retry logic with exponential backoff")
            print("  - Poll async tasks with reasonable intervals (2-5s)")
            print("  - Set timeout on task polling (avoid infinite waits)")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify batch endpoints are enabled in backend")
        print("  2. Check Celery broker (Redis) is running")
        print("  3. Ensure API_KEY has batch operation permissions")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
