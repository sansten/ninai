"""
Memory Consolidation Task

Celery task for consolidating related memories into summarized entries.
"""

from celery import Task
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import logging

from app.core.celery_app import celery_app
from app.core.database import async_session_maker
from app.services.memory_service import MemoryService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class MemoryConsolidationTask(Task):
    """Task for consolidating related memories."""
    
    name = "app.tasks.memory_consolidation.consolidate_memory_task"
    
    def run(self, memory_id: str, org_id: str, user_id: str):
        """
        Consolidate a memory with related memories.
        
        Steps:
        1. Find semantically similar memories
        2. Extract common themes and deduplicate
        3. Create consolidated memory entry
        4. Update references and maintain lineage
        """
        return asyncio.run(self._async_consolidate(memory_id, org_id, user_id))
    
    async def _async_consolidate(self, memory_id: str, org_id: str, user_id: str):
        """Async implementation of memory consolidation."""
        async with async_session_maker() as db:
            try:
                memory_service = MemoryService(db)
                embedding_service = EmbeddingService()
                
                # Get the source memory
                memory = await memory_service.get_memory(memory_id)
                if not memory:
                    logger.error(f"Memory {memory_id} not found for consolidation")
                    return {"status": "error", "error": "Memory not found"}
                
                # Find related memories using hybrid search
                from app.schemas.memory import MemorySearchRequest, SearchHnmsMode
                search_request = MemorySearchRequest(
                    query=memory.content,
                    hnms_mode=SearchHnmsMode.AUTO,
                    limit=10,
                    min_score=0.7,
                )
                
                related = await memory_service.search_memories(
                    search_request,
                    user_id=user_id,
                    org_id=org_id
                )
                
                # Filter out the source memory itself
                related_memories = [m for m in related.results if m.id != memory_id]
                
                if not related_memories:
                    logger.info(f"No related memories found for {memory_id}")
                    return {
                        "status": "completed",
                        "consolidated_count": 0,
                        "message": "No related memories to consolidate"
                    }
                
                # Generate consolidated summary
                # This is a simplified version - you can enhance with LLM summarization
                consolidated_content = f"# Consolidated Memory\n\n"
                consolidated_content += f"**Source Memory**: {memory.title}\n\n"
                consolidated_content += f"{memory.content}\n\n"
                consolidated_content += f"## Related Memories ({len(related_memories)})\n\n"
                
                for rm in related_memories[:5]:  # Limit to top 5
                    consolidated_content += f"### {rm.title}\n{rm.content[:200]}...\n\n"
                
                # Create consolidated memory
                from app.schemas.memory import MemoryCreate
                consolidated_memory_data = MemoryCreate(
                    title=f"Consolidated: {memory.title}",
                    content=consolidated_content,
                    tags=list(set(memory.tags + [tag for rm in related_memories for tag in (rm.tags or [])])),
                    scope=memory.scope,
                    metadata={
                        "consolidated_from": [memory_id] + [rm.id for rm in related_memories],
                        "consolidation_count": len(related_memories) + 1,
                        "source_memory_id": memory_id,
                    }
                )
                
                consolidated = await memory_service.create_memory(
                    consolidated_memory_data,
                    user_id=user_id,
                    org_id=org_id,
                )
                
                # Update source memory to reference consolidated version
                from app.schemas.memory import MemoryUpdate
                await memory_service.update_memory(
                    memory_id,
                    MemoryUpdate(
                        metadata={
                            **(memory.metadata or {}),
                            "consolidated_to": consolidated.id,
                            "consolidated_at": consolidated.created_at.isoformat(),
                        }
                    )
                )
                
                await db.commit()
                
                logger.info(
                    f"Consolidated memory {memory_id} with {len(related_memories)} related memories "
                    f"into {consolidated.id}"
                )
                
                return {
                    "status": "completed",
                    "consolidated_id": consolidated.id,
                    "source_id": memory_id,
                    "consolidated_count": len(related_memories) + 1,
                    "related_memory_ids": [rm.id for rm in related_memories],
                }
                
            except Exception as e:
                logger.error(f"Error consolidating memory {memory_id}: {e}", exc_info=True)
                await db.rollback()
                return {"status": "error", "error": str(e)}


# Register task
consolidate_memory_task = celery_app.register_task(MemoryConsolidationTask())
