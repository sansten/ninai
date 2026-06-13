"""Async embedding and Qdrant indexing task.

Runs after a memory write returns 201. Decouples the OpenAI/vLLM round-trip
from the synchronous write path so POST /memories returns in <50 ms while the
vector index is populated within seconds in the background.
"""

from __future__ import annotations

from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.core.qdrant import QdrantService
from app.models.memory import MemoryMetadata
from app.services.embedding_service import EmbeddingService
from app.tasks.async_runtime import run_async

logger = get_task_logger(__name__)


@celery_app.task(
    name="embed_and_index_task",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def embed_and_index_task(self, *, memory_id: str, content: str, org_id: str):
    """Embed *content* with the configured provider and upsert into Qdrant."""

    async def _run():
        embedding = await EmbeddingService.embed(content)
        if not any(embedding):
            logger.warning("embed_and_index_task: zero embedding for memory_id=%s — skipping Qdrant upsert", memory_id)
            return

        vector_id = None
        payload = None
        async with async_session_factory() as session:
            memory = await session.get(MemoryMetadata, memory_id)
            if not memory:
                logger.warning("embed_and_index_task: memory_id=%s not found in DB", memory_id)
                return
            vector_id = memory.vector_id
            meta = memory.extra_metadata or {}
            payload = {
                "memory_id": memory_id,
                "scope": memory.scope,
                "scope_id": memory.scope_id,
                "team_id": memory.scope_id if str(memory.scope) == "team" else None,
                "owner_id": memory.owner_id,
                "tags": memory.tags or [],
                "classification": memory.classification,
                "memory_type": memory.memory_type,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "event_time": meta.get("event_time"),
                "write_actor_id": meta.get("write_actor_id", "anonymous"),
                "write_actor_type": meta.get("write_actor_type", "anonymous"),
                "write_role": meta.get("write_role", "anonymous"),
            }

        await QdrantService.upsert_memory(
            memory_id=vector_id,
            org_id=org_id,
            vector=embedding,
            payload=payload,
        )
        logger.debug("embed_and_index_task: indexed memory_id=%s org_id=%s", memory_id, org_id)

    try:
        run_async(_run())
    except Exception as exc:
        logger.warning("embed_and_index_task failed for memory_id=%s: %s — retrying", memory_id, exc)
        raise self.retry(exc=exc)


def enqueue_embed_and_index(*, memory_id: str, content: str, org_id: str):
    """Fire-and-forget enqueue. No-ops gracefully when Celery broker is unavailable."""
    try:
        from app.core.celery_app import celery_app as _app
        broker = _app.conf.broker_url
        if not broker or str(broker).startswith("memory://"):
            return
        embed_and_index_task.apply_async(
            kwargs={"memory_id": memory_id, "content": content, "org_id": org_id},
            queue="q.agent_enrich",
            retry=False,
        )
    except Exception as exc:
        logger.warning("enqueue_embed_and_index failed (non-fatal): %s", exc)
