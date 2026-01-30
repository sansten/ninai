"""
Celery tasks for graph relationship management.

Periodic tasks:
- Nightly: Populate relationships for all organizations
- Weekly: Recalculate similarity scores
- Daily: Cleanup stale/orphaned relationships
"""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from app.celery_app import celery_app
from app.core.config import settings
from app.services.graph_relationship_service import GraphRelationshipService
from app.db.session import AsyncSessionLocal
import redis

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=300,  # Retry after 5 minutes
    time_limit=3600,  # 1 hour timeout
    name="graph.populate_relationships"
)
def populate_graph_relationships(
    self,
    org_id: str = None,
    similarity_threshold: float = 0.75,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Celery task to populate graph relationships for organization(s).
    
    Can be run:
    - Periodically (nightly) for all organizations
    - On-demand for specific organization
    
    Args:
        org_id: Specific org to populate (or all if None)
        similarity_threshold: Minimum similarity to create relationship
        batch_size: Process memories in batches
        
    Returns:
        Task result dict
    """
    try:
        # This will be run in a sync context via run_async wrapper
        import asyncio
        result = asyncio.run(
            _populate_relationships_async(org_id, similarity_threshold, batch_size)
        )
        return result

    except Exception as exc:
        logger.error(f"Error in populate_graph_relationships task: {exc}", exc_info=True)
        
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=self.request.retries * 300)


async def _populate_relationships_async(
    org_id: str = None,
    similarity_threshold: float = 0.75,
    batch_size: int = 100
) -> Dict[str, Any]:
    """Async implementation of relationship population."""
    
    async with AsyncSessionLocal() as session:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        
        service = GraphRelationshipService(session, redis_client)
        
        if org_id:
            # Single organization
            result = await service.populate_relationships(
                org_id=org_id,
                similarity_threshold=similarity_threshold,
                batch_size=batch_size
            )
            return result
        else:
            # All organizations
            results = []
            
            # Get all organization IDs
            org_result = await session.execute(
                text("SELECT DISTINCT organization_id FROM memories WHERE deleted_at IS NULL")
            )
            org_ids = [row[0] for row in org_result]
            
            logger.info(f"Populating relationships for {len(org_ids)} organizations")
            
            for org in org_ids:
                try:
                    result = await service.populate_relationships(
                        org_id=str(org),
                        similarity_threshold=similarity_threshold,
                        batch_size=batch_size
                    )
                    results.append({
                        "org_id": str(org),
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Error populating relationships for org {org}: {e}")
                    results.append({
                        "org_id": str(org),
                        "error": str(e)
                    })
            
            return {
                "organizations_processed": len(results),
                "results": results
            }


@celery_app.task(
    bind=True,
    max_retries=2,
    default_retry_delay=600,
    time_limit=1800,
    name="graph.cleanup_orphaned_relationships"
)
def cleanup_orphaned_relationships(self) -> Dict[str, Any]:
    """
    Cleanup relationships where memories have been deleted.
    
    Run periodically (weekly) to maintain graph integrity.
    """
    try:
        import asyncio
        result = asyncio.run(_cleanup_orphaned_async())
        return result
    except Exception as exc:
        logger.error(f"Error in cleanup_orphaned_relationships task: {exc}")
        raise self.retry(exc=exc, countdown=600)


async def _cleanup_orphaned_async() -> Dict[str, Any]:
    """Async implementation of cleanup."""
    
    async with AsyncSessionLocal() as session:
        # Find relationships where source or target memory is deleted
        result = await session.execute(
            text("""
                DELETE FROM graph_relationships gr
                WHERE gr.from_memory_id NOT IN (
                    SELECT id FROM memories WHERE deleted_at IS NULL
                )
                OR gr.to_memory_id NOT IN (
                    SELECT id FROM memories WHERE deleted_at IS NULL
                )
            """)
        )
        
        deleted_count = result.rowcount
        await session.commit()
        
        logger.info(f"Cleanup: Deleted {deleted_count} orphaned relationships")
        
        return {
            "orphaned_relationships_deleted": deleted_count
        }


@celery_app.task(
    bind=True,
    name="graph.recalculate_similarities"
)
def recalculate_similarities(
    self,
    org_id: str = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Recalculate similarity scores for existing relationships.
    
    Useful for:
    - Tuning similarity threshold
    - Updating after embedding model changes
    - Improving relationship quality over time
    
    Args:
        org_id: Specific organization
        sample_size: Sample size for testing (0 = all)
    """
    try:
        import asyncio
        result = asyncio.run(
            _recalculate_similarities_async(org_id, sample_size)
        )
        return result
    except Exception as exc:
        logger.error(f"Error recalculating similarities: {exc}")
        raise self.retry(exc=exc, countdown=300)


async def _recalculate_similarities_async(
    org_id: str = None,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """Async implementation of similarity recalculation."""
    
    async with AsyncSessionLocal() as session:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        
        service = GraphRelationshipService(session, redis_client)
        
        # Implementation would:
        # 1. Get auto-created relationships
        # 2. Fetch embeddings for both memories
        # 3. Recalculate similarity
        # 4. Update similarity_score in DB
        
        logger.info("Recalculating similarity scores - not yet implemented")
        
        return {
            "status": "not_implemented",
            "org_id": org_id,
            "sample_size": sample_size
        }


# Schedule periodic tasks
def setup_graph_tasks():
    """Setup Celery beat schedule for graph tasks."""
    
    # Nightly relationship population at 2 AM UTC
    celery_app.conf.beat_schedule['populate_graph_relationships'] = {
        'task': 'graph.populate_relationships',
        'schedule': 86400.0,  # Every 24 hours
        'args': (None, 0.75, 100),
        'kwargs': {},
        'options': {
            'queue': 'default',
        }
    }
    
    # Weekly cleanup on Sundays at 3 AM UTC
    celery_app.conf.beat_schedule['cleanup_orphaned_relationships'] = {
        'task': 'graph.cleanup_orphaned_relationships',
        'schedule': 604800.0,  # Every 7 days
        'options': {
            'queue': 'default',
        }
    }
    
    logger.info("Graph tasks scheduled")
