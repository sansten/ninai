"""
ORM RLS Defense-in-Depth Loader Criteria
=========================================

Automatic query filtering at the ORM level for Row-Level Security.
This is a second layer of defense on top of PostgreSQL RLS policies.

Prevents accidental data leaks from:
- SQL injection bypassing Postgres RLS
- ORM bugs that skip Postgres RLS execution
- Manual SQL queries that misconfigure session vars

Usage:
  async with get_tenant_session(user_id, org_id) as session:
      attach_org_filter(session, org_id, user_id)
      # All subsequent ORM queries now auto-filter by org_id
"""

from typing import Optional, Dict, Any
from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy.orm import with_loader_criteria, Session
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)


@event.listens_for(Session, "do_orm_execute")
def _apply_tenant_loader_criteria(execute_state):
    """Apply per-session tenant loader criteria to all ORM SELECTs.

    attach_org_filter stores LoaderCriteriaOption objects in session.info.
    This hook ensures they are applied to every ORM statement executed
    through the session (defense-in-depth).
    """
    if not execute_state.is_select:
        return

    session = execute_state.session
    if not session.info.get("org_filter_active"):
        return

    criteria = session.info.get("org_filter_criteria") or {}
    if not criteria:
        return

    execute_state.statement = execute_state.statement.options(*criteria.values())


# Lazy-loaded to avoid circular imports
# Models are imported only when needed (in attach_org_filter)
_TENANT_MODELS: Optional[Dict[type, Column]] = None


def _get_tenant_models() -> Dict[type, Column]:
    """
    Lazy-load tenant models to avoid circular imports.
    
    This function is called on-demand to build the mapping of
    models to organization_id columns.
    """
    global _TENANT_MODELS
    
    if _TENANT_MODELS is not None:
        return _TENANT_MODELS
    
    # Import models here to avoid circular dependency
    from app.models.event import Event
    from app.models.webhook_subscription import WebhookSubscription
    from app.models.snapshot import Snapshot
    from app.models.memory import MemoryMetadata
    from app.models.memory_edge import MemoryEdge
    from app.models.memory_feedback import MemoryFeedback
    from app.models.memory_attachment import MemoryAttachment
    from app.models.knowledge_item import KnowledgeItem
    from app.models.knowledge_item_version import KnowledgeItemVersion
    from app.models.knowledge_review_request import KnowledgeReviewRequest
    from app.models.agent_run import AgentRun
    from app.models.audit import AuditEvent, MemoryAccessLog
    from app.models.knowledge import Knowledge
    
    _TENANT_MODELS = {
        Event: Event.__table__.c.organization_id,
        WebhookSubscription: WebhookSubscription.__table__.c.organization_id,
        Snapshot: Snapshot.__table__.c.organization_id,
        MemoryMetadata: MemoryMetadata.__table__.c.organization_id,
        MemoryEdge: MemoryEdge.__table__.c.organization_id,
        MemoryFeedback: MemoryFeedback.__table__.c.organization_id,
        MemoryAttachment: MemoryAttachment.__table__.c.organization_id,
        KnowledgeItem: KnowledgeItem.__table__.c.organization_id,
        KnowledgeItemVersion: KnowledgeItemVersion.__table__.c.organization_id,
        KnowledgeReviewRequest: KnowledgeReviewRequest.__table__.c.organization_id,
        AgentRun: AgentRun.__table__.c.organization_id,
        AuditEvent: AuditEvent.__table__.c.organization_id,
        MemoryAccessLog: MemoryAccessLog.__table__.c.organization_id,
        Knowledge: Knowledge.__table__.c.organization_id,
    }
    
    return _TENANT_MODELS


def attach_org_filter(session: AsyncSession, org_id: str, user_id: str) -> None:
    """
    Attach ORM-level criteria to filter by organization.
    
    Call this on the session after setting tenant context to ensure
    all ORM queries automatically filter to the current org.
    
    This is defense-in-depth: even if Postgres RLS is bypassed,
    the ORM layer will still enforce org isolation.
    
    Args:
        session: SQLAlchemy AsyncSession
        org_id: Organization ID to filter by (UUID string)
        user_id: User ID (for audit/logging purposes)
    
    Example:
        async with get_tenant_session(user_id, org_id) as session:
            attach_org_filter(session, org_id, user_id)
            result = await session.execute(select(Event))
            # Automatically filtered to user's org
    """
    if not org_id:
        # Security posture: empty org_id must never be permissive.
        # Use a valid UUID that is effectively unmatchable in tests.
        logger.warning(
            "attach_org_filter called with empty org_id for user %s; applying deny-all filter",
            user_id,
        )
        org_id = "00000000-0000-0000-0000-000000000000"
    
    # Get tenant models (lazy-loaded to avoid circular imports)
    TENANT_MODELS = _get_tenant_models()
    
    # Build loader criteria for each tenant model
    # with_loader_criteria automatically appends WHERE clause to all queries for that model
    loader_criteria = {}
    for model_class, org_column in TENANT_MODELS.items():
        try:
            loader_criteria[model_class] = with_loader_criteria(
                model_class,
                org_column == org_id,
                include_aliases=True,
            )
        except Exception as e:
            logger.error(f"Failed to attach criteria for {model_class.__name__}: {e}")
    
    # Store criteria in session info for access by query execution
    if loader_criteria:
        session.info['org_filter_criteria'] = loader_criteria
        session.info['org_filter_active'] = True
        session.info['org_id'] = org_id
        session.info['user_id'] = user_id
        logger.debug(f"Attached ORM filters for org {org_id} ({len(loader_criteria)} models)")


def get_org_filter_status(session: AsyncSession) -> Optional[Dict[str, Any]]:
    """
    Check if org filter is active and return filter metadata.
    
    Returns:
        Dict with org_id, user_id, and active status, or None if not set
    """
    if session.info.get('org_filter_active'):
        return {
            'org_id': session.info.get('org_id'),
            'user_id': session.info.get('user_id'),
            'active': True,
            'model_count': len(session.info.get('org_filter_criteria', {})),
        }
    return None


async def apply_org_filter(session: AsyncSession, org_id: str, user_id: str) -> None:
    """
    Apply org filter to session for all subsequent queries.
    
    This is the async wrapper that can be called in async contexts.
    
    Args:
        session: Database session
        org_id: Organization ID to filter by
        user_id: User ID for audit/logging
    """
    attach_org_filter(session, org_id, user_id)


@asynccontextmanager
async def org_filtered_session(session: AsyncSession, org_id: str, user_id: str):
    """
    Context manager that ensures org filtering is active.
    
    Args:
        session: Database session (typically from dependency injection)
        org_id: Organization ID to enforce
        user_id: Current user ID
    
    Example:
        async with org_filtered_session(db, org_id, user_id):
            result = await db.execute(select(Event))
            # Automatically filtered by org_id at ORM layer
    """
    attach_org_filter(session, org_id, user_id)
    try:
        yield session
    finally:
        # Cleanup: optionally clear filter info
        session.info.pop('org_filter_criteria', None)
        session.info.pop('org_filter_active', None)
        session.info.pop('org_id', None)
        session.info.pop('user_id', None)
