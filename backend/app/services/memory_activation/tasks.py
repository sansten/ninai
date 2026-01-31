"""Celery tasks for Memory Activation background processing.

These tasks handle asynchronous updates to memory access metrics,
co-activation edges, and other long-running operations.

IMPORTANT: These tasks must run with tenant context set so Postgres RLS
policies can be enforced (and FORCE RLS won't break workers).
"""

import logging
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Optional
from uuid import UUID

from celery import shared_task
from sqlalchemy import delete, select, update, func

from app.core.config import get_settings
from app.core.database import async_session_factory, get_tenant_session
from app.models.memory import MemoryMetadata
from app.models.memory_activation import (
    MemoryActivationState,
    MemoryCoactivationEdge,
    CausalHypothesis,
    MemoryRetrievalExplanation,
)
from app.models.organization import Organization

logger = logging.getLogger(__name__)

def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    return asyncio.run(coro)


def _broker_enabled() -> bool:
    broker = get_settings().CELERY_BROKER_URL or ""
    return bool(broker) and not str(broker).startswith("memory://")


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def memory_access_update_task(
    self,
    memory_id: str,
    org_id: str,
    user_id: str,
    retrieval_explanation_id: Optional[str] = None,
) -> dict:
    """
    Update memory access metrics after retrieval.

    This task is enqueued asynchronously from the retrieval pipeline
    to avoid blocking search results. It performs:

    1. Increments access_count
    2. Updates last_accessed_at timestamp
    3. Writes audit event to memory_activation_state
    4. Optionally updates Qdrant payload with latest metrics

    Args:
        memory_id: UUID of retrieved memory
        org_id: Organization ID for authorization
        user_id: User ID for audit trail
        retrieval_explanation_id: Optional FK to retrieval_explanations for tracing

    Returns:
        dict: Status and updated metrics
        {
            "status": "success" | "failed",
            "memory_id": str,
            "access_count": int,
            "last_accessed_at": str (ISO format),
            "retrieval_explanation_id": str | None,
            "audit_event": dict,
        }

    Raises:
        Exception: Task retries on database errors (max 3 retries)
    """
    try:
        # Run async operation in event loop
        result = _run_async(
            _memory_access_update_async(
                memory_id=memory_id,
                org_id=org_id,
                user_id=user_id,
                retrieval_explanation_id=retrieval_explanation_id,
            )
        )
        return result

    except Exception as exc:
        logger.error(
            f"memory_access_update_task failed for memory {memory_id}",
            exc_info=exc,
            extra={
                "memory_id": memory_id,
                "org_id": org_id,
                "user_id": user_id,
                "retry_count": self.request.retries,
            },
        )

        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)

        return {
            "status": "failed",
            "memory_id": memory_id,
            "error": str(exc),
            "retry_count": self.request.retries,
        }


async def _memory_access_update_async(
    memory_id: str,
    org_id: str,
    user_id: str,
    retrieval_explanation_id: Optional[str] = None,
) -> dict:
    """
    Async implementation of memory access update.

    Steps:
    1. Load memory activation state (create if doesn't exist)
    2. Increment access_count
    3. Update last_accessed_at
    4. Persist to database
    5. Create audit event
    6. Return metrics

    Args:
        memory_id: UUID of retrieved memory
        org_id: Organization ID for authorization
        user_id: User ID for audit trail
        retrieval_explanation_id: Optional FK to retrieval_explanations

    Returns:
        dict: Status and updated metrics
    """
    async with get_tenant_session(
        user_id=user_id,
        org_id=org_id,
        roles="",
        clearance_level=0,
        justification="memory_access_update_task",
    ) as session:
        # Parse UUIDs
        mem_id = UUID(memory_id)
        user_uuid = UUID(user_id)

        # Verify memory exists and belongs to org (RLS + ORM criteria enforce org)
        memory_stmt = select(MemoryMetadata).where(
            MemoryMetadata.id == memory_id,
            MemoryMetadata.organization_id == org_id,
        )
        memory = (await session.execute(memory_stmt)).scalar_one_or_none()

        if not memory:
            logger.warning(
                f"Memory not found or unauthorized: {memory_id}",
                extra={"org_id": org_id, "user_id": user_id},
            )
            return {
                "status": "failed",
                "memory_id": memory_id,
                "error": "Memory not found or unauthorized",
            }

        # Load or create activation state
        state_stmt = select(MemoryActivationState).where(
            MemoryActivationState.memory_id == memory_id,
            MemoryActivationState.organization_id == org_id,
        )
        state = (await session.execute(state_stmt)).scalar_one_or_none()

        if not state:
            state = MemoryActivationState(
                memory_id=memory_id,
                organization_id=org_id,
                access_count=1,
                last_accessed_at=datetime.now(UTC),
                base_importance=0.5,
                confidence=0.8,
                contradicted=False,
                risk_factor=0.0,
            )
            session.add(state)
            logger.info(
                f"Created new activation state for memory {memory_id}",
                extra={"org_id": org_id},
            )
        else:
            state.access_count = (state.access_count or 0) + 1
            state.last_accessed_at = datetime.now(UTC)

        await session.flush()

        audit_event = {
            "event_type": "access_update",
            "memory_id": str(mem_id),
            "user_id": str(user_uuid),
            "timestamp": datetime.now(UTC).isoformat(),
            "access_count": state.access_count,
            "last_accessed_at": state.last_accessed_at.isoformat() if state.last_accessed_at else None,
            "retrieval_explanation_id": retrieval_explanation_id,
        }

        logger.info(
            "Updated memory access metrics",
            extra={
                "memory_id": memory_id,
                "access_count": state.access_count,
                "retrieval_explanation_id": retrieval_explanation_id,
            },
        )

        return {
            "status": "success",
            "memory_id": memory_id,
            "access_count": state.access_count,
            "last_accessed_at": state.last_accessed_at.isoformat() if state.last_accessed_at else None,
            "retrieval_explanation_id": retrieval_explanation_id,
            "audit_event": audit_event,
        }


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def coactivation_update_task(
    self,
    primary_memory_id: str,
    coactivated_memory_ids: list[str],
    org_id: str,
    time_window_hours: int = 24,
    top_n_pairs: int = 10,
) -> dict:
    """
    Update co-activation edges from a retrieval session.

    This task tracks which memories are frequently retrieved together,
    creating a dynamic graph of memory associations. It performs:

    1. Groups co-activated memory pairs
    2. Deduplicates within time window
    3. Computes edge weights: weight = 1 - exp(-λ * count)
    4. Caps top-N edges to prevent unbounded growth
    5. Writes edges to database

    Args:
        primary_memory_id: UUID of primary retrieved memory
        coactivated_memory_ids: UUIDs of other memories in same session
        org_id: Organization ID
        time_window_hours: Deduplication window (default 24h)
        top_n_pairs: Cap edges per memory (default 10)

    Returns:
        dict: Status and updated edges
        {
            "status": "success" | "failed",
            "primary_memory_id": str,
            "edges_created": int,
            "edges_updated": int,
            "edges_pruned": int,
        }

    Raises:
        Exception: Task retries on database errors (max 3 retries)
    """
    try:
        result = _run_async(
            _coactivation_update_async(
                primary_memory_id=primary_memory_id,
                coactivated_memory_ids=coactivated_memory_ids,
                org_id=org_id,
                time_window_hours=time_window_hours,
                top_n_pairs=top_n_pairs,
            )
        )
        return result

    except Exception as exc:
        logger.error(
            f"coactivation_update_task failed for memory {primary_memory_id}",
            exc_info=exc,
            extra={
                "primary_memory_id": primary_memory_id,
                "org_id": org_id,
                "retry_count": self.request.retries,
            },
        )

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=2 ** self.request.retries)

        return {
            "status": "failed",
            "primary_memory_id": primary_memory_id,
            "error": str(exc),
        }


async def _coactivation_update_async(
    primary_memory_id: str,
    coactivated_memory_ids: list[str],
    org_id: str,
    time_window_hours: int = 24,
    top_n_pairs: int = 10,
) -> dict:
    """
    Async implementation of co-activation edge updates.

    Algorithm:
    1. For each co-activated memory:
       a. Load or create edge
       b. Increment co_count within time window
       c. Compute weight: 1 - exp(-0.1 * co_count)  [λ=0.1]
       d. Update last_coactivated_at
    2. For each direction (A→B, B→A):
       a. Load top-N edges by weight
       b. Prune lower-weight edges if > top_n_pairs
    3. Persist all changes

    Args:
        primary_memory_id: UUID of primary memory
        coactivated_memory_ids: UUIDs of co-activated memories
        org_id: Organization ID
        time_window_hours: Deduplication window
        top_n_pairs: Max edges per memory

    Returns:
        dict: Status and edge counts
    """
    service_user_id = str(getattr(get_settings(), "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    async with get_tenant_session(
        user_id=service_user_id or "00000000-0000-0000-0000-000000000000",
        org_id=org_id,
        roles=service_roles,
        clearance_level=0,
        justification="coactivation_update_task",
    ) as session:
        # Parse UUIDs
        primary_id = UUID(primary_memory_id)
        org_uuid = UUID(org_id)
        coactivated_ids = [UUID(mid) for mid in coactivated_memory_ids]

        DECAY_LAMBDA = 0.1

        edges_created = 0
        edges_updated = 0
        edges_pruned = 0

        now = datetime.now(UTC)
        time_window = now - timedelta(hours=time_window_hours)

        for coactivated_id in coactivated_ids:
            if coactivated_id == primary_id:
                continue

            a_id, b_id = (
                (primary_id, coactivated_id)
                if str(primary_id) < str(coactivated_id)
                else (coactivated_id, primary_id)
            )

            edge_stmt = select(MemoryCoactivationEdge).where(
                MemoryCoactivationEdge.organization_id == org_uuid,
                MemoryCoactivationEdge.memory_id_a == a_id,
                MemoryCoactivationEdge.memory_id_b == b_id,
            )
            edge = (await session.execute(edge_stmt)).scalar_one_or_none()

            if not edge:
                edge = MemoryCoactivationEdge(
                    organization_id=org_uuid,
                    memory_id_a=a_id,
                    memory_id_b=b_id,
                    coactivation_count=1,
                    edge_weight=1.0 - __import__("math").exp(-DECAY_LAMBDA * 1),
                    last_coactivated_at=now,
                )
                session.add(edge)
                edges_created += 1
            else:
                if edge.last_coactivated_at and edge.last_coactivated_at >= time_window:
                    edge.coactivation_count = int(edge.coactivation_count or 0) + 1
                else:
                    edge.coactivation_count = 1

                import math

                edge.edge_weight = 1.0 - math.exp(-DECAY_LAMBDA * int(edge.coactivation_count or 0))
                edge.last_coactivated_at = now
                edges_updated += 1

        # Keep top-N by weight for this memory
        source_edges_stmt = (
            select(MemoryCoactivationEdge)
            .where(MemoryCoactivationEdge.organization_id == org_uuid)
            .where(
                (MemoryCoactivationEdge.memory_id_a == primary_id)
                | (MemoryCoactivationEdge.memory_id_b == primary_id)
            )
            .order_by(MemoryCoactivationEdge.edge_weight.desc())
        )
        source_edges = (await session.execute(source_edges_stmt)).scalars().all()

        if len(source_edges) > top_n_pairs:
            for edge_to_delete in source_edges[top_n_pairs:]:
                await session.delete(edge_to_delete)
            edges_pruned += max(0, len(source_edges) - top_n_pairs)

        await session.flush()

        logger.info(
            "Updated coactivation edges",
            extra={
                "primary_memory_id": primary_memory_id,
                "edges_created": edges_created,
                "edges_updated": edges_updated,
                "edges_pruned": edges_pruned,
            },
        )

        return {
            "status": "success",
            "primary_memory_id": primary_memory_id,
            "edges_created": edges_created,
            "edges_updated": edges_updated,
            "edges_pruned": edges_pruned,
        }


async def _nightly_decay_refresh_org_async(
    *,
    org_id: str,
    decay_lambda: float,
    prune_min_weight: float,
    prune_older_than_days: int,
) -> dict:
    now = datetime.now(UTC)

    service_user_id = str(getattr(get_settings(), "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    async with get_tenant_session(
        user_id=service_user_id or "00000000-0000-0000-0000-000000000000",
        org_id=org_id,
        roles=service_roles,
        clearance_level=0,
        justification="nightly_decay_refresh_task",
    ) as session:
        # 1) Sanitize activation_state values (defense-in-depth)
        activation_stmt = (
            update(MemoryActivationState)
            .where(MemoryActivationState.organization_id == org_id)
            .values(
                base_importance=func.least(1.0, func.greatest(0.0, MemoryActivationState.base_importance)),
                confidence=func.least(1.0, func.greatest(0.0, MemoryActivationState.confidence)),
                risk_factor=func.least(1.0, func.greatest(0.0, MemoryActivationState.risk_factor)),
                access_count=func.greatest(0, MemoryActivationState.access_count),
                updated_at=func.now(),
            )
        )
        activation_res = await session.execute(activation_stmt)

        # 2) Recompute coactivation edge weights from counts
        edge_stmt = (
            update(MemoryCoactivationEdge)
            .where(MemoryCoactivationEdge.organization_id == org_id)
            .values(
                edge_weight=1.0 - func.exp((-1.0 * decay_lambda) * MemoryCoactivationEdge.coactivation_count)
            )
        )
        edge_res = await session.execute(edge_stmt)

        # 3) Prune very old + low-weight edges (best-effort)
        cutoff = now - timedelta(days=max(1, int(prune_older_than_days)))
        prune_stmt = (
            delete(MemoryCoactivationEdge)
            .where(MemoryCoactivationEdge.organization_id == org_id)
            .where(MemoryCoactivationEdge.edge_weight < float(prune_min_weight))
            .where(MemoryCoactivationEdge.last_coactivated_at.is_not(None))
            .where(MemoryCoactivationEdge.last_coactivated_at < cutoff)
        )
        prune_res = await session.execute(prune_stmt)

        return {
            "ok": True,
            "org_id": org_id,
            "activation_rows_sanitized": int(getattr(activation_res, "rowcount", 0) or 0),
            "edges_reweighted": int(getattr(edge_res, "rowcount", 0) or 0),
            "edges_pruned": int(getattr(prune_res, "rowcount", 0) or 0),
            "ran_at": now.isoformat(),
        }


@shared_task(bind=True, name="app.services.memory_activation.tasks.nightly_decay_refresh_task")
def nightly_decay_refresh_task(
    self,
    org_id: Optional[str] = None,
    prune_min_weight: float = 0.01,
    prune_older_than_days: int = 90,
) -> dict:
    """Nightly maintenance for memory activation data.

    - Reweights coactivation edges from counts.
    - Clamps/sanitizes activation state fields to expected ranges.
    - Prunes extremely stale + low-weight edges.

    If org_id is not provided, runs for all active orgs.
    """

    if not _broker_enabled():
        return {"ok": True, "skipped": True, "reason": "broker_disabled"}

    decay_lambda = 0.1  # keep consistent with coactivation_update_task

    async def _run() -> dict:
        if org_id:
            return await _nightly_decay_refresh_org_async(
                org_id=org_id,
                decay_lambda=decay_lambda,
                prune_min_weight=prune_min_weight,
                prune_older_than_days=prune_older_than_days,
            )

        async with async_session_factory() as session:
            org_ids = (
                await session.execute(select(Organization.id).where(Organization.is_active.is_(True)))
            ).scalars().all()

        results = []
        for oid in org_ids:
            results.append(
                await _nightly_decay_refresh_org_async(
                    org_id=str(oid),
                    decay_lambda=decay_lambda,
                    prune_min_weight=prune_min_weight,
                    prune_older_than_days=prune_older_than_days,
                )
            )

        return {
            "ok": True,
            "orgs_processed": len(org_ids),
            "results": results,
            "ran_at": datetime.now(UTC).isoformat(),
        }

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("nightly_decay_refresh_task failed", exc_info=exc, extra={"org_id": org_id})
        raise


async def _causal_hypothesis_refresh_org_async(
    *,
    org_id: str,
    min_edge_weight: float,
    limit: int,
) -> dict:
    """Derive simple 'correlates' hypotheses from coactivation edges.

    This is intentionally conservative and explainable:
    - Each hypothesis corresponds to one coactivation pair (A,B)
    - confidence is derived from edge_weight
    - evidence_memory_ids stores [A, B] (sorted)

    Future work can extend this to use events/episodes and richer causal inference.
    """

    now = datetime.now(UTC)

    service_user_id = str(getattr(get_settings(), "SYSTEM_TASK_USER_ID", None) or "")
    service_roles = "system_admin" if service_user_id else ""

    created = 0
    updated = 0

    async with get_tenant_session(
        user_id=service_user_id or "00000000-0000-0000-0000-000000000000",
        org_id=org_id,
        roles=service_roles,
        clearance_level=0,
        justification="causal_hypothesis_update_task",
    ) as session:
        edge_stmt = (
            select(MemoryCoactivationEdge)
            .where(MemoryCoactivationEdge.organization_id == org_id)
            .where(MemoryCoactivationEdge.edge_weight >= float(min_edge_weight))
            .order_by(MemoryCoactivationEdge.edge_weight.desc())
            .limit(int(limit))
        )

        edges = (await session.execute(edge_stmt)).scalars().all()

        for edge in edges:
            evidence = sorted([str(edge.memory_id_a), str(edge.memory_id_b)])
            confidence = float(edge.edge_weight or 0.0)

            existing_stmt = select(CausalHypothesis).where(
                CausalHypothesis.organization_id == org_id,
                CausalHypothesis.relation == "correlates",
                CausalHypothesis.evidence_memory_ids == evidence,
            )
            existing = (await session.execute(existing_stmt)).scalar_one_or_none()

            if existing is None:
                session.add(
                    CausalHypothesis(
                        organization_id=org_id,
                        episode_id=None,
                        from_event_id=None,
                        to_event_id=None,
                        relation="correlates",
                        confidence=confidence,
                        evidence_memory_ids=evidence,
                        status="proposed",
                    )
                )
                created += 1
            else:
                # Keep maximum confidence observed from edges; do not flip rejected.
                if str(existing.status) != "rejected":
                    existing.confidence = max(float(existing.confidence or 0.0), confidence)
                    if str(existing.status) == "contested":
                        existing.status = "proposed"
                    updated += 1

        await session.flush()

    return {
        "ok": True,
        "org_id": org_id,
        "created": created,
        "updated": updated,
        "edges_considered": len(edges),
        "ran_at": now.isoformat(),
    }


@shared_task(bind=True, name="app.services.memory_activation.tasks.causal_hypothesis_update_task")
def causal_hypothesis_update_task(
    self,
    org_id: str,
    min_edge_weight: float = 0.25,
    limit: int = 200,
) -> dict:
    """Refresh causal hypotheses for an org.

    Current implementation derives "correlates" hypotheses from strong coactivation edges.
    """

    try:
        return _run_async(
            _causal_hypothesis_refresh_org_async(
                org_id=org_id,
                min_edge_weight=min_edge_weight,
                limit=limit,
            )
        )
    except Exception as exc:
        logger.error(
            "causal_hypothesis_update_task failed",
            exc_info=exc,
            extra={"org_id": org_id, "min_edge_weight": min_edge_weight, "limit": limit},
        )
        raise


