"""Cognitive Diff & Change Subscription API (Feature 17)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import asc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.audit import AuditEvent
from app.models.contradiction import Contradiction
from app.models.goal import Goal
from app.models.memory import MemoryMetadata
from app.models.meta_agent import MetaConflictRegistry
from app.models.provenance_edge import ProvenanceEdge

router = APIRouter()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_since(since: datetime | None) -> datetime:
    if since is None:
        return _utcnow() - timedelta(days=30)
    if since.tzinfo is None:
        return since.replace(tzinfo=timezone.utc)
    return since


def _event_payload(event: AuditEvent) -> dict[str, Any]:
    return {
        "id": str(event.id),
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "resource_type": event.resource_type,
        "resource_id": str(event.resource_id) if event.resource_id else None,
        "success": bool(event.success),
        "details": dict(event.details or {}),
        "changes": dict(event.changes or {}) if event.changes else {},
    }


def _memory_payload(memory: MemoryMetadata) -> dict[str, Any]:
    return {
        "memory_id": str(memory.id),
        "title": memory.title,
        "content_preview": memory.content_preview,
        "tags": list(memory.tags or []),
        "updated_at": memory.updated_at,
        "created_at": memory.created_at,
        "is_active": bool(memory.is_active),
    }


def _matches_topic(*, topic: str, memory: MemoryMetadata) -> bool:
    needle = topic.strip().lower()
    if not needle:
        return True

    haystacks = [
        str(memory.title or ""),
        str(memory.content_preview or ""),
        " ".join(str(tag) for tag in (memory.tags or [])),
    ]
    return any(needle in value.lower() for value in haystacks if value)


@router.get("")
async def get_cognitive_topic_diff(
    topic: str = Query(..., min_length=1, description="Topic phrase to diff"),
    since: datetime | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """What changed about a topic since a given timestamp."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    since_ts = _normalize_since(since)

    mem_res = await db.execute(
        select(MemoryMetadata).where(MemoryMetadata.organization_id == tenant.org_id)
    )
    topic_memories = [m for m in mem_res.scalars().all() if _matches_topic(topic=topic, memory=m)]
    topic_memory_ids = {str(m.id) for m in topic_memories}

    event_res = await db.execute(
        select(AuditEvent)
        .where(
            AuditEvent.organization_id == tenant.org_id,
            AuditEvent.timestamp >= since_ts,
            AuditEvent.event_type.in_(
                [
                    "memory.create",
                    "memory.update",
                    "memory.delete",
                    "memory.created",
                    "memory.updated",
                    "memory.deleted",
                ]
            ),
        )
        .order_by(asc(AuditEvent.timestamp))
        .limit(limit)
    )
    all_events = list(event_res.scalars().all())

    needle = topic.strip().lower()
    matched_events: list[AuditEvent] = []
    added_memories: set[str] = set()
    updated_memories: set[str] = set()
    invalidated_memories: set[str] = set()

    for event in all_events:
        resource_id = str(event.resource_id) if event.resource_id else ""
        details_str = json.dumps(event.details or {}, default=str).lower()
        event_matches = (resource_id in topic_memory_ids) or (needle in details_str)
        if not event_matches:
            continue

        matched_events.append(event)

        if event.event_type in {"memory.create", "memory.created"} and resource_id:
            added_memories.add(resource_id)
        elif event.event_type in {"memory.update", "memory.updated"} and resource_id:
            updated_memories.add(resource_id)
        elif event.event_type in {"memory.delete", "memory.deleted"} and resource_id:
            invalidated_memories.add(resource_id)

    contradiction_res = await db.execute(
        select(Contradiction).where(
            Contradiction.organization_id == tenant.org_id,
            Contradiction.created_at >= since_ts,
        )
    )
    new_conflicts = [
        {
            "id": str(row.id),
            "type": "fact_contradiction",
            "severity": str(row.severity),
            "reason": row.reason,
            "created_at": row.created_at,
        }
        for row in contradiction_res.scalars().all()
    ]

    return {
        "topic": topic,
        "since": since_ts,
        "changed": bool(matched_events or added_memories or updated_memories or invalidated_memories or new_conflicts),
        "payload": {
            "added_memories": sorted(added_memories),
            "updated_memories": sorted(updated_memories),
            "invalidated_memories": sorted(invalidated_memories),
            "new_conflicts": new_conflicts,
        },
        "events": [_event_payload(event) for event in matched_events],
        "matched_memories": [_memory_payload(mem) for mem in topic_memories[:limit]],
    }


@router.get("/memory/{memory_id}")
async def get_memory_diff(
    memory_id: str,
    since: datetime | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """How a memory evolved since a given timestamp."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    since_ts = _normalize_since(since)

    event_res = await db.execute(
        select(AuditEvent)
        .where(
            AuditEvent.organization_id == tenant.org_id,
            AuditEvent.timestamp >= since_ts,
            AuditEvent.resource_id == memory_id,
            AuditEvent.event_type.in_(
                [
                    "memory.create",
                    "memory.update",
                    "memory.delete",
                    "memory.created",
                    "memory.updated",
                    "memory.deleted",
                ]
            ),
        )
        .order_by(asc(AuditEvent.timestamp))
        .limit(limit)
    )
    events = list(event_res.scalars().all())

    prov_res = await db.execute(
        select(ProvenanceEdge)
        .where(
            ProvenanceEdge.org_id == tenant.org_id,
            ProvenanceEdge.target_id == memory_id,
            ProvenanceEdge.created_at >= since_ts,
        )
        .order_by(asc(ProvenanceEdge.created_at))
        .limit(limit)
    )
    edges = list(prov_res.scalars().all())

    timeline = [
        {
            "kind": "audit",
            "at": event.timestamp,
            "event": _event_payload(event),
        }
        for event in events
    ]
    timeline.extend(
        [
            {
                "kind": "provenance",
                "at": edge.created_at,
                "edge": {
                    "id": str(edge.id),
                    "source_id": str(edge.source_id),
                    "target_id": str(edge.target_id),
                    "edge_type": str(edge.edge_type),
                    "agent_name": str(edge.agent_name),
                    "metadata": dict(edge.edge_metadata or {}),
                },
            }
            for edge in edges
        ]
    )
    timeline.sort(key=lambda item: item.get("at") or datetime.min.replace(tzinfo=timezone.utc))

    return {
        "memory_id": memory_id,
        "since": since_ts,
        "change_count": len(timeline),
        "timeline": timeline,
    }


@router.get("/goals")
async def get_goal_diff(
    since: datetime | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """Goal status changes since a given timestamp."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    since_ts = _normalize_since(since)

    stmt = (
        select(Goal)
        .where(
            Goal.organization_id == tenant.org_id,
            or_(Goal.updated_at >= since_ts, Goal.created_at >= since_ts),
        )
        .order_by(asc(Goal.updated_at))
        .limit(limit)
    )
    if status:
        stmt = stmt.where(Goal.status == status)

    res = await db.execute(stmt)
    goals = list(res.scalars().all())

    return {
        "since": since_ts,
        "count": len(goals),
        "changes": [
            {
                "goal_id": str(goal.id),
                "title": goal.title,
                "status": goal.status,
                "priority": int(goal.priority or 0),
                "updated_at": goal.updated_at,
                "created_at": goal.created_at,
                "completed_at": goal.completed_at,
            }
            for goal in goals
        ],
    }


@router.get("/conflicts")
async def get_conflict_diff(
    since: datetime | None = Query(default=None),
    include_meta: bool = Query(default=True),
    limit: int = Query(default=200, ge=1, le=1000),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
):
    """New and resolved conflicts since a given timestamp."""
    await set_tenant_context(db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level)
    since_ts = _normalize_since(since)

    contradiction_res = await db.execute(
        select(Contradiction)
        .where(
            Contradiction.organization_id == tenant.org_id,
            or_(
                Contradiction.created_at >= since_ts,
                Contradiction.resolved_at >= since_ts,
            ),
        )
        .order_by(asc(Contradiction.created_at))
        .limit(limit)
    )
    contradictions = list(contradiction_res.scalars().all())

    new_conflicts = [
        {
            "id": str(row.id),
            "conflict_type": "fact_contradiction",
            "severity": str(row.severity),
            "reason": row.reason,
            "created_at": row.created_at,
        }
        for row in contradictions
        if row.created_at
    ]
    resolved_conflicts = [
        {
            "id": str(row.id),
            "conflict_type": "fact_contradiction",
            "resolved_at": row.resolved_at,
            "severity": str(row.severity),
        }
        for row in contradictions
        if row.resolved_at
    ]

    meta_new: list[dict[str, Any]] = []
    meta_resolved: list[dict[str, Any]] = []
    if include_meta:
        meta_res = await db.execute(
            select(MetaConflictRegistry)
            .where(
                MetaConflictRegistry.organization_id == tenant.org_id,
                or_(
                    MetaConflictRegistry.created_at >= since_ts,
                    MetaConflictRegistry.resolved_at >= since_ts,
                ),
            )
            .order_by(asc(MetaConflictRegistry.created_at))
            .limit(limit)
        )
        meta_rows = list(meta_res.scalars().all())
        meta_new = [
            {
                "id": str(row.id),
                "conflict_type": row.conflict_type,
                "status": row.status,
                "resource_type": row.resource_type,
                "resource_id": row.resource_id,
                "created_at": row.created_at,
            }
            for row in meta_rows
            if row.created_at
        ]
        meta_resolved = [
            {
                "id": str(row.id),
                "conflict_type": row.conflict_type,
                "status": row.status,
                "resolved_at": row.resolved_at,
            }
            for row in meta_rows
            if row.resolved_at
        ]

    return {
        "since": since_ts,
        "new_conflicts": new_conflicts,
        "resolved_conflicts": resolved_conflicts,
        "new_meta_conflicts": meta_new,
        "resolved_meta_conflicts": meta_resolved,
    }