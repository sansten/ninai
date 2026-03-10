"""Tenant-Scoped Knowledge Graph API — Phase 30.

Exposes WorldModelAgent graph data via queryable REST endpoints.
All queries are RLS-scoped to the requesting tenant.

Endpoints
---------
GET /graph/neighbors
    Returns entity neighbors — all entities co-occurring with a given entity
    across the tenant's enriched memories, with edge weights and metadata.

GET /graph/entity-path
    BFS path between two entity names through the co-occurrence edge graph.

GET /graph/changes
    State-change events detected by WorldModelAgent (domain expansions,
    high-confidence nodes), optionally filtered to a single entity.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, set_tenant_context
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.models.agent_run import AgentRun

router = APIRouter()

_WORLD_MODEL_AGENT = "WorldModelAgent"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class NeighborItem(BaseModel):
    entity: str
    entity_type: str
    domains: list[str] = Field(default_factory=list)
    edge_count: int
    max_weight: float


class NeighborsResponse(BaseModel):
    entity: str
    neighbors: list[NeighborItem]
    total: int
    retrieved_at: datetime


class EntityPathResponse(BaseModel):
    from_entity: str
    to_entity: str
    path: list[str]
    hop_count: int
    found: bool
    retrieved_at: datetime


class ChangeItem(BaseModel):
    entity: str
    change_type: str
    description: str
    memory_id: str


class ChangesResponse(BaseModel):
    entity: Optional[str]
    changes: list[ChangeItem]
    total: int
    retrieved_at: datetime


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


async def _load_world_model_runs(
    db: AsyncSession,
    org_id: str,
    limit: int = 2000,
) -> list[AgentRun]:
    stmt = (
        select(AgentRun)
        .where(
            AgentRun.organization_id == org_id,
            AgentRun.agent_name == _WORLD_MODEL_AGENT,
            AgentRun.status == "success",
        )
        .order_by(AgentRun.finished_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


def _collect_nodes(runs: list[AgentRun]) -> dict[str, dict[str, Any]]:
    """Merge world_nodes across all runs into a dict keyed by canonical entity name."""
    nodes: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run.outputs, dict):
            continue
        for node in run.outputs.get("world_nodes") or []:
            name = node.get("entity", "")
            if not name:
                continue
            if name not in nodes:
                nodes[name] = {
                    "entity": name,
                    "entity_type": node.get("entity_type", "concept"),
                    "domains": list(node.get("domains") or []),
                    "node_confidence": float(node.get("node_confidence", 0.5)),
                }
            else:
                # merge domains
                existing_domains = set(nodes[name]["domains"])
                existing_domains.update(node.get("domains") or [])
                nodes[name]["domains"] = list(existing_domains)
                nodes[name]["node_confidence"] = max(
                    nodes[name]["node_confidence"],
                    float(node.get("node_confidence", 0.5)),
                )
    return nodes


def _collect_edges(runs: list[AgentRun]) -> list[dict[str, Any]]:
    """Collect all entity_edges across runs."""
    edges: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run.outputs, dict):
            continue
        for edge in run.outputs.get("entity_edges") or []:
            if edge.get("from_entity") and edge.get("to_entity"):
                edges.append(edge)
    return edges


def _collect_changes(runs: list[AgentRun]) -> list[dict[str, Any]]:
    """Collect state_changes across all runs, including the source memory_id."""
    changes: list[dict[str, Any]] = []
    for run in runs:
        if not isinstance(run.outputs, dict):
            continue
        for change in run.outputs.get("state_changes") or []:
            if change.get("entity") and change.get("change_type"):
                changes.append({**change, "memory_id": run.memory_id})
    return changes


def _build_adjacency(
    edges: list[dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    """Build undirected adjacency list from entity edges."""
    adj: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        a = edge["from_entity"]
        b = edge["to_entity"]
        w = float(edge.get("weight", 0.5))
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return adj


def _bfs_path(
    adj: dict[str, list[tuple[str, float]]],
    start: str,
    end: str,
    max_hops: int = 6,
) -> list[str]:
    """BFS over entity adjacency; returns shortest hop path or empty list."""
    if start == end:
        return [start]
    visited = {start}
    queue: deque[list[str]] = deque([[start]])
    while queue:
        path = queue.popleft()
        if len(path) > max_hops + 1:
            break
        node = path[-1]
        for neighbor, _ in adj.get(node, []):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/neighbors",
    response_model=NeighborsResponse,
    summary="Entity neighbors from WorldModelAgent graph",
)
async def get_entity_neighbors(
    entity: str = Query(..., description="Canonical entity name to look up"),
    limit: int = Query(20, ge=1, le=100, description="Max neighbors to return"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> NeighborsResponse:
    """Return all entities that co-occur with the given entity across tenant memories."""
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_world_model_runs(db, tenant.org_id)
    if not runs:
        return NeighborsResponse(
            entity=entity,
            neighbors=[],
            total=0,
            retrieved_at=datetime.now(timezone.utc),
        )

    nodes = _collect_nodes(runs)
    edges = _collect_edges(runs)

    # aggregate neighbor stats
    neighbor_stats: dict[str, dict[str, Any]] = {}
    for edge in edges:
        a = edge["from_entity"]
        b = edge["to_entity"]
        w = float(edge.get("weight", 0.5))
        if a == entity:
            neighbor = b
        elif b == entity:
            neighbor = a
        else:
            continue
        if neighbor not in neighbor_stats:
            neighbor_stats[neighbor] = {"edge_count": 0, "max_weight": 0.0}
        neighbor_stats[neighbor]["edge_count"] += 1
        neighbor_stats[neighbor]["max_weight"] = max(
            neighbor_stats[neighbor]["max_weight"], w
        )

    result: list[NeighborItem] = []
    for name, stats in neighbor_stats.items():
        node_meta = nodes.get(name, {})
        result.append(
            NeighborItem(
                entity=name,
                entity_type=node_meta.get("entity_type", "concept"),
                domains=node_meta.get("domains") or [],
                edge_count=stats["edge_count"],
                max_weight=round(stats["max_weight"], 3),
            )
        )

    result.sort(key=lambda n: (-n.edge_count, -n.max_weight))
    result = result[:limit]

    return NeighborsResponse(
        entity=entity,
        neighbors=result,
        total=len(result),
        retrieved_at=datetime.now(timezone.utc),
    )


@router.get(
    "/entity-path",
    response_model=EntityPathResponse,
    summary="Shortest entity-to-entity path through the co-occurrence graph",
)
async def get_entity_path(
    from_entity: str = Query(..., description="Source entity name"),
    to_entity: str = Query(..., description="Target entity name"),
    max_hops: int = Query(6, ge=1, le=10, description="Maximum BFS depth"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> EntityPathResponse:
    """Find the shortest path between two entities in the WorldModelAgent graph."""
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_world_model_runs(db, tenant.org_id)
    edges = _collect_edges(runs)
    adj = _build_adjacency(edges)
    path = _bfs_path(adj, from_entity, to_entity, max_hops=max_hops)

    return EntityPathResponse(
        from_entity=from_entity,
        to_entity=to_entity,
        path=path,
        hop_count=max(0, len(path) - 1),
        found=len(path) > 0,
        retrieved_at=datetime.now(timezone.utc),
    )


@router.get(
    "/changes",
    response_model=ChangesResponse,
    summary="Entity state changes detected by WorldModelAgent",
)
async def get_entity_changes(
    entity: Optional[str] = Query(
        None, description="Filter to a specific entity (omit for all)"
    ),
    limit: int = Query(50, ge=1, le=500, description="Max changes to return"),
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> ChangesResponse:
    """Return state-change events (domain expansions, high-confidence nodes) for the tenant."""
    await set_tenant_context(
        db, tenant.user_id, tenant.org_id, tenant.roles_string, tenant.clearance_level
    )
    runs = await _load_world_model_runs(db, tenant.org_id)
    raw = _collect_changes(runs)

    if entity:
        raw = [c for c in raw if c["entity"] == entity]

    raw = raw[:limit]
    changes = [
        ChangeItem(
            entity=c["entity"],
            change_type=c["change_type"],
            description=c.get("description", ""),
            memory_id=c["memory_id"],
        )
        for c in raw
    ]

    return ChangesResponse(
        entity=entity,
        changes=changes,
        total=len(changes),
        retrieved_at=datetime.now(timezone.utc),
    )
