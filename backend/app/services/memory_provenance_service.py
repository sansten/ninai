"""Memory provenance graph service (Phase 77)."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provenance_edge import ProvenanceEdge


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class MemoryProvenanceService:
    async def record_edge(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        agent_name: str,
        metadata: dict | None = None,
    ) -> ProvenanceEdge:
        row = ProvenanceEdge(
            org_id=org_id,
            source_id=str(source_id),
            target_id=str(target_id),
            edge_type=str(edge_type),
            agent_name=str(agent_name),
            edge_metadata=dict(metadata or {}),
        )
        db.add(row)
        await db.flush()
        return row

    @staticmethod
    def _edge_dict(edge: ProvenanceEdge) -> dict[str, Any]:
        return {
            "id": str(edge.id),
            "source_id": str(edge.source_id),
            "target_id": str(edge.target_id),
            "edge_type": str(edge.edge_type),
            "agent_name": str(edge.agent_name),
            "created_at": _as_utc(edge.created_at).isoformat() if edge.created_at else None,
            "metadata": dict(edge.edge_metadata or {}),
        }

    async def get_lineage(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        memory_id: str,
        max_depth: int = 10,
    ) -> dict:
        depth_limit = max(0, int(max_depth if max_depth is not None else 10))

        rows = list(
            (
                await db.execute(
                    select(ProvenanceEdge).where(ProvenanceEdge.org_id == org_id)
                )
            )
            .scalars()
            .all()
        )

        by_target: dict[str, list[ProvenanceEdge]] = {}
        by_source: dict[str, list[ProvenanceEdge]] = {}
        for edge in rows:
            by_target.setdefault(str(edge.target_id), []).append(edge)
            by_source.setdefault(str(edge.source_id), []).append(edge)

        queue: deque[tuple[str, int, list[ProvenanceEdge]]] = deque([(str(memory_id), 0, [])])
        visited_nodes: set[tuple[str, int]] = {(str(memory_id), 0)}
        traversed_edge_ids: set[str] = set()
        traversed_edges: list[ProvenanceEdge] = []

        root_sources: set[str] = set()
        max_reached_depth = 0
        best_path: list[ProvenanceEdge] = []

        while queue:
            node_id, depth, path = queue.popleft()
            max_reached_depth = max(max_reached_depth, depth)

            incoming = by_target.get(node_id, [])
            if not incoming or depth >= depth_limit:
                if node_id != str(memory_id):
                    root_sources.add(node_id)
                if len(path) > len(best_path):
                    best_path = path
                continue

            for edge in incoming:
                edge_id = str(edge.id)
                if edge_id not in traversed_edge_ids:
                    traversed_edge_ids.add(edge_id)
                    traversed_edges.append(edge)

                next_node = str(edge.source_id)
                next_depth = depth + 1
                key = (next_node, next_depth)
                if key in visited_nodes:
                    continue
                visited_nodes.add(key)
                queue.append((next_node, next_depth, [edge, *path]))

        if not traversed_edges:
            return {
                "root_sources": [str(memory_id)],
                "edges": [],
                "depth": 0,
                "agent_chain": [],
            }

        agent_chain = [str(edge.agent_name) for edge in best_path]

        return {
            "root_sources": sorted(root_sources),
            "edges": [self._edge_dict(e) for e in traversed_edges],
            "depth": max_reached_depth,
            "agent_chain": agent_chain,
        }

    async def get_descendants(
        self,
        *,
        db: AsyncSession,
        org_id: str,
        source_id: str,
    ) -> list[str]:
        rows = list(
            (
                await db.execute(
                    select(ProvenanceEdge).where(ProvenanceEdge.org_id == org_id)
                )
            )
            .scalars()
            .all()
        )

        by_source: dict[str, list[str]] = {}
        for edge in rows:
            by_source.setdefault(str(edge.source_id), []).append(str(edge.target_id))

        start = str(source_id)
        queue: deque[str] = deque([start])
        seen: set[str] = {start}
        descendants: set[str] = set()

        while queue:
            node = queue.popleft()
            for nxt in by_source.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                descendants.add(nxt)
                queue.append(nxt)

        return sorted(descendants)

    def summarise_lineage(self, lineage: dict) -> str:
        agent_chain = list(lineage.get("agent_chain") or [])
        if not agent_chain:
            return "No provenance lineage available"

        parts = [f"enriched by {agent_chain[0]}"]
        for name in agent_chain[1:]:
            parts.append(f"then transformed by {name}")
        return " -> ".join(parts)
