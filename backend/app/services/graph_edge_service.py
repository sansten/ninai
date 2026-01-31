"""Graph edge materialization.

Turns GraphLinkingAgent outputs into persisted edges in Postgres.

Design goals:
- Tenant-safe: requires organization_id and runs under RLS context.
- Idempotent: uses a unique constraint and inserts with ON CONFLICT DO NOTHING.
- Defensive parsing: ignores malformed edge entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from app.models.memory_edge import MemoryEdge


@dataclass(frozen=True)
class ExtractedEdge:
    from_node: str
    to_node: str
    relation: str
    weight: float = 1.0
    explanation: str | None = None


def _coerce_weight(val: Any) -> float:
    try:
        w = float(val)
    except Exception:
        return 1.0
    if w != w:  # NaN
        return 1.0
    return max(0.0, min(1.0, w))


def extract_edges_from_graph_outputs(outputs: dict[str, Any] | None) -> list[ExtractedEdge]:
    """Extract normalized edges from GraphLinkingAgent outputs."""

    if not isinstance(outputs, dict):
        return []

    graph = outputs.get("graph")
    if not isinstance(graph, dict):
        return []

    edges = graph.get("edges")
    if not isinstance(edges, list):
        return []

    seen: set[tuple[str, str, str]] = set()
    out: list[ExtractedEdge] = []

    for e in edges:
        if not isinstance(e, dict):
            continue

        src = str(e.get("source") or "").strip()
        dst = str(e.get("target") or "").strip()
        rel = str(e.get("relation") or "").strip()
        if not src or not dst or not rel:
            continue

        key = (src, dst, rel)
        if key in seen:
            continue
        seen.add(key)

        out.append(
            ExtractedEdge(
                from_node=src,
                to_node=dst,
                relation=rel,
                weight=_coerce_weight(e.get("weight", 1.0)),
                explanation=(str(e.get("explanation")).strip() if e.get("explanation") is not None else None),
            )
        )

    return out


class GraphEdgeService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_edges_for_memory(
        self,
        *,
        organization_id: str,
        memory_id: str,
        outputs: dict[str, Any] | None,
        created_by: str = "agent",
    ) -> int:
        edges = extract_edges_from_graph_outputs(outputs)
        if not edges:
            return 0

        values: list[dict[str, Any]] = []
        for e in edges:
            values.append(
                {
                    "id": str(uuid4()),
                    "organization_id": organization_id,
                    "memory_id": memory_id,
                    "from_node": e.from_node,
                    "to_node": e.to_node,
                    "relation": e.relation,
                    "weight": float(e.weight),
                    "explanation": e.explanation,
                    "created_by": created_by,
                }
            )

        stmt = insert(MemoryEdge).values(values)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[
                "organization_id",
                "memory_id",
                "from_node",
                "to_node",
                "relation",
            ]
        )

        res = await self.session.execute(stmt)
        # rowcount is driver-dependent; treat it as best-effort
        inserted = int(getattr(res, "rowcount", 0) or 0)
        await self.session.flush()
        return inserted
