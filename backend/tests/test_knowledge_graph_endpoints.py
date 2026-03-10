"""Tests for Tenant-Scoped Knowledge Graph API — Phase 30."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers(*, org_id: str = "org-1", user_id: str = "user-1") -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=["member"])
    return {"Authorization": f"Bearer {token}"}


@dataclass
class _FakeRun:
    id: str
    organization_id: str
    memory_id: str
    agent_name: str
    agent_version: str = "v1"
    inputs_hash: str = "h" * 64
    status: str = "success"
    confidence: float = 0.80
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    provenance: list = field(default_factory=list)


class _Scalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class _Result:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


def _mock_db(runs: list[_FakeRun]):
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL" in sql:
            return AsyncMock()
        if "agent_runs" in sql.lower():
            return _Result(runs)
        return _Result([])

    session.execute = AsyncMock(side_effect=_execute)
    return session


def _override(session):
    async def _get_db():
        yield session

    app.dependency_overrides[get_db] = _get_db


def _clear():
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

def _wm_run(
    memory_id: str = "mem-1",
    org_id: str = "org-1",
    run_id: str = "ar-1",
    world_nodes: list | None = None,
    entity_edges: list | None = None,
    state_changes: list | None = None,
) -> _FakeRun:
    return _FakeRun(
        id=run_id,
        organization_id=org_id,
        memory_id=memory_id,
        agent_name="WorldModelAgent",
        outputs={
            "world_nodes": world_nodes or [],
            "entity_edges": entity_edges or [],
            "state_changes": state_changes or [],
            "graph_summary": {"total_nodes": len(world_nodes or [])},
            "confidence": 0.75,
            "rationale": "heuristic",
        },
    )


_NODES = [
    {"entity": "Acme Corp", "entity_type": "org", "domains": ["sales", "finance"],
     "silo_count": 2, "state_version": 2, "node_confidence": 0.82},
    {"entity": "Alice", "entity_type": "person", "domains": ["sales"],
     "silo_count": 1, "state_version": 1, "node_confidence": 0.65},
    {"entity": "Project X", "entity_type": "project", "domains": ["engineering"],
     "silo_count": 1, "state_version": 1, "node_confidence": 0.60},
]

_EDGES = [
    {"from_entity": "Acme Corp", "to_entity": "Alice", "relationship_type": "co_occurrence", "weight": 0.735},
    {"from_entity": "Acme Corp", "to_entity": "Project X", "relationship_type": "co_occurrence", "weight": 0.71},
]

_CHANGES = [
    {"entity": "Acme Corp", "change_type": "domain_expansion",
     "description": "Acme Corp is active across 2 silos: sales, finance"},
    {"entity": "Acme Corp", "change_type": "high_confidence_node",
     "description": "Acme Corp resolved with node_confidence 0.82"},
]


# ---------------------------------------------------------------------------
# TestGetNeighbors
# ---------------------------------------------------------------------------


class TestGetNeighbors:
    @pytest.mark.asyncio
    async def test_returns_neighbors_for_known_entity(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["entity"] == "Acme Corp"
        assert body["total"] == 2
        entities = {n["entity"] for n in body["neighbors"]}
        assert "Alice" in entities
        assert "Project X" in entities

    @pytest.mark.asyncio
    async def test_empty_neighbors_for_unknown_entity(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Unknown Entity"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["neighbors"] == []

    @pytest.mark.asyncio
    async def test_200_with_no_runs_returns_empty(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors", params={"entity": "Acme Corp"}
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_respects_limit_param(self):
        # build 5 neighbors for "Acme Corp"
        nodes = [{"entity": f"Entity{i}", "entity_type": "concept", "domains": [],
                  "silo_count": 1, "state_version": 1, "node_confidence": 0.5}
                 for i in range(5)]
        nodes.append({"entity": "Acme Corp", "entity_type": "org", "domains": [],
                      "silo_count": 1, "state_version": 1, "node_confidence": 0.7})
        edges = [{"from_entity": "Acme Corp", "to_entity": f"Entity{i}",
                  "relationship_type": "co_occurrence", "weight": 0.5}
                 for i in range(5)]
        run = _wm_run(world_nodes=nodes, entity_edges=edges)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp", "limit": 3},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        assert len(resp.json()["neighbors"]) == 3

    @pytest.mark.asyncio
    async def test_neighbor_includes_entity_type_and_domains(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        alice = next(n for n in resp.json()["neighbors"] if n["entity"] == "Alice")
        assert alice["entity_type"] == "person"
        assert "sales" in alice["domains"]

    @pytest.mark.asyncio
    async def test_merges_edges_from_multiple_runs(self):
        run1 = _wm_run(
            run_id="ar-1", memory_id="mem-1",
            world_nodes=_NODES,
            entity_edges=[{"from_entity": "Acme Corp", "to_entity": "Alice",
                           "relationship_type": "co_occurrence", "weight": 0.7}],
        )
        run2 = _wm_run(
            run_id="ar-2", memory_id="mem-2",
            world_nodes=_NODES,
            entity_edges=[{"from_entity": "Acme Corp", "to_entity": "Alice",
                           "relationship_type": "co_occurrence", "weight": 0.9}],
        )
        _override(_mock_db([run1, run2]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        alice = next(n for n in resp.json()["neighbors"] if n["entity"] == "Alice")
        assert alice["edge_count"] == 2
        assert alice["max_weight"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_sorts_by_edge_count_descending(self):
        nodes = [
            {"entity": "Acme Corp", "entity_type": "org", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.7},
            {"entity": "Alice", "entity_type": "person", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.6},
            {"entity": "Bob", "entity_type": "person", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.6},
        ]
        # Alice appears in 2 edges, Bob in 1
        edges = [
            {"from_entity": "Acme Corp", "to_entity": "Alice",
             "relationship_type": "co_occurrence", "weight": 0.6},
            {"from_entity": "Acme Corp", "to_entity": "Alice",
             "relationship_type": "co_occurrence", "weight": 0.7},
            {"from_entity": "Acme Corp", "to_entity": "Bob",
             "relationship_type": "co_occurrence", "weight": 0.5},
        ]
        run = _wm_run(world_nodes=nodes, entity_edges=edges)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        neighbors = resp.json()["neighbors"]
        assert neighbors[0]["entity"] == "Alice"
        assert neighbors[1]["entity"] == "Bob"

    @pytest.mark.asyncio
    async def test_handles_run_with_none_outputs(self):
        run = _FakeRun(
            id="ar-bad", organization_id="org-1", memory_id="mem-1",
            agent_name="WorldModelAgent", outputs=None,
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_merges_domains_across_runs(self):
        nodes_a = [{"entity": "Alice", "entity_type": "person",
                    "domains": ["sales"], "silo_count": 1, "state_version": 1, "node_confidence": 0.6},
                   {"entity": "Acme Corp", "entity_type": "org",
                    "domains": ["sales"], "silo_count": 1, "state_version": 1, "node_confidence": 0.7}]
        nodes_b = [{"entity": "Alice", "entity_type": "person",
                    "domains": ["finance"], "silo_count": 1, "state_version": 1, "node_confidence": 0.65},
                   {"entity": "Acme Corp", "entity_type": "org",
                    "domains": ["finance"], "silo_count": 1, "state_version": 1, "node_confidence": 0.72}]
        edge = {"from_entity": "Acme Corp", "to_entity": "Alice",
                "relationship_type": "co_occurrence", "weight": 0.65}
        run1 = _wm_run(run_id="ar-1", memory_id="mem-1", world_nodes=nodes_a, entity_edges=[edge])
        run2 = _wm_run(run_id="ar-2", memory_id="mem-2", world_nodes=nodes_b, entity_edges=[edge])
        _override(_mock_db([run1, run2]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        alice = next(n for n in resp.json()["neighbors"] if n["entity"] == "Alice")
        assert set(alice["domains"]) == {"sales", "finance"}

    @pytest.mark.asyncio
    async def test_response_has_retrieved_at(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        assert "retrieved_at" in resp.json()

    @pytest.mark.asyncio
    async def test_bidirectional_edges(self):
        """Edge B→A should count A as neighbor of B."""
        nodes = [
            {"entity": "Alpha", "entity_type": "concept", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.5},
            {"entity": "Beta", "entity_type": "concept", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.5},
        ]
        edges = [{"from_entity": "Beta", "to_entity": "Alpha",
                  "relationship_type": "co_occurrence", "weight": 0.6}]
        run = _wm_run(world_nodes=nodes, entity_edges=edges)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/neighbors",
                params={"entity": "Alpha"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.json()["total"] == 1
        assert resp.json()["neighbors"][0]["entity"] == "Beta"


# ---------------------------------------------------------------------------
# TestGetEntityPath
# ---------------------------------------------------------------------------


class TestGetEntityPath:
    @pytest.mark.asyncio
    async def test_finds_direct_path(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "Acme Corp", "to_entity": "Alice"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert body["hop_count"] == 1
        assert body["path"] == ["Acme Corp", "Alice"]

    @pytest.mark.asyncio
    async def test_finds_multi_hop_path(self):
        nodes = [
            {"entity": "A", "entity_type": "concept", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.5},
            {"entity": "B", "entity_type": "concept", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.5},
            {"entity": "C", "entity_type": "concept", "domains": [],
             "silo_count": 1, "state_version": 1, "node_confidence": 0.5},
        ]
        edges = [
            {"from_entity": "A", "to_entity": "B", "relationship_type": "co_occurrence", "weight": 0.5},
            {"from_entity": "B", "to_entity": "C", "relationship_type": "co_occurrence", "weight": 0.5},
        ]
        run = _wm_run(world_nodes=nodes, entity_edges=edges)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "A", "to_entity": "C"},
                headers=_auth_headers(),
            )
        _clear()
        body = resp.json()
        assert body["found"] is True
        assert body["hop_count"] == 2
        assert body["path"] == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_not_found_returns_found_false(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "Alice", "to_entity": "Orphan Entity"},
                headers=_auth_headers(),
            )
        _clear()
        body = resp.json()
        assert resp.status_code == 200
        assert body["found"] is False
        assert body["path"] == []
        assert body["hop_count"] == 0

    @pytest.mark.asyncio
    async def test_path_includes_both_endpoints(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "Acme Corp", "to_entity": "Project X"},
                headers=_auth_headers(),
            )
        _clear()
        path = resp.json()["path"]
        assert path[0] == "Acme Corp"
        assert path[-1] == "Project X"

    @pytest.mark.asyncio
    async def test_hop_count_is_path_length_minus_one(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "Acme Corp", "to_entity": "Alice"},
                headers=_auth_headers(),
            )
        _clear()
        body = resp.json()
        assert body["hop_count"] == len(body["path"]) - 1

    @pytest.mark.asyncio
    async def test_same_entity_path(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "Acme Corp", "to_entity": "Acme Corp"},
                headers=_auth_headers(),
            )
        _clear()
        body = resp.json()
        assert body["found"] is True
        assert body["path"] == ["Acme Corp"]
        assert body["hop_count"] == 0

    @pytest.mark.asyncio
    async def test_200_with_no_runs(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "A", "to_entity": "B"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["found"] is False

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "A", "to_entity": "B"},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_max_hops_respected(self):
        # chain of 10 nodes
        n = 10
        nodes = [{"entity": f"N{i}", "entity_type": "concept", "domains": [],
                  "silo_count": 1, "state_version": 1, "node_confidence": 0.5}
                 for i in range(n)]
        edges = [{"from_entity": f"N{i}", "to_entity": f"N{i+1}",
                  "relationship_type": "co_occurrence", "weight": 0.5}
                 for i in range(n - 1)]
        run = _wm_run(world_nodes=nodes, entity_edges=edges)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # max_hops=2 → should not find N0→N9 (9 hops away)
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "N0", "to_entity": "N9", "max_hops": 2},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.json()["found"] is False

    @pytest.mark.asyncio
    async def test_response_has_retrieved_at(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/entity-path",
                params={"from_entity": "A", "to_entity": "B"},
                headers=_auth_headers(),
            )
        _clear()
        assert "retrieved_at" in resp.json()


# ---------------------------------------------------------------------------
# TestGetChanges
# ---------------------------------------------------------------------------


class TestGetChanges:
    @pytest.mark.asyncio
    async def test_returns_all_changes_without_filter(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES, state_changes=_CHANGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/changes",
                headers=_auth_headers(),
            )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == len(_CHANGES)
        assert body["entity"] is None

    @pytest.mark.asyncio
    async def test_filters_changes_by_entity(self):
        changes = [
            *_CHANGES,
            {"entity": "Alice", "change_type": "domain_expansion",
             "description": "Alice active across sales"},
        ]
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES, state_changes=changes)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/changes",
                params={"entity": "Alice"},
                headers=_auth_headers(),
            )
        _clear()
        body = resp.json()
        assert body["entity"] == "Alice"
        assert all(c["entity"] == "Alice" for c in body["changes"])
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_empty_when_no_runs(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_empty_when_entity_not_in_changes(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES, state_changes=_CHANGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/changes",
                params={"entity": "Unknown"},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_change_item_includes_memory_id(self):
        run = _wm_run(
            run_id="ar-1", memory_id="mem-99",
            world_nodes=_NODES, entity_edges=_EDGES, state_changes=_CHANGES,
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes", headers=_auth_headers())
        _clear()
        for c in resp.json()["changes"]:
            assert c["memory_id"] == "mem-99"

    @pytest.mark.asyncio
    async def test_respects_limit_param(self):
        changes = [
            {"entity": f"Ent{i}", "change_type": "domain_expansion", "description": f"desc {i}"}
            for i in range(10)
        ]
        run = _wm_run(state_changes=changes)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                "/api/v1/graph/changes",
                params={"limit": 5},
                headers=_auth_headers(),
            )
        _clear()
        assert resp.json()["total"] == 5

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_collects_changes_from_multiple_runs(self):
        run1 = _wm_run(
            run_id="ar-1", memory_id="mem-1",
            state_changes=[_CHANGES[0]],
        )
        run2 = _wm_run(
            run_id="ar-2", memory_id="mem-2",
            state_changes=[_CHANGES[1]],
        )
        _override(_mock_db([run1, run2]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes", headers=_auth_headers())
        _clear()
        assert resp.json()["total"] == 2
        memory_ids = {c["memory_id"] for c in resp.json()["changes"]}
        assert memory_ids == {"mem-1", "mem-2"}

    @pytest.mark.asyncio
    async def test_change_has_required_fields(self):
        run = _wm_run(state_changes=_CHANGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes", headers=_auth_headers())
        _clear()
        for c in resp.json()["changes"]:
            assert "entity" in c
            assert "change_type" in c
            assert "description" in c
            assert "memory_id" in c

    @pytest.mark.asyncio
    async def test_total_matches_len_changes(self):
        run = _wm_run(state_changes=_CHANGES)
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes", headers=_auth_headers())
        _clear()
        body = resp.json()
        assert body["total"] == len(body["changes"])

    @pytest.mark.asyncio
    async def test_handles_run_with_no_state_changes(self):
        run = _wm_run(world_nodes=_NODES, entity_edges=_EDGES, state_changes=[])
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/graph/changes", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
