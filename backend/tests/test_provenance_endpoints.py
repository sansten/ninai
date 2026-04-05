from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app
from app.services.memory_provenance_service import MemoryProvenanceService


@dataclass
class _FakeMemory:
    id: str
    organization_id: str
    content_preview: str
    source_id: str | None
    created_at: datetime
    updated_at: datetime


@dataclass
class _FakeEdge:
    id: str
    source_id: str
    target_id: str
    edge_type: str
    agent_name: str
    created_at: datetime
    edge_metadata: dict[str, Any]


class _Scalars:
    def __init__(self, items: list[Any]):
        self._items = items

    def all(self):
        return self._items


class _ListResult:
    def __init__(self, items: list[Any]):
        self._items = items

    def scalars(self):
        return _Scalars(self._items)


class _ScalarOneOrNoneResult:
    def __init__(self, item: Any | None):
        self._item = item

    def scalar_one_or_none(self):
        return self._item


def _auth_headers(*, org_id: str = "o1", user_id: str = "u1", roles: list[str] | None = None) -> dict[str, str]:
    token = create_access_token(user_id=user_id, org_id=org_id, roles=roles or ["org_admin"])
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_memory_lineage_returns_lineage_payload(monkeypatch):
    lineage = {
        "root_sources": ["jira:INC-1"],
        "edges": [
            {
                "id": "e1",
                "source_id": "jira:INC-1",
                "target_id": "m1",
                "edge_type": "ingest",
                "agent_name": "IngestService",
                "created_at": "2026-04-05T00:00:00+00:00",
                "metadata": {},
            }
        ],
        "depth": 1,
        "agent_chain": ["IngestService"],
    }
    monkeypatch.setattr(MemoryProvenanceService, "get_lineage", AsyncMock(return_value=lineage))

    session = AsyncMock(spec=AsyncSession)
    session.execute = AsyncMock(return_value=AsyncMock())

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/provenance/m1/lineage", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["memory_id"] == "m1"
        assert body["lineage"]["root_sources"] == ["jira:INC-1"]
        assert "IngestService" in body["summary"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_memory_citations_returns_formatted_citation(monkeypatch):
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    memory = _FakeMemory(
        id="m1",
        organization_id="o1",
        content_preview="Deployment failed at 14:32 UTC",
        source_id="jira:INC-4421",
        created_at=now,
        updated_at=now,
    )
    lineage = {
        "root_sources": ["jira:INC-4421"],
        "edges": [
            {
                "id": "e1",
                "source_id": "jira:INC-4421",
                "target_id": "m1",
                "edge_type": "ingest",
                "agent_name": "IngestService",
                "created_at": "2026-03-15T14:35:00Z",
                "metadata": {
                    "source": "PagerDuty incident INC-4421",
                    "author": "alice@company.com",
                    "confidence": 0.95,
                },
            }
        ],
        "depth": 1,
        "agent_chain": ["CredibilityAgent v1", "AnomalyDetectionAgent v1"],
    }
    monkeypatch.setattr(MemoryProvenanceService, "get_lineage", AsyncMock(return_value=lineage))

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM memory_metadata" in sql:
            return _ScalarOneOrNoneResult(memory)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/provenance/m1/citations", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["memory_id"] == "m1"
        assert body["content"] == "Deployment failed at 14:32 UTC"
        assert body["citation"]["source"] == "PagerDuty incident INC-4421"
        assert body["citation"]["author"] == "alice@company.com"
        assert body["citation"]["confidence"] == 0.95
        assert body["citation"]["verified_by"] == ["CredibilityAgent v1", "AnomalyDetectionAgent v1"]
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_search_provenance_by_source_returns_matching_memories():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    edges = [
        _FakeEdge(
            id="e1",
            source_id="jira:INC-1",
            target_id="m1",
            edge_type="ingest",
            agent_name="IngestService",
            created_at=now,
            edge_metadata={"source_system": "jira"},
        ),
        _FakeEdge(
            id="e2",
            source_id="slack:123",
            target_id="m2",
            edge_type="ingest",
            agent_name="IngestService",
            created_at=now,
            edge_metadata={"source_system": "slack"},
        ),
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM provenance_edges" in sql:
            return _ListResult(edges)
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/provenance/search?source=jira", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["source"] == "jira"
        assert body["count"] == 1
        assert body["memory_ids"] == ["m1"]
        assert body["edges"][0]["source_id"] == "jira:INC-1"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_assert_provenance_creates_manual_edge(monkeypatch):
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    memory = _FakeMemory(
        id="m1",
        organization_id="o1",
        content_preview="x",
        source_id=None,
        created_at=now,
        updated_at=now,
    )
    edge = _FakeEdge(
        id="e-manual",
        source_id="manual:jira",
        target_id="m1",
        edge_type="manual_assert",
        agent_name="ManualProvenanceAssertion",
        created_at=now,
        edge_metadata={"source": "jira"},
    )

    monkeypatch.setattr(MemoryProvenanceService, "record_edge", AsyncMock(return_value=edge))

    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM memory_metadata" in sql:
            return _ScalarOneOrNoneResult(memory)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/provenance/assert",
                headers=_auth_headers(roles=["org_admin"]),
                json={
                    "memory_id": "m1",
                    "source": "jira",
                    "author": "alice@company.com",
                    "confidence": 0.95,
                    "verified_by": ["CredibilityAgent v1", "AnomalyDetectionAgent v1"],
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["asserted"] is True
        assert body["memory_id"] == "m1"
        assert body["edge"]["id"] == "e-manual"
        assert body["edge"]["edge_type"] == "manual_assert"
    finally:
        app.dependency_overrides.clear()