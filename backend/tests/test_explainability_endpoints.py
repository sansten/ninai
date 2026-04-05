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


@dataclass
class _FakeAgentRun:
    id: str
    organization_id: str
    memory_id: str
    agent_name: str
    agent_version: str
    inputs_hash: str
    status: str
    confidence: float
    outputs: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    started_at: datetime
    finished_at: datetime
    trace_id: str | None
    provenance: list[dict[str, Any]]


@dataclass
class _FakeAgentRunEvent:
    id: str
    organization_id: str
    agent_run_id: str
    memory_id: str
    event_type: str
    step_index: int
    payload: dict[str, Any]
    summary_text: str
    created_at: datetime
    trace_id: str | None = None


@dataclass
class _FakeDecisionTrail:
    id: str
    timestamp: datetime
    memory_id: str
    organization_id: str
    agent_name: str
    agent_version: str
    decision: str
    confidence: float
    reasoning_snapshot: dict[str, Any]
    trace_id: str | None = None


@dataclass
class _FakeContradiction:
    id: str
    organization_id: str
    fact_a: str
    fact_b: str
    reason: str
    severity: str
    created_at: datetime
    resolved_at: datetime | None


@dataclass
class _FakeFact:
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source_memory_id: str


@dataclass
class _FakeAuditEvent:
    id: str
    timestamp: datetime
    event_type: str
    severity: str
    resource_type: str
    resource_id: str
    success: bool
    details: dict[str, Any]


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
async def test_explain_decision_returns_reasoning_trace():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    run = _FakeAgentRun(
        id="dec-1",
        organization_id="o1",
        memory_id="m-1",
        agent_name="CognitiveGateway",
        agent_version="1",
        inputs_hash="h" * 64,
        status="success",
        confidence=0.87,
        outputs={"decision": "Roll back", "confidence": 0.87, "alternatives_considered": [{"option": "hotfix"}]},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id="trail-1",
        provenance=[{"memory_id": "m-2"}],
    )
    events = [
        _FakeAgentRunEvent(
            id="e1",
            organization_id="o1",
            agent_run_id="dec-1",
            memory_id="m-1",
            event_type="summary",
            step_index=1,
            payload={"finding": "Error rate 340% above baseline"},
            summary_text="",
            created_at=now,
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(run)
        if "FROM agent_run_events" in sql:
            return _ListResult(events)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/explain/dec-1", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["decision_id"] == "dec-1"
        assert body["decision"] == "Roll back"
        assert body["audit_trail_id"] == "trail-1"
        assert body["memories_used"] == ["m-1", "m-2"]
        assert len(body["reasoning_steps"]) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_explain_memory_returns_decision_trail():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    trails = [
        _FakeDecisionTrail(
            id="t1",
            timestamp=now,
            memory_id="m1",
            organization_id="o1",
            agent_name="AnomalyDetectionAgent",
            agent_version="1",
            decision="flagged unusual spike",
            confidence=0.8,
            reasoning_snapshot={"anomaly_score": 0.9},
            trace_id="trace-1",
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_decision_trails" in sql:
            return _ListResult(trails)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/explain/memory/m1", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["memory_id"] == "m1"
        assert body["agents"] == ["AnomalyDetectionAgent"]
        assert body["reasoning_steps"][0]["finding"] == "flagged unusual spike"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_explain_conflict_returns_fact_contradiction_details():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    contradiction = _FakeContradiction(
        id="c1",
        organization_id="o1",
        fact_a="fa",
        fact_b="fb",
        reason="values disagree",
        severity="high",
        created_at=now,
        resolved_at=None,
    )
    fact_a = _FakeFact(id="fa", subject="service", predicate="status", object="healthy", confidence=0.9, source_memory_id="m1")
    fact_b = _FakeFact(id="fb", subject="service", predicate="status", object="down", confidence=0.88, source_memory_id="m2")

    fact_query_idx = 0
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        nonlocal fact_query_idx
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM contradictions" in sql:
            return _ScalarOneOrNoneResult(contradiction)
        if "FROM memory_facts" in sql:
            fact_query_idx += 1
            if fact_query_idx == 1:
                return _ScalarOneOrNoneResult(fact_a)
            if fact_query_idx == 2:
                return _ScalarOneOrNoneResult(fact_b)
        return _ScalarOneOrNoneResult(None)

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/explain/conflict/c1", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["conflict_id"] == "c1"
        assert body["reason"] == "values disagree"
        assert len(body["facts"]) == 2
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_explain_anomaly_returns_anomaly_trace():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    run = _FakeAgentRun(
        id="a1",
        organization_id="o1",
        memory_id="m1",
        agent_name="AnomalyDetectionAgent",
        agent_version="1",
        inputs_hash="h" * 64,
        status="success",
        confidence=0.91,
        outputs={"anomaly_detected": True, "anomaly_type": "conflict_flood", "severity": "high", "anomaly_score": 0.92},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id="tr-a1",
        provenance=[],
    )
    events = [
        _FakeAgentRunEvent(
            id="ae1",
            organization_id="o1",
            agent_run_id="a1",
            memory_id="m1",
            event_type="summary",
            step_index=1,
            payload={"finding": "conflict_count exceeded threshold"},
            summary_text="threshold exceeded",
            created_at=now,
        )
    ]

    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM agent_runs" in sql:
            return _ScalarOneOrNoneResult(run)
        if "FROM agent_run_events" in sql:
            return _ListResult(events)
        return AsyncMock()

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/explain/anomaly/a1", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["anomaly_id"] == "a1"
        assert body["anomaly_detected"] is True
        assert body["anomaly_type"] == "conflict_flood"
        assert len(body["reasoning_steps"]) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_audit_trail_endpoint_returns_session_trail():
    now = datetime(2026, 4, 5, tzinfo=timezone.utc)
    audit_events = [
        _FakeAuditEvent(
            id="aud1",
            timestamp=now,
            event_type="session.updated",
            severity="info",
            resource_type="cognitive_session",
            resource_id="s1",
            success=True,
            details={"status": "running"},
        )
    ]
    run = _FakeAgentRun(
        id="r1",
        organization_id="o1",
        memory_id="m1",
        agent_name="CognitiveGateway",
        agent_version="1",
        inputs_hash="h" * 64,
        status="success",
        confidence=0.8,
        outputs={},
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id="s1",
        provenance=[],
    )
    run_events = [
        _FakeAgentRunEvent(
            id="re1",
            organization_id="o1",
            agent_run_id="r1",
            memory_id="m1",
            event_type="summary",
            step_index=1,
            payload={"finding": "ok"},
            summary_text="ok",
            created_at=now,
        )
    ]

    call_idx = 0
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        nonlocal call_idx
        call_idx += 1
        sql = str(stmt)
        if "SET LOCAL app.current_" in sql:
            return AsyncMock()
        if "FROM audit_events" in sql:
            return _ListResult(audit_events)
        if "FROM agent_runs" in sql:
            return _ListResult([run])
        if "FROM agent_run_events" in sql:
            return _ListResult(run_events)
        if "FROM action_execution_records" in sql:
            return _ListResult([])
        return _ListResult([])

    session.execute = AsyncMock(side_effect=_execute)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/audit/trail/s1", headers=_auth_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "s1"
        assert len(body["audit_events"]) == 1
        assert len(body["agent_runs"]) == 1
        assert len(body["agent_events"]) == 1
    finally:
        app.dependency_overrides.clear()
