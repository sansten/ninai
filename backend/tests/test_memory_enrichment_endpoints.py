"""Tests for memory enrichment API endpoints — Phase 26."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


def _mock_db(runs: list[_FakeRun], add_ok: bool = True):
    session = AsyncMock(spec=AsyncSession)

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL" in sql:
            return AsyncMock()
        if "agent_runs" in sql.lower():
            return _Result(runs)
        return _Result([])

    session.execute = AsyncMock(side_effect=_execute)
    if add_ok:
        session.add = MagicMock()
        session.commit = AsyncMock()

    return session


def _override(session):
    async def _get_db():
        yield session

    app.dependency_overrides[get_db] = _get_db


def _clear():
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# TestGetEnrichment
# ---------------------------------------------------------------------------

class TestGetEnrichment:
    @pytest.mark.asyncio
    async def test_returns_merged_enrichment(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="CredibilityAgent",
            outputs={"credibility_score": 0.85, "source_tier": "internal"},
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/enrichment", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["memory_id"] == "mem-1"
        assert body["enrichment"]["credibility_score"] == pytest.approx(0.85)
        assert body["agent_count"] == 1

    @pytest.mark.asyncio
    async def test_404_when_no_runs(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/enrichment", headers=_auth_headers())
        _clear()
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_merges_multiple_agent_outputs(self):
        runs = [
            _FakeRun(id="ar-1", organization_id="org-1", memory_id="mem-1",
                     agent_name="CredibilityAgent",
                     outputs={"credibility_score": 0.75}),
            _FakeRun(id="ar-2", organization_id="org-1", memory_id="mem-1",
                     agent_name="UncertaintyReportingAgent",
                     outputs={"uncertainty_level": "low", "review_recommended": False}),
        ]
        _override(_mock_db(runs))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/enrichment", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        enrichment = resp.json()["enrichment"]
        assert "credibility_score" in enrichment
        assert "uncertainty_level" in enrichment
        assert resp.json()["agent_count"] == 2

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/enrichment")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_response_has_retrieved_at(self):
        run = _FakeRun(id="ar-1", organization_id="org-1", memory_id="mem-1",
                       agent_name="CredibilityAgent", outputs={"credibility_score": 0.7})
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/enrichment", headers=_auth_headers())
        _clear()
        assert "retrieved_at" in resp.json()


# ---------------------------------------------------------------------------
# TestGetNarrative
# ---------------------------------------------------------------------------

class TestGetNarrative:
    @pytest.mark.asyncio
    async def test_returns_narrative_fields(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="NarrativeSynthesisAgent",
            outputs={
                "narrative_text": "This memory concerns Acme Corp.",
                "key_entities": ["Acme Corp"],
                "timeline_summary": None,
                "action_items": [],
                "tone": "informational",
                "confidence": 0.75,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/narrative", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["narrative_text"] == "This memory concerns Acme Corp."
        assert body["tone"] == "informational"
        assert body["memory_id"] == "mem-1"

    @pytest.mark.asyncio
    async def test_404_when_no_narrative_run(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-x/narrative", headers=_auth_headers())
        _clear()
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_key_entities_is_list(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="NarrativeSynthesisAgent",
            outputs={"narrative_text": "Text.", "key_entities": ["A", "B"], "tone": "cautionary"},
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/narrative", headers=_auth_headers())
        _clear()
        assert isinstance(resp.json()["key_entities"], list)


# ---------------------------------------------------------------------------
# TestGetUncertainty
# ---------------------------------------------------------------------------

class TestGetUncertainty:
    @pytest.mark.asyncio
    async def test_returns_uncertainty_fields(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="UncertaintyReportingAgent",
            outputs={
                "uncertainty_level": "medium",
                "review_recommended": True,
                "uncertainty_score": 0.45,
                "unresolved_conflicts": [],
                "low_confidence_fields": ["credibility_score"],
                "confidence": 0.75,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/uncertainty", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["uncertainty_level"] == "medium"
        assert body["review_recommended"] is True
        assert body["uncertainty_score"] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_404_when_no_uncertainty_run(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-x/uncertainty", headers=_auth_headers())
        _clear()
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_low_confidence_fields_is_list(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="UncertaintyReportingAgent",
            outputs={"uncertainty_level": "low", "low_confidence_fields": [], "uncertainty_score": 0.1},
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/uncertainty", headers=_auth_headers())
        _clear()
        assert isinstance(resp.json()["low_confidence_fields"], list)


# ---------------------------------------------------------------------------
# TestGetAnomalies
# ---------------------------------------------------------------------------

class TestGetAnomalies:
    @pytest.mark.asyncio
    async def test_returns_anomaly_fields(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="AnomalyDetectionAgent",
            outputs={
                "anomaly_detected": True,
                "anomaly_type": "credibility_spike",
                "severity": "medium",
                "affected_fields": ["credibility_score"],
                "anomaly_score": 0.42,
                "confidence": 0.75,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/anomalies", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["anomaly_detected"] is True
        assert body["anomaly_type"] == "credibility_spike"
        assert body["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_404_when_no_anomaly_run(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-x/anomalies", headers=_auth_headers())
        _clear()
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_anomaly_clean_memory(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="AnomalyDetectionAgent",
            outputs={
                "anomaly_detected": False,
                "anomaly_type": None,
                "severity": None,
                "affected_fields": [],
                "anomaly_score": 0.0,
                "confidence": 0.80,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/memories/mem-1/anomalies", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        assert resp.json()["anomaly_detected"] is False


# ---------------------------------------------------------------------------
# TestSubmitFeedback
# ---------------------------------------------------------------------------

class TestSubmitFeedback:
    @pytest.mark.asyncio
    async def test_approve_feedback_201(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                headers=_auth_headers(),
                json={"verdict": "approve", "resolution_confirmed": True},
            )
        _clear()
        assert resp.status_code == 201
        body = resp.json()
        assert body["memory_id"] == "mem-1"
        assert body["verdict"] == "approve"
        assert "feedback_id" in body
        assert "accepted_at" in body

    @pytest.mark.asyncio
    async def test_reject_feedback_201(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                headers=_auth_headers(),
                json={"verdict": "reject", "playbook_correct": False},
            )
        _clear()
        assert resp.status_code == 201
        assert resp.json()["verdict"] == "reject"

    @pytest.mark.asyncio
    async def test_feedback_with_credibility_override(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                headers=_auth_headers(),
                json={"verdict": "partial", "credibility_override": 0.90},
            )
        _clear()
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_feedback_credibility_out_of_range_422(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                headers=_auth_headers(),
                json={"credibility_override": 1.5},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_feedback_body_accepted(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                headers=_auth_headers(),
                json={},
            )
        _clear()
        assert resp.status_code == 201
        assert resp.json()["verdict"] is None

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                json={"verdict": "approve"},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_feedback_id_is_string(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/memories/mem-1/enrichment-feedback",
                headers=_auth_headers(),
                json={"verdict": "approve"},
            )
        _clear()
        assert isinstance(resp.json()["feedback_id"], str)
        assert len(resp.json()["feedback_id"]) > 0


# ---------------------------------------------------------------------------
# TestRouteRegistration
# ---------------------------------------------------------------------------

class TestRouteRegistration:
    def _route_paths(self):
        return {r.path for r in app.routes}

    def test_enrichment_route_registered(self):
        paths = self._route_paths()
        assert any("enrichment" in p for p in paths)

    def test_narrative_route_registered(self):
        paths = self._route_paths()
        assert any("narrative" in p for p in paths)

    def test_uncertainty_route_registered(self):
        paths = self._route_paths()
        assert any("uncertainty" in p for p in paths)

    def test_anomalies_route_registered(self):
        paths = self._route_paths()
        assert any("anomalies" in p for p in paths)

    def test_feedback_route_registered(self):
        paths = self._route_paths()
        assert any("feedback" in p for p in paths)
