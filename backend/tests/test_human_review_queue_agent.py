"""Tests for Human Review Queue — Phase 32."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.human_review_queue_agent import (
    HumanReviewQueueAgent,
    build_review_context,
    build_review_reasons,
    compute_priority,
    run_heuristic,
)
from app.agents.types import AgentContext, AgentResult
from app.core.database import get_db
from app.core.security import create_access_token
from app.main import app


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _ctx(
    enrichment: dict[str, Any] | None = None,
    memory_id: str = "mem-1",
) -> AgentContext:
    return {
        "org_id": "org-1",
        "memory": {"id": memory_id, "enrichment": enrichment or {}},
        "runtime": {"job_id": "job-test"},
    }


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
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


def _mock_db_two_queries(first_runs: list[_FakeRun], second_runs: list[_FakeRun]):
    """Mock DB that returns different results for first vs second agent_runs query."""
    session = AsyncMock(spec=AsyncSession)
    call_count = [0]

    async def _execute(stmt, *args, **kwargs):
        sql = str(stmt)
        if "SET LOCAL" in sql:
            return AsyncMock()
        if "agent_runs" in sql.lower():
            idx = call_count[0]
            call_count[0] += 1
            return _Result(first_runs if idx == 0 else second_runs)
        return _Result([])

    session.execute = AsyncMock(side_effect=_execute)
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
# TestComputePriority
# ---------------------------------------------------------------------------


class TestComputePriority:
    def test_critical_level_returns_urgent(self):
        result = compute_priority(
            uncertainty_level="critical", anomaly_detected=False, uncertainty_score=0.9
        )
        assert result == "urgent"

    def test_high_level_returns_high(self):
        result = compute_priority(
            uncertainty_level="high", anomaly_detected=False, uncertainty_score=0.6
        )
        assert result == "high"

    def test_anomaly_bumps_to_high(self):
        result = compute_priority(
            uncertainty_level="medium", anomaly_detected=True, uncertainty_score=0.3
        )
        assert result == "high"

    def test_medium_level_returns_normal(self):
        result = compute_priority(
            uncertainty_level="medium", anomaly_detected=False, uncertainty_score=0.4
        )
        assert result == "normal"

    def test_low_level_high_score_returns_normal(self):
        result = compute_priority(
            uncertainty_level="low", anomaly_detected=False, uncertainty_score=0.40
        )
        assert result == "normal"

    def test_low_level_low_score_returns_low(self):
        result = compute_priority(
            uncertainty_level="low", anomaly_detected=False, uncertainty_score=0.10
        )
        assert result == "low"

    def test_empty_level_treated_as_low(self):
        result = compute_priority(
            uncertainty_level="", anomaly_detected=False, uncertainty_score=0.0
        )
        assert result == "low"

    def test_anomaly_with_critical_is_urgent(self):
        result = compute_priority(
            uncertainty_level="critical", anomaly_detected=True, uncertainty_score=0.95
        )
        assert result == "urgent"


# ---------------------------------------------------------------------------
# TestBuildReviewReasons
# ---------------------------------------------------------------------------


class TestBuildReviewReasons:
    def _call(self, **kwargs):
        defaults = dict(
            unresolved_conflicts=[],
            low_confidence_fields=[],
            anomaly_detected=False,
            anomaly_type=None,
            severity=None,
            uncertainty_level="low",
            uncertainty_score=0.2,
        )
        defaults.update(kwargs)
        return build_review_reasons(**defaults)

    def test_no_signals_returns_fallback(self):
        reasons = self._call()
        assert len(reasons) == 1
        assert "UncertaintyReportingAgent" in reasons[0]

    def test_unresolved_conflict_singular(self):
        reasons = self._call(
            unresolved_conflicts=[{"entity": "A", "severity": "high"}]
        )
        assert any("1 unresolved conflict" in r for r in reasons)

    def test_unresolved_conflicts_plural(self):
        conflicts = [{"entity": "A"}, {"entity": "B"}]
        reasons = self._call(unresolved_conflicts=conflicts)
        assert any("2 unresolved conflicts" in r for r in reasons)

    def test_low_confidence_fields_listed(self):
        reasons = self._call(low_confidence_fields=["credibility_score", "temporal_confidence"])
        assert any("credibility_score" in r for r in reasons)

    def test_anomaly_included_in_reasons(self):
        reasons = self._call(anomaly_detected=True, anomaly_type="drift", severity="medium")
        assert any("drift" in r and "medium" in r for r in reasons)

    def test_high_uncertainty_adds_score(self):
        reasons = self._call(uncertainty_level="high", uncertainty_score=0.72)
        assert any("0.72" in r for r in reasons)

    def test_critical_uncertainty_adds_score(self):
        reasons = self._call(uncertainty_level="critical", uncertainty_score=0.88)
        assert any("0.88" in r for r in reasons)

    def test_multiple_reasons_combined(self):
        reasons = self._call(
            unresolved_conflicts=[{"entity": "X"}],
            anomaly_detected=True,
            anomaly_type="spike",
            severity="high",
            uncertainty_level="high",
            uncertainty_score=0.65,
        )
        assert len(reasons) >= 2


# ---------------------------------------------------------------------------
# TestBuildReviewContext
# ---------------------------------------------------------------------------


class TestBuildReviewContext:
    def test_returns_dict(self):
        ctx = build_review_context({})
        assert isinstance(ctx, dict)

    def test_includes_uncertainty_level(self):
        ctx = build_review_context({"uncertainty_level": "medium"})
        assert ctx["uncertainty_level"] == "medium"

    def test_unresolved_conflict_count(self):
        enrichment = {
            "unresolved_conflicts": [{"entity": "A"}, {"entity": "B"}]
        }
        ctx = build_review_context(enrichment)
        assert ctx["unresolved_conflict_count"] == 2

    def test_audit_snapshot_included(self):
        enrichment = {
            "audit_log": [
                {"agent_name": "CredibilityAgent", "decision": "low cred", "confidence": 0.4},
                {"agent_name": "AnomalyDetectionAgent", "decision": "drift", "confidence": 0.7},
            ]
        }
        ctx = build_review_context(enrichment)
        assert "audit_snapshot" in ctx
        assert len(ctx["audit_snapshot"]) == 2

    def test_audit_snapshot_truncated_to_3(self):
        entries = [
            {"agent_name": f"Agent{i}", "decision": "d", "confidence": 0.5}
            for i in range(6)
        ]
        ctx = build_review_context({"audit_log": entries})
        assert len(ctx["audit_snapshot"]) == 3

    def test_empty_enrichment_defaults(self):
        ctx = build_review_context({})
        assert ctx["anomaly_detected"] is False
        assert ctx["low_confidence_fields"] == []


# ---------------------------------------------------------------------------
# TestRunHeuristic
# ---------------------------------------------------------------------------


class TestRunHeuristic:
    def test_not_eligible_when_no_review_recommended(self):
        result = run_heuristic("mem-1", {"review_recommended": False})
        assert result["queue_eligible"] is False
        assert result["queue_entry_id"] is None
        assert result["review_reasons"] == []
        assert result["review_context"] == {}

    def test_eligible_when_review_recommended(self):
        result = run_heuristic("mem-1", {
            "review_recommended": True,
            "uncertainty_level": "medium",
            "uncertainty_score": 0.40,
        })
        assert result["queue_eligible"] is True
        assert result["queue_entry_id"] == "mem-1"

    def test_eligible_via_anomaly_threshold(self):
        result = run_heuristic("mem-2", {
            "review_recommended": False,
            "anomaly_detected": True,
            "uncertainty_score": 0.25,
        })
        assert result["queue_eligible"] is True

    def test_not_eligible_via_anomaly_below_threshold(self):
        result = run_heuristic("mem-2", {
            "review_recommended": False,
            "anomaly_detected": True,
            "uncertainty_score": 0.10,
        })
        assert result["queue_eligible"] is False

    def test_priority_urgent_for_critical(self):
        result = run_heuristic("mem-1", {
            "review_recommended": True,
            "uncertainty_level": "critical",
            "uncertainty_score": 0.85,
        })
        assert result["priority"] == "urgent"

    def test_priority_high_for_high_level(self):
        result = run_heuristic("mem-1", {
            "review_recommended": True,
            "uncertainty_level": "high",
            "uncertainty_score": 0.60,
        })
        assert result["priority"] == "high"

    def test_confidence_in_range(self):
        result = run_heuristic("mem-1", {
            "review_recommended": True,
            "uncertainty_level": "medium",
            "uncertainty_score": 0.40,
        })
        assert 0.0 <= result["confidence"] <= 1.0

    def test_rationale_is_heuristic(self):
        result = run_heuristic("mem-1", {"review_recommended": True})
        assert result["rationale"] == "heuristic"

    def test_review_reasons_non_empty_when_eligible(self):
        result = run_heuristic("mem-1", {
            "review_recommended": True,
            "unresolved_conflicts": [{"entity": "X"}],
        })
        assert len(result["review_reasons"]) >= 1

    def test_non_eligible_has_low_priority(self):
        result = run_heuristic("mem-1", {"review_recommended": False})
        assert result["priority"] == "low"


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------


class TestValidateOutputs:
    def _valid_outputs(self):
        return {
            "queue_eligible": True,
            "priority": "normal",
            "review_reasons": ["reason A"],
            "review_context": {"uncertainty_level": "medium"},
            "queue_entry_id": "mem-1",
            "confidence": 0.75,
            "rationale": "heuristic",
        }

    def _make_result(self, outputs: dict) -> AgentResult:
        return AgentResult(
            agent_name="HumanReviewQueueAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=outputs.get("confidence", 0.75),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_outputs_passes(self):
        agent = HumanReviewQueueAgent()
        agent.validate_outputs(self._make_result(self._valid_outputs()))

    def test_missing_queue_eligible_raises(self):
        outputs = self._valid_outputs()
        del outputs["queue_eligible"]
        agent = HumanReviewQueueAgent()
        with pytest.raises(ValueError, match="queue_eligible"):
            agent.validate_outputs(self._make_result(outputs))

    def test_invalid_priority_raises(self):
        outputs = self._valid_outputs()
        outputs["priority"] = "extreme"
        agent = HumanReviewQueueAgent()
        with pytest.raises(ValueError, match="priority"):
            agent.validate_outputs(self._make_result(outputs))

    def test_review_reasons_not_list_raises(self):
        outputs = self._valid_outputs()
        outputs["review_reasons"] = "reason A"
        agent = HumanReviewQueueAgent()
        with pytest.raises(ValueError, match="review_reasons"):
            agent.validate_outputs(self._make_result(outputs))

    def test_review_context_not_dict_raises(self):
        outputs = self._valid_outputs()
        outputs["review_context"] = ["some", "list"]
        agent = HumanReviewQueueAgent()
        with pytest.raises(ValueError, match="review_context"):
            agent.validate_outputs(self._make_result(outputs))

    def test_confidence_out_of_range_raises(self):
        outputs = self._valid_outputs()
        outputs["confidence"] = 1.5
        agent = HumanReviewQueueAgent()
        with pytest.raises(ValueError, match="confidence"):
            agent.validate_outputs(self._make_result(outputs))

    def test_queue_entry_id_none_is_valid(self):
        outputs = self._valid_outputs()
        outputs["queue_entry_id"] = None
        agent = HumanReviewQueueAgent()
        agent.validate_outputs(self._make_result(outputs))

    def test_skips_validation_on_non_success(self):
        result = AgentResult(
            agent_name="HumanReviewQueueAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )
        HumanReviewQueueAgent().validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# TestAgentRun
# ---------------------------------------------------------------------------


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_heuristic_run_eligible(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx(enrichment={
            "review_recommended": True,
            "uncertainty_level": "medium",
            "uncertainty_score": 0.45,
        })
        with patch.object(agent, "_get_strategy", return_value="heuristic", create=True):
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["queue_eligible"] is True
        assert result.outputs["priority"] in ("low", "normal", "high", "urgent")

    @pytest.mark.asyncio
    async def test_heuristic_run_not_eligible(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx(enrichment={"review_recommended": False})
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["queue_eligible"] is False
        assert result.outputs["queue_entry_id"] is None

    @pytest.mark.asyncio
    async def test_run_returns_agent_result(self):
        agent = HumanReviewQueueAgent()
        result = await agent.run("mem-1", _ctx())
        assert isinstance(result, AgentResult)
        assert result.agent_name == "HumanReviewQueueAgent"
        assert result.agent_version == "v1"

    @pytest.mark.asyncio
    async def test_run_with_empty_enrichment(self):
        agent = HumanReviewQueueAgent()
        result = await agent.run("mem-1", _ctx(enrichment={}))
        assert result.status == "success"
        assert result.outputs["queue_eligible"] is False

    @pytest.mark.asyncio
    async def test_run_urgent_for_critical(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx(enrichment={
            "review_recommended": True,
            "uncertainty_level": "critical",
            "uncertainty_score": 0.90,
        })
        result = await agent.run("mem-1", ctx)
        assert result.outputs["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_run_includes_review_context(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx(enrichment={
            "review_recommended": True,
            "uncertainty_level": "high",
            "uncertainty_score": 0.60,
            "low_confidence_fields": ["credibility_score"],
        })
        result = await agent.run("mem-1", ctx)
        assert isinstance(result.outputs["review_context"], dict)

    @pytest.mark.asyncio
    async def test_llm_path_fallback_on_bad_response(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx(enrichment={
            "review_recommended": True,
            "uncertainty_level": "high",
            "uncertainty_score": 0.65,
        })

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"invalid": "response"})

        with patch("app.agents.human_review_queue_agent.create_ollama_client", return_value=mock_client):
            with patch("app.agents.human_review_queue_agent.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.HUMAN_REVIEW_QUEUE_STRATEGY = None
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "qwen2.5:7b"
                mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_settings.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_path_uses_llm_response(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx(enrichment={
            "review_recommended": True,
            "uncertainty_level": "high",
            "uncertainty_score": 0.65,
        })
        llm_resp = {
            "queue_eligible": True,
            "priority": "high",
            "review_reasons": ["llm reason"],
            "review_context": {"llm": True},
            "queue_entry_id": "mem-1",
            "confidence": 0.82,
            "rationale": "llm",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)

        with patch("app.agents.human_review_queue_agent.create_ollama_client", return_value=mock_client):
            with patch("app.agents.human_review_queue_agent.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.HUMAN_REVIEW_QUEUE_STRATEGY = None
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "qwen2.5:7b"
                mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_settings.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-1", ctx)

        assert result.outputs["rationale"] == "llm"
        assert result.outputs["priority"] == "high"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = HumanReviewQueueAgent()
        ctx = _ctx()
        ctx["runtime"]["job_id"] = "trace-xyz"
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "trace-xyz"

    @pytest.mark.asyncio
    async def test_dependencies_declared(self):
        agent = HumanReviewQueueAgent()
        deps = agent.dependencies()
        assert "UncertaintyReportingAgent" in deps
        assert "AnomalyDetectionAgent" in deps
        assert "AuditTrailAgent" in deps


# ---------------------------------------------------------------------------
# TestRegistryIntegration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_agent_in_registry(self):
        from app.agents.registry import get_agent
        agent = get_agent("human_review_queue")
        assert agent is not None
        assert isinstance(agent, HumanReviewQueueAgent)

    def test_agent_camel_case_lookup(self):
        from app.agents.registry import get_agent
        agent = get_agent("humanreviewqueueagent")
        assert agent is not None

    def test_agent_snake_case_lookup(self):
        from app.agents.registry import get_agent
        agent = get_agent("human_review_queue_agent")
        assert agent is not None


# ---------------------------------------------------------------------------
# TestQueueEndpoint
# ---------------------------------------------------------------------------


class TestQueueEndpoint:
    @pytest.mark.asyncio
    async def test_empty_queue_returns_200(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_returns_eligible_items(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="HumanReviewQueueAgent",
            outputs={
                "queue_eligible": True,
                "priority": "high",
                "review_reasons": ["1 unresolved conflict"],
                "review_context": {"uncertainty_level": "high"},
                "queue_entry_id": "mem-1",
                "confidence": 0.85,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["memory_id"] == "mem-1"
        assert items[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_filters_non_eligible_runs(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="HumanReviewQueueAgent",
            outputs={"queue_eligible": False, "priority": "low", "review_reasons": [],
                     "review_context": {}, "queue_entry_id": None, "confidence": 0.80},
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue", headers=_auth_headers())
        _clear()
        assert resp.json()["total"] == 0

    @pytest.mark.asyncio
    async def test_fallback_to_uncertainty_runs(self):
        unc_run = _FakeRun(
            id="ar-2", organization_id="org-1", memory_id="mem-2",
            agent_name="UncertaintyReportingAgent",
            outputs={
                "review_recommended": True,
                "uncertainty_level": "high",
                "uncertainty_score": 0.60,
                "unresolved_conflicts": [],
                "low_confidence_fields": [],
            },
        )
        _override(_mock_db([unc_run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue", headers=_auth_headers())
        _clear()
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["memory_id"] == "mem-2"

    @pytest.mark.asyncio
    async def test_priority_filter_query_param(self):
        runs = [
            _FakeRun(id="ar-1", organization_id="org-1", memory_id="mem-1",
                     agent_name="HumanReviewQueueAgent",
                     outputs={"queue_eligible": True, "priority": "high",
                              "review_reasons": ["r"], "review_context": {},
                              "queue_entry_id": "mem-1", "confidence": 0.85}),
            _FakeRun(id="ar-2", organization_id="org-1", memory_id="mem-2",
                     agent_name="HumanReviewQueueAgent",
                     outputs={"queue_eligible": True, "priority": "normal",
                              "review_reasons": ["r"], "review_context": {},
                              "queue_entry_id": "mem-2", "confidence": 0.75}),
        ]
        _override(_mock_db(runs))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue?priority=high", headers=_auth_headers())
        _clear()
        items = resp.json()["items"]
        assert all(i["priority"] == "high" for i in items)

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_deduplication_keeps_highest_priority(self):
        runs = [
            _FakeRun(id="ar-1", organization_id="org-1", memory_id="mem-1",
                     agent_name="HumanReviewQueueAgent",
                     outputs={"queue_eligible": True, "priority": "normal",
                              "review_reasons": ["r"], "review_context": {},
                              "queue_entry_id": "mem-1", "confidence": 0.75}),
            _FakeRun(id="ar-2", organization_id="org-1", memory_id="mem-1",
                     agent_name="HumanReviewQueueAgent",
                     outputs={"queue_eligible": True, "priority": "urgent",
                              "review_reasons": ["r2"], "review_context": {},
                              "queue_entry_id": "mem-1", "confidence": 0.90}),
        ]
        _override(_mock_db(runs))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue", headers=_auth_headers())
        _clear()
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_response_has_retrieved_at(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue", headers=_auth_headers())
        _clear()
        assert "retrieved_at" in resp.json()

    @pytest.mark.asyncio
    async def test_pagination_limit(self):
        runs = [
            _FakeRun(id=f"ar-{i}", organization_id="org-1", memory_id=f"mem-{i}",
                     agent_name="HumanReviewQueueAgent",
                     outputs={"queue_eligible": True, "priority": "normal",
                              "review_reasons": ["r"], "review_context": {},
                              "queue_entry_id": f"mem-{i}", "confidence": 0.75})
            for i in range(5)
        ]
        _override(_mock_db(runs))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/v1/review/queue?limit=2", headers=_auth_headers())
        _clear()
        assert len(resp.json()["items"]) == 2
        assert resp.json()["total"] == 5


# ---------------------------------------------------------------------------
# TestClaimEndpoint
# ---------------------------------------------------------------------------


class TestClaimEndpoint:
    @pytest.mark.asyncio
    async def test_claim_returns_200_for_eligible_memory(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="HumanReviewQueueAgent",
            outputs={
                "queue_eligible": True,
                "priority": "high",
                "review_reasons": ["1 unresolved conflict"],
                "review_context": {"uncertainty_level": "high"},
                "queue_entry_id": "mem-1",
                "confidence": 0.85,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/claim",
                headers=_auth_headers(),
                json={"memory_id": "mem-1"},
            )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["memory_id"] == "mem-1"
        assert body["priority"] == "high"
        assert "claimed_at" in body

    @pytest.mark.asyncio
    async def test_claim_404_when_not_in_queue(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="HumanReviewQueueAgent",
            outputs={"queue_eligible": False, "priority": "low", "review_reasons": [],
                     "review_context": {}, "queue_entry_id": None, "confidence": 0.80},
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/claim",
                headers=_auth_headers(),
                json={"memory_id": "mem-1"},
            )
        _clear()
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_claim_fallback_to_uncertainty_agent(self):
        unc_run = _FakeRun(
            id="ar-2", organization_id="org-1", memory_id="mem-2",
            agent_name="UncertaintyReportingAgent",
            outputs={
                "review_recommended": True,
                "uncertainty_level": "high",
                "uncertainty_score": 0.65,
                "unresolved_conflicts": [],
                "low_confidence_fields": ["credibility_score"],
            },
        )
        # First query (HumanReviewQueueAgent) returns nothing; second (UncertaintyReportingAgent) returns the run
        _override(_mock_db_two_queries([], [unc_run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/claim",
                headers=_auth_headers(),
                json={"memory_id": "mem-2"},
            )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["priority"] == "high"

    @pytest.mark.asyncio
    async def test_claim_returns_review_context(self):
        run = _FakeRun(
            id="ar-1", organization_id="org-1", memory_id="mem-1",
            agent_name="HumanReviewQueueAgent",
            outputs={
                "queue_eligible": True,
                "priority": "normal",
                "review_reasons": ["reason"],
                "review_context": {"uncertainty_level": "medium", "uncertainty_score": 0.4},
                "queue_entry_id": "mem-1",
                "confidence": 0.75,
            },
        )
        _override(_mock_db([run]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/claim",
                headers=_auth_headers(),
                json={"memory_id": "mem-1"},
            )
        _clear()
        ctx = resp.json()["review_context"]
        assert isinstance(ctx, dict)

    @pytest.mark.asyncio
    async def test_claim_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/v1/review/claim", json={"memory_id": "mem-1"})
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_claim_no_runs_404(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/claim",
                headers=_auth_headers(),
                json={"memory_id": "mem-999"},
            )
        _clear()
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestResolveEndpoint
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    @pytest.mark.asyncio
    async def test_approve_returns_201(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "verdict": "approve"},
            )
        _clear()
        assert resp.status_code == 201
        body = resp.json()
        assert body["memory_id"] == "mem-1"
        assert body["verdict"] == "approve"
        assert "feedback_id" in body
        assert "resolved_at" in body

    @pytest.mark.asyncio
    async def test_reject_returns_201(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "verdict": "reject"},
            )
        _clear()
        assert resp.status_code == 201
        assert resp.json()["verdict"] == "reject"

    @pytest.mark.asyncio
    async def test_partial_verdict_accepted(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "verdict": "partial", "notes": "needs clarification"},
            )
        _clear()
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_null_verdict_accepted(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1"},
            )
        _clear()
        assert resp.status_code == 201
        assert resp.json()["verdict"] is None

    @pytest.mark.asyncio
    async def test_with_credibility_override(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "verdict": "approve", "credibility_override": 0.95},
            )
        _clear()
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_credibility_override_out_of_range_422(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "credibility_override": 1.5},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_feedback_id_is_string(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "verdict": "approve"},
            )
        _clear()
        assert isinstance(resp.json()["feedback_id"], str)
        assert len(resp.json()["feedback_id"]) > 0

    @pytest.mark.asyncio
    async def test_401_without_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                json={"memory_id": "mem-1", "verdict": "approve"},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_with_resolution_confirmed(self):
        _override(_mock_db([]))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/v1/review/resolve",
                headers=_auth_headers(),
                json={"memory_id": "mem-1", "verdict": "approve", "resolution_confirmed": True},
            )
        _clear()
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# TestRouteRegistration
# ---------------------------------------------------------------------------


class TestRouteRegistration:
    def _route_paths(self):
        return {r.path for r in app.routes}

    def test_queue_route_registered(self):
        paths = self._route_paths()
        assert any("queue" in p for p in paths)

    def test_claim_route_registered(self):
        paths = self._route_paths()
        assert any("claim" in p for p in paths)

    def test_resolve_route_registered(self):
        paths = self._route_paths()
        assert any("resolve" in p for p in paths)
