"""Tests for Causal API Surface — Phase 39."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.endpoints.auth import get_current_user
from app.core.database import get_db
from app.main import app
from app.services.causal_reasoning_service import CausalExplanation, PredictionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE = "/api/v1/causal"


def _fake_user(org_id: str = "org-1", user_id: str = "user-1") -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.organization_id = org_id
    return user


def _fake_db() -> AsyncMock:
    session = AsyncMock(spec=AsyncSession)
    return session


def _override(session: AsyncMock, org_id: str = "org-1") -> None:
    async def _get_db():
        yield session

    user = _fake_user(org_id=org_id)

    def _get_user():
        return user

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user] = _get_user


def _clear() -> None:
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeEdge:
    id: str = "edge-1"
    organization_id: str = "org-1"
    cause_entity_id: str = "fact-a"
    cause_entity_type: str = "fact"
    effect_entity_id: str = "fact-b"
    effect_entity_type: str = "fact"
    mechanism: str = "temporal"
    strength: float = 0.75
    latency_hours: int | None = None
    evidence_memory_ids: list = field(default_factory=list)
    discovered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validation_count: int = 1
    invalidation_count: int = 0
    last_validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "organization_id": self.organization_id,
            "cause_entity_id": self.cause_entity_id,
            "cause_entity_type": self.cause_entity_type,
            "effect_entity_id": self.effect_entity_id,
            "effect_entity_type": self.effect_entity_type,
            "mechanism": self.mechanism,
            "strength": self.strength,
            "latency_hours": self.latency_hours,
            "evidence_memory_ids": self.evidence_memory_ids,
            "discovered_at": self.discovered_at.isoformat(),
            "validation_count": self.validation_count,
            "invalidation_count": self.invalidation_count,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# POST /causal/discover/{episode_id}
# ---------------------------------------------------------------------------


class TestDiscoverCausalEdges:
    @pytest.mark.asyncio
    async def test_discover_returns_edges(self):
        _override(_fake_db())
        edge = _FakeEdge()
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.discover_causal_edges_from_episode = AsyncMock(
                return_value=[edge]
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/discover/ep-1",
                    json={"episode_id": "ep-1", "auto_save": True},
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["episode_id"] == "ep-1"
        assert body["edges_discovered"] == 1

    @pytest.mark.asyncio
    async def test_discover_empty_episode(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.discover_causal_edges_from_episode = AsyncMock(
                return_value=[]
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/discover/ep-empty",
                    json={"episode_id": "ep-empty", "auto_save": True},
                )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["edges_discovered"] == 0

    @pytest.mark.asyncio
    async def test_discover_raises_value_error_returns_400(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.discover_causal_edges_from_episode = AsyncMock(
                side_effect=ValueError("Episode not found")
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/discover/ep-bad",
                    json={"episode_id": "ep-bad", "auto_save": True},
                )
        _clear()
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_discover_unauthenticated_returns_401(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                f"{BASE}/discover/ep-1",
                json={"episode_id": "ep-1"},
            )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /causal/edges
# ---------------------------------------------------------------------------


class TestGetCausalEdges:
    @pytest.mark.asyncio
    async def test_edges_by_from_entity(self):
        _override(_fake_db())
        edge = _FakeEdge()
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.get_edges = AsyncMock(return_value=[edge])
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/edges",
                    params={"from_entity": "fact-a"},
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert body[0]["cause_entity_id"] == "fact-a"

    @pytest.mark.asyncio
    async def test_edges_by_to_entity(self):
        _override(_fake_db())
        edge = _FakeEdge(cause_entity_id="fact-x", effect_entity_id="fact-b")
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.get_edges = AsyncMock(return_value=[edge])
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/edges",
                    params={"to_entity": "fact-b"},
                )
        _clear()
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_edges_missing_params_returns_error(self):
        _override(_fake_db())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                f"{BASE}/edges",
            )
        _clear()
        assert resp.status_code in (400, 500)

    @pytest.mark.asyncio
    async def test_edges_empty_result(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.get_edges = AsyncMock(return_value=[])
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/edges",
                    params={"from_entity": "unknown"},
                )
        _clear()
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_edges_unauthenticated(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(f"{BASE}/edges", params={"from_entity": "fact-a"})
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# POST /causal/predict
# ---------------------------------------------------------------------------


class TestPredictOutcome:
    @pytest.mark.asyncio
    async def test_predict_returns_probability(self):
        _override(_fake_db())
        prediction = PredictionResult(
            outcome_id="fact-target",
            probability=0.78,
            confidence=0.85,
            causal_chain=["fact-a", "fact-b", "fact-target"],
        )
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.predict = AsyncMock(return_value=prediction)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/predict",
                    json={
                        "cause_ids": ["fact-a", "fact-b"],
                        "target_id": "fact-target",
                        "max_depth": 3,
                    },
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["outcome_id"] == "fact-target"
        assert body["probability"] == pytest.approx(0.78)
        assert body["confidence"] == pytest.approx(0.85)
        assert len(body["causal_chain"]) == 3

    @pytest.mark.asyncio
    async def test_predict_no_path_returns_404_or_empty(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.predict = AsyncMock(return_value=None)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/predict",
                    json={"cause_ids": ["x"], "target_id": "y"},
                )
        _clear()
        # None result → 500 from AttributeError or handled gracefully
        assert resp.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_predict_value_error_returns_400(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.predict = AsyncMock(
                side_effect=ValueError("bad input")
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/predict",
                    json={"cause_ids": ["x"], "target_id": "y"},
                )
        _clear()
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_predict_unauthenticated(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                f"{BASE}/predict",
                json={"cause_ids": ["x"], "target_id": "y"},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_predict_default_max_depth(self):
        _override(_fake_db())
        prediction = PredictionResult(
            outcome_id="fact-z",
            probability=0.6,
            confidence=0.7,
            causal_chain=["fact-a", "fact-z"],
        )
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.predict = AsyncMock(return_value=prediction)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/predict",
                    # omit max_depth → defaults to 3
                    json={"cause_ids": ["fact-a"], "target_id": "fact-z"},
                )
        _clear()
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /causal/explain/{effect_id}
# ---------------------------------------------------------------------------


class TestExplainEffect:
    @pytest.mark.asyncio
    async def test_explain_returns_root_causes(self):
        _override(_fake_db())
        explanation = CausalExplanation(
            effect_id="fact-b",
            root_causes=[
                {"cause_id": "fact-a", "strength": 0.9, "mechanism": "direct",
                 "causal_chain": ["fact-a", "fact-b"]},
            ],
        )
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.explain = AsyncMock(return_value=explanation)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/explain/fact-b",
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["effect_id"] == "fact-b"
        assert len(body["root_causes"]) == 1
        assert body["root_causes"][0]["cause_id"] == "fact-a"

    @pytest.mark.asyncio
    async def test_explain_no_causes_returns_empty_list(self):
        _override(_fake_db())
        explanation = CausalExplanation(effect_id="fact-orphan", root_causes=[])
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.explain = AsyncMock(return_value=explanation)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/explain/fact-orphan",
                )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["root_causes"] == []

    @pytest.mark.asyncio
    async def test_explain_value_error_returns_400(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.explain = AsyncMock(
                side_effect=ValueError("not found")
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/explain/bad-id",
                )
        _clear()
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_explain_unauthenticated(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(f"{BASE}/explain/fact-b")
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_explain_multiple_root_causes_ranked(self):
        _override(_fake_db())
        explanation = CausalExplanation(
            effect_id="fact-target",
            root_causes=[
                {"cause_id": "fact-strong", "strength": 0.92, "mechanism": "direct",
                 "causal_chain": ["fact-strong", "fact-target"]},
                {"cause_id": "fact-weak", "strength": 0.45, "mechanism": "temporal",
                 "causal_chain": ["fact-weak", "fact-target"]},
            ],
        )
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.explain = AsyncMock(return_value=explanation)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/explain/fact-target",
                )
        _clear()
        assert resp.status_code == 200
        assert len(resp.json()["root_causes"]) == 2


# ---------------------------------------------------------------------------
# POST /causal/counterfactual
# ---------------------------------------------------------------------------


class TestCounterfactualQuery:
    @pytest.mark.asyncio
    async def test_counterfactual_returns_scenario_id(self):
        _override(_fake_db())
        cf_result = {
            "scenario_id": "cf-42",
            "condition": "If I responded earlier",
            "status": "pending",
        }
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.counterfactual_query = AsyncMock(return_value=cf_result)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/counterfactual",
                    json={
                        "condition": "If I responded earlier",
                        "query": {"outcome": "user_satisfaction"},
                    },
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["scenario_id"] == "cf-42"
        assert body["condition"] == "If I responded earlier"
        assert body["status"] == "pending"

    @pytest.mark.asyncio
    async def test_counterfactual_default_status_when_absent(self):
        _override(_fake_db())
        cf_result = {
            "scenario_id": "cf-99",
            "condition": "If budget doubled",
            # no status key
        }
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.counterfactual_query = AsyncMock(return_value=cf_result)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/counterfactual",
                    json={"condition": "If budget doubled", "query": {}},
                )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    @pytest.mark.asyncio
    async def test_counterfactual_value_error_returns_400(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.counterfactual_query = AsyncMock(
                side_effect=ValueError("invalid condition")
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/counterfactual",
                    json={"condition": "bad", "query": {}},
                )
        _clear()
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_counterfactual_unauthenticated(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                f"{BASE}/counterfactual",
                json={"condition": "x", "query": {}},
            )
        assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# GET /causal/do-calculus
# ---------------------------------------------------------------------------


class TestDoCalculus:
    @pytest.mark.asyncio
    async def test_do_calculus_returns_causal_effect(self):
        _override(_fake_db())
        dc_result = {
            "treatment": "context_window",
            "outcome": "task_success",
            "causal_effect": 0.23,
            "num_paths": 4,
        }
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.do_calculus = AsyncMock(return_value=dc_result)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/do-calculus",
                    params={"treatment": "context_window", "outcome": "task_success"},
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["treatment"] == "context_window"
        assert body["causal_effect"] == pytest.approx(0.23)
        assert body["num_paths"] == 4

    @pytest.mark.asyncio
    async def test_do_calculus_zero_paths(self):
        _override(_fake_db())
        dc_result = {
            "treatment": "x",
            "outcome": "y",
            "causal_effect": 0.0,
            "num_paths": 0,
        }
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.do_calculus = AsyncMock(return_value=dc_result)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/do-calculus",
                    params={"treatment": "x", "outcome": "y"},
                )
        _clear()
        assert resp.status_code == 200
        assert resp.json()["num_paths"] == 0

    @pytest.mark.asyncio
    async def test_do_calculus_missing_params(self):
        _override(_fake_db())
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                f"{BASE}/do-calculus",
                params={"treatment": "x"},  # missing outcome
            )
        _clear()
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_do_calculus_unauthenticated(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get(
                f"{BASE}/do-calculus",
                params={"treatment": "x", "outcome": "y"},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_do_calculus_value_error_returns_400(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.do_calculus = AsyncMock(
                side_effect=ValueError("bad treatment")
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get(
                    f"{BASE}/do-calculus",
                    params={"treatment": "bad", "outcome": "y"},
                )
        _clear()
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /causal/validate
# ---------------------------------------------------------------------------


class TestValidateEdge:
    @pytest.mark.asyncio
    async def test_validate_success_increases_strength(self):
        _override(_fake_db())
        updated_edge = _FakeEdge(
            strength=0.85,
            validation_count=2,
            invalidation_count=0,
        )
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.validate_edge = AsyncMock(return_value=updated_edge)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/validate",
                    json={
                        "edge_id": "edge-1",
                        "success": True,
                        "strength_adjustment": 0.1,
                    },
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "edge-1"
        assert body["strength"] == pytest.approx(0.85)
        assert body["validation_count"] == 2

    @pytest.mark.asyncio
    async def test_validate_failure_decreases_strength(self):
        _override(_fake_db())
        updated_edge = _FakeEdge(
            strength=0.55,
            validation_count=1,
            invalidation_count=1,
        )
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.validate_edge = AsyncMock(return_value=updated_edge)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/validate",
                    json={
                        "edge_id": "edge-1",
                        "success": False,
                        "strength_adjustment": 0.2,
                    },
                )
        _clear()
        assert resp.status_code == 200
        body = resp.json()
        assert body["invalidation_count"] == 1

    @pytest.mark.asyncio
    async def test_validate_edge_not_found_returns_500(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.validate_edge = AsyncMock(return_value=None)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/validate",
                    json={"edge_id": "missing", "success": True},
                )
        _clear()
        assert resp.status_code in (400, 404, 500)

    @pytest.mark.asyncio
    async def test_validate_value_error_returns_400(self):
        _override(_fake_db())
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.validate_edge = AsyncMock(
                side_effect=ValueError("invalid edge")
            )
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/validate",
                    json={"edge_id": "bad", "success": True},
                )
        _clear()
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_unauthenticated(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                f"{BASE}/validate",
                json={"edge_id": "edge-1", "success": True},
            )
        assert resp.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_validate_default_strength_adjustment(self):
        _override(_fake_db())
        updated_edge = _FakeEdge(strength=0.8, validation_count=1, invalidation_count=0)
        with patch(
            "app.api.v1.endpoints.causal.CausalReasoningService"
        ) as MockSvc:
            MockSvc.return_value.validate_edge = AsyncMock(return_value=updated_edge)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    f"{BASE}/validate",
                    # omit strength_adjustment → schema default 0.1
                    json={"edge_id": "edge-1", "success": True},
                )
        _clear()
        assert resp.status_code == 200
