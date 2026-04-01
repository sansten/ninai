from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.registry import get_agent
from app.agents.self_improvement_planner_agent import (
    SelfImprovementPlannerAgent,
    _proposal_from_failure_type,
    run_heuristic,
)
from app.agents.types import AgentResult
from app.models.improvement_proposal import ImprovementProposal


def _ctx(
    *,
    performance_metrics: list[dict],
    failure_records: list[dict],
    current_config: dict,
    improvement_threshold: float = 0.15,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "performance_metrics": performance_metrics,
                "failure_records": failure_records,
                "current_config": current_config,
                "improvement_threshold": improvement_threshold,
            }
        },
        "runtime": {"job_id": "trace-60"},
    }


class TestHeuristicHelpers:
    def test_timeout_maps_to_data_preprocessing(self):
        ptype, _, _ = _proposal_from_failure_type(failure_rate=0.4, errors=["timeout"])
        assert ptype == "data_preprocessing"

    def test_graph_too_large_maps_to_data_preprocessing(self):
        ptype, _, _ = _proposal_from_failure_type(failure_rate=0.4, errors=["graph_too_large"])
        assert ptype == "data_preprocessing"

    def test_low_confidence_maps_to_parameter_tune(self):
        ptype, _, _ = _proposal_from_failure_type(failure_rate=0.4, errors=["low_confidence"])
        assert ptype == "parameter_tune"

    def test_unknown_maps_to_routing_change(self):
        ptype, _, _ = _proposal_from_failure_type(failure_rate=0.4, errors=["misc_error"])
        assert ptype == "routing_change"

    def test_expected_gain_clamped(self):
        _, _, gain = _proposal_from_failure_type(failure_rate=0.9, errors=["timeout"])
        assert gain == 0.5


class TestHeuristicCore:
    def test_failure_rate_zero_no_proposal(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 0.0, "avg_confidence": 0.9, "sample_count": 10}],
            failure_records=[],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["proposals"] == []

    def test_timeout_errors_create_data_preprocessing_proposal(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 0.3}],
            failure_records=[{"agent_name": "A", "error_type": "timeout"}],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["proposals"][0]["proposal_type"] == "data_preprocessing"

    def test_low_confidence_errors_create_parameter_tune_proposal(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "B", "failure_rate": 0.3}],
            failure_records=[{"agent_name": "B", "error_type": "low_confidence"}],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["proposals"][0]["proposal_type"] == "parameter_tune"

    def test_other_errors_create_routing_change_proposal(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "C", "failure_rate": 0.3}],
            failure_records=[{"agent_name": "C", "error_type": "unknown"}],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["proposals"][0]["proposal_type"] == "routing_change"

    def test_improvement_threshold_filters_low_gain(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 0.21}],
            failure_records=[{"agent_name": "A", "error_type": "timeout"}],
            current_config={},
            improvement_threshold=0.4,
        )
        assert out["proposals"] == []

    def test_high_priority_subset_of_proposals(self):
        out = run_heuristic(
            performance_metrics=[
                {"agent_name": "A", "failure_rate": 0.25},
                {"agent_name": "B", "failure_rate": 0.35},
            ],
            failure_records=[
                {"agent_name": "A", "error_type": "timeout"},
                {"agent_name": "B", "error_type": "timeout"},
            ],
            current_config={},
            improvement_threshold=0.15,
        )
        high = out["high_priority_proposals"]
        props = out["proposals"]
        assert all(h in props for h in high)

    def test_system_health_score_all_zero_failure_rates(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 0.0}],
            failure_records=[],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["system_health_score"] == 1.0

    def test_system_health_score_all_one_failure_rates(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 1.0}, {"agent_name": "B", "failure_rate": 1.0}],
            failure_records=[],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["system_health_score"] == 0.0

    def test_confidence_has_proposals(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 0.3}],
            failure_records=[{"agent_name": "A", "error_type": "timeout"}],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["confidence"] == 0.7

    def test_confidence_no_proposals(self):
        out = run_heuristic(
            performance_metrics=[{"agent_name": "A", "failure_rate": 0.1}],
            failure_records=[],
            current_config={},
            improvement_threshold=0.15,
        )
        assert out["confidence"] == 0.5


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic_success(self):
        agent = SelfImprovementPlannerAgent()
        with patch("app.agents.self_improvement_planner_agent.settings") as mock_settings:
            mock_settings.SELF_IMPROVEMENT_PLANNER_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(performance_metrics=[], failure_records=[], current_config={}))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = SelfImprovementPlannerAgent()
        with patch("app.agents.self_improvement_planner_agent.settings") as mock_settings:
            mock_settings.SELF_IMPROVEMENT_PLANNER_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(performance_metrics=[], failure_records=[], current_config={}))
        assert result.trace_id == "trace-60"

    @pytest.mark.asyncio
    async def test_strategy_fallback_uses_agent_strategy(self):
        agent = SelfImprovementPlannerAgent()
        with patch("app.agents.self_improvement_planner_agent.settings") as mock_settings:
            mock_settings.SELF_IMPROVEMENT_PLANNER_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(performance_metrics=[], failure_records=[], current_config={}))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = SelfImprovementPlannerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "proposals": [],
                "high_priority_proposals": [],
                "system_health_score": 0.8,
                "confidence": 0.6,
                "rationale": "llm",
            }
        )
        with patch("app.agents.self_improvement_planner_agent.settings") as mock_settings, patch(
            "app.agents.self_improvement_planner_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.SELF_IMPROVEMENT_PLANNER_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(performance_metrics=[], failure_records=[], current_config={}))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = SelfImprovementPlannerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.self_improvement_planner_agent.settings") as mock_settings, patch(
            "app.agents.self_improvement_planner_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.SELF_IMPROVEMENT_PLANNER_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(performance_metrics=[], failure_records=[], current_config={}))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="SelfImprovementPlannerAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "proposals": [],
            "high_priority_proposals": [],
            "system_health_score": 0.8,
            "confidence": 0.7,
        }

    def test_validate_outputs_passes(self):
        SelfImprovementPlannerAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_invalid_proposals_type_raises(self):
        with pytest.raises(ValueError, match="proposals"):
            SelfImprovementPlannerAgent().validate_outputs(self._result(dict(self._valid_outputs(), proposals="x")))

    def test_invalid_high_priority_type_raises(self):
        with pytest.raises(ValueError, match="high_priority_proposals"):
            SelfImprovementPlannerAgent().validate_outputs(self._result(dict(self._valid_outputs(), high_priority_proposals="x")))

    def test_invalid_system_health_score_raises(self):
        with pytest.raises(ValueError, match="system_health_score"):
            SelfImprovementPlannerAgent().validate_outputs(self._result(dict(self._valid_outputs(), system_health_score=1.2)))

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="SelfImprovementPlannerAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        SelfImprovementPlannerAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("self_improvement_planner"), SelfImprovementPlannerAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("selfimprovementplanner"), SelfImprovementPlannerAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("SelfImprovementPlannerAgent"), SelfImprovementPlannerAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("improvement_planner"), SelfImprovementPlannerAgent)


class TestModelShape:
    def test_model_has_required_fields(self):
        fields = ImprovementProposal.__table__.columns.keys()
        for name in ("org_id", "target_agent", "proposal_type", "description", "evidence", "expected_gain", "status", "created_at"):
            assert name in fields
