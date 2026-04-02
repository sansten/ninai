from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.episodic_future_simulation_agent import (
    EpisodicFutureSimulationAgent,
    _dominant_tag,
    _episode_text,
    _extract_entities,
    _overlap_score,
    _severity_change,
    _tokenize,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


def _ep(content: str, tags: list[str] | None = None, event_description: str | None = None) -> dict:
    return {
        "content": content,
        "tags": tags or [],
        "event_description": event_description or content,
    }


def _ctx(
    *,
    current_state: dict,
    planned_action: str,
    historical_episodes: list[dict],
    simulation_steps: int = 3,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "current_state": current_state,
                "planned_action": planned_action,
                "historical_episodes": historical_episodes,
                "simulation_steps": simulation_steps,
            }
        },
        "runtime": {"job_id": "trace-67"},
    }


class TestHelpers:
    def test_tokenize_lower(self):
        assert "deploy" in _tokenize("Deploy")

    def test_episode_text_joins_fields(self):
        text = _episode_text({"content": "a", "summary": "b", "event_description": "c", "tags": ["d"]})
        assert all(t in text for t in ["a", "b", "c", "d"])

    def test_overlap_score_zero_on_empty(self):
        assert _overlap_score(set(), {"a"}) == 0.0

    def test_overlap_score_non_zero(self):
        assert _overlap_score({"a", "b"}, {"b", "c"}) > 0.0

    def test_dominant_tag(self):
        assert _dominant_tag({"tags": ["Ops", "DB"]}) == "ops"

    def test_severity_increase(self):
        assert _severity_change("critical outage alert") == "increase"

    def test_severity_decrease(self):
        assert _severity_change("incident resolved and fixed") == "decrease"

    def test_severity_stable(self):
        assert _severity_change("normal operation") == "stable"

    def test_extract_entities_from_state(self):
        entities = _extract_entities({"entities": ["svc-a", "svc-b"]}, {})
        assert entities == ["svc-a", "svc-b"]

    def test_extract_entities_from_tags_when_state_empty(self):
        entities = _extract_entities({"entities": []}, {"tags": ["db", "api"]})
        assert entities == ["db", "api"]


class TestHeuristic:
    def test_no_matching_history_uses_generic_steps(self):
        out = run_heuristic(
            current_state={"entities": ["svc-a"]},
            planned_action="rotate credentials",
            historical_episodes=[_ep("unrelated billing report")],
            simulation_steps=3,
        )
        assert len(out["simulated_episodes"]) == 3
        assert "Likely follow-up" in out["simulated_episodes"][1]["event_description"]

    def test_historical_match_uses_sequel_template_for_step1(self):
        history = [
            _ep("deploy fix for api latency", ["api"]),
            _ep("critical alert on api after deploy", ["api"]),
        ]
        out = run_heuristic(
            current_state={"entities": ["api"]},
            planned_action="deploy fix for api latency",
            historical_episodes=history,
            simulation_steps=3,
        )
        assert out["simulated_episodes"][1]["event_description"] == "critical alert on api after deploy"

    def test_probability_decreases_by_step(self):
        out = run_heuristic(
            current_state={},
            planned_action="restart service",
            historical_episodes=[],
            simulation_steps=4,
        )
        probs = [e["probability"] for e in out["simulated_episodes"]]
        assert probs[0] == 0.9
        assert probs[1] > probs[2] > probs[3]

    def test_risk_events_filter_increase_and_prob_gt_point_five(self):
        history = [
            _ep("restart service", ["svc"]),
            _ep("critical error on svc", ["svc"]),
        ]
        out = run_heuristic(
            current_state={"entities": ["svc"]},
            planned_action="restart service",
            historical_episodes=history,
            simulation_steps=2,
        )
        assert len(out["risk_events"]) == 1
        assert out["risk_events"][0]["severity_change"] == "increase"
        assert out["risk_events"][0]["probability"] > 0.5

    def test_success_probability_reduced_by_risk(self):
        history = [_ep("restart service"), _ep("critical failure down")]
        out = run_heuristic(
            current_state={},
            planned_action="restart service",
            historical_episodes=history,
            simulation_steps=2,
        )
        assert out["success_probability"] < 0.9

    def test_precautions_non_empty_when_risk_events_non_empty(self):
        history = [_ep("restart service"), _ep("critical failure down")]
        out = run_heuristic(
            current_state={"entities": ["svc"]},
            planned_action="restart service",
            historical_episodes=history,
            simulation_steps=2,
        )
        assert len(out["recommended_precautions"]) > 0

    def test_simulation_steps_one_only_step_zero(self):
        out = run_heuristic(
            current_state={},
            planned_action="restart",
            historical_episodes=[],
            simulation_steps=1,
        )
        assert len(out["simulated_episodes"]) == 1
        assert out["simulated_episodes"][0]["step"] == 0

    def test_confidence_constant(self):
        out = run_heuristic(current_state={}, planned_action="x", historical_episodes=[], simulation_steps=1)
        assert out["confidence"] == 0.6

    def test_dedupe_templates_by_dominant_tag(self):
        history = [
            _ep("deploy api fix", ["api"]),
            _ep("alert api one", ["api"]),
            _ep("deploy api update", ["api"]),
            _ep("alert api two", ["api"]),
        ]
        out = run_heuristic(
            current_state={"entities": ["api"]},
            planned_action="deploy api fix",
            historical_episodes=history,
            simulation_steps=3,
        )
        assert out["simulated_episodes"][1]["event_description"] == "alert api one"

    def test_step_zero_entities_from_state(self):
        out = run_heuristic(
            current_state={"entities": ["db", "api"]},
            planned_action="action",
            historical_episodes=[],
            simulation_steps=1,
        )
        assert out["simulated_episodes"][0]["entities_affected"] == ["db", "api"]

    def test_entities_from_tags_when_state_missing(self):
        history = [_ep("restart", ["ops"]), _ep("critical error", ["ops", "db"])]
        out = run_heuristic(current_state={}, planned_action="restart", historical_episodes=history, simulation_steps=2)
        assert out["simulated_episodes"][1]["entities_affected"] == ["ops", "db"]

    def test_simulation_steps_minimum_one(self):
        out = run_heuristic(current_state={}, planned_action="x", historical_episodes=[], simulation_steps=0)
        assert len(out["simulated_episodes"]) == 1

    def test_success_probability_bounds(self):
        out = run_heuristic(current_state={}, planned_action="x", historical_episodes=[], simulation_steps=3)
        assert 0.0 <= out["success_probability"] <= 1.0


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = EpisodicFutureSimulationAgent()
        with patch("app.agents.episodic_future_simulation_agent.settings") as mock_settings:
            mock_settings.EPISODIC_FUTURE_SIMULATION_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(current_state={}, planned_action="x", historical_episodes=[]))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = EpisodicFutureSimulationAgent()
        with patch("app.agents.episodic_future_simulation_agent.settings") as mock_settings:
            mock_settings.EPISODIC_FUTURE_SIMULATION_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(current_state={}, planned_action="x", historical_episodes=[]))
        assert result.trace_id == "trace-67"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = EpisodicFutureSimulationAgent()
        with patch("app.agents.episodic_future_simulation_agent.settings") as mock_settings:
            mock_settings.EPISODIC_FUTURE_SIMULATION_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(current_state={}, planned_action="x", historical_episodes=[]))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = EpisodicFutureSimulationAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "simulated_episodes": [],
                "success_probability": 0.5,
                "risk_events": [],
                "recommended_precautions": [],
                "confidence": 0.6,
                "rationale": "llm",
            }
        )
        with patch("app.agents.episodic_future_simulation_agent.settings") as mock_settings, patch(
            "app.agents.episodic_future_simulation_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.EPISODIC_FUTURE_SIMULATION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(current_state={}, planned_action="x", historical_episodes=[]))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = EpisodicFutureSimulationAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.episodic_future_simulation_agent.settings") as mock_settings, patch(
            "app.agents.episodic_future_simulation_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.EPISODIC_FUTURE_SIMULATION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(current_state={}, planned_action="x", historical_episodes=[]))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="EpisodicFutureSimulationAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.6,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "simulated_episodes": [],
            "success_probability": 0.5,
            "risk_events": [],
            "recommended_precautions": [],
            "confidence": 0.6,
        }

    def test_validate_outputs_passes(self):
        EpisodicFutureSimulationAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_simulated_episodes_type_raises(self):
        with pytest.raises(ValueError, match="simulated_episodes"):
            EpisodicFutureSimulationAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), simulated_episodes="x"))
            )

    def test_success_probability_type_raises(self):
        with pytest.raises(ValueError, match="success_probability"):
            EpisodicFutureSimulationAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), success_probability="x"))
            )

    def test_success_probability_range_raises(self):
        with pytest.raises(ValueError, match="success_probability"):
            EpisodicFutureSimulationAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), success_probability=1.1))
            )

    def test_risk_events_type_raises(self):
        with pytest.raises(ValueError, match="risk_events"):
            EpisodicFutureSimulationAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), risk_events="x"))
            )

    def test_precautions_type_raises(self):
        with pytest.raises(ValueError, match="recommended_precautions"):
            EpisodicFutureSimulationAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), recommended_precautions="x"))
            )

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="EpisodicFutureSimulationAgent",
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
        EpisodicFutureSimulationAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("episodic_future_simulation"), EpisodicFutureSimulationAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("episodicfuturesimulation"), EpisodicFutureSimulationAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("EpisodicFutureSimulationAgent"), EpisodicFutureSimulationAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("future_simulation"), EpisodicFutureSimulationAgent)
