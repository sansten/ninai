"""Tests for MetaCognitivePlanningAgent — Phase 36."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.agents.meta_cognitive_planning_agent import (
    MetaCognitivePlanningAgent,
    _VALID_MODES,
    _DEFAULT_CONFIDENCE_THRESHOLD,
    _DEFAULT_TIME_BUDGET,
    _uncertainty_to_complexity_boost,
    _derive_confidence_threshold,
    _parse_llm_output,
    _safe_default,
    run_heuristic_async,
)
from app.agents.registry import get_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk_enrichment(
    *,
    query: str = "what is the deployment status?",
    uncertainty_level: str = "low",
    budget_confidence: float = 0.6,
) -> dict:
    return {
        "query": query,
        "uncertainty_level": uncertainty_level,
        "budget_confidence": budget_confidence,
    }


def _mk_context(
    enrichment: dict | None = None,
    time_budget_seconds: int = 30,
    org_id: str = "org-test",
    job_id: str | None = "job-001",
) -> dict:
    return {
        "tenant": {"org_id": org_id, "org_slug": "test"},
        "memory": {"enrichment": enrichment or _mk_enrichment()},
        "runtime": {
            "job_id": job_id,
            "time_budget_seconds": time_budget_seconds,
        },
    }


# ---------------------------------------------------------------------------
# _uncertainty_to_complexity_boost
# ---------------------------------------------------------------------------

class TestUncertaintyBoost:
    def test_low(self):
        assert _uncertainty_to_complexity_boost("low") == 0.0

    def test_medium(self):
        assert _uncertainty_to_complexity_boost("medium") == 0.10

    def test_high(self):
        assert _uncertainty_to_complexity_boost("high") == 0.20

    def test_empty_string(self):
        assert _uncertainty_to_complexity_boost("") == 0.0

    def test_none(self):
        assert _uncertainty_to_complexity_boost(None) == 0.0

    def test_uppercase(self):
        assert _uncertainty_to_complexity_boost("HIGH") == 0.20

    def test_mixed_case(self):
        assert _uncertainty_to_complexity_boost("Medium") == 0.10

    def test_unknown(self):
        assert _uncertainty_to_complexity_boost("extreme") == 0.0


# ---------------------------------------------------------------------------
# _derive_confidence_threshold
# ---------------------------------------------------------------------------

class TestDeriveConfidenceThreshold:
    def test_low_complexity_high_budget_conf(self):
        t = _derive_confidence_threshold(0.1, 0.9, "low")
        assert 0.50 <= t <= 0.95

    def test_high_complexity_raises_threshold(self):
        t_low = _derive_confidence_threshold(0.1, 0.5, "low")
        t_high = _derive_confidence_threshold(0.9, 0.5, "low")
        assert t_high > t_low

    def test_high_uncertainty_raises_threshold(self):
        t_low = _derive_confidence_threshold(0.5, 0.5, "low")
        t_high = _derive_confidence_threshold(0.5, 0.5, "high")
        assert t_high > t_low

    def test_high_budget_conf_lowers_threshold(self):
        t_low_conf = _derive_confidence_threshold(0.5, 0.0, "low")
        t_high_conf = _derive_confidence_threshold(0.5, 1.0, "low")
        assert t_low_conf >= t_high_conf

    def test_clamped_to_0_50_minimum(self):
        t = _derive_confidence_threshold(0.0, 1.0, "low")
        assert t >= 0.50

    def test_clamped_to_0_95_maximum(self):
        t = _derive_confidence_threshold(1.0, 0.0, "high")
        assert t <= 0.95

    def test_returns_float(self):
        t = _derive_confidence_threshold(0.5, 0.5, "medium")
        assert isinstance(t, float)


# ---------------------------------------------------------------------------
# _parse_llm_output
# ---------------------------------------------------------------------------

class TestParseLlmOutput:
    def _valid(self, **overrides) -> dict:
        base = {
            "reasoning_mode": "mixed",
            "budget_allocated": 20,
            "confidence_threshold": 0.70,
            "seek_validation": False,
            "complexity_score": 0.5,
            "strategy_selected": "mixed",
        }
        base.update(overrides)
        return base

    def test_valid_returns_dict(self):
        result = _parse_llm_output(self._valid())
        assert result is not None
        assert result["reasoning_mode"] == "mixed"

    def test_rationale_set_to_llm(self):
        result = _parse_llm_output(self._valid())
        assert result["rationale"] == "llm"

    def test_invalid_mode_returns_none(self):
        assert _parse_llm_output(self._valid(reasoning_mode="deep")) is None

    def test_invalid_budget_string_returns_none(self):
        assert _parse_llm_output(self._valid(budget_allocated="twenty")) is None

    def test_budget_zero_returns_none(self):
        assert _parse_llm_output(self._valid(budget_allocated=0)) is None

    def test_invalid_threshold_returns_none(self):
        assert _parse_llm_output(self._valid(confidence_threshold=1.5)) is None

    def test_non_dict_returns_none(self):
        assert _parse_llm_output("not a dict") is None

    def test_none_returns_none(self):
        assert _parse_llm_output(None) is None

    def test_all_valid_modes_accepted(self):
        for mode in _VALID_MODES:
            result = _parse_llm_output(self._valid(reasoning_mode=mode, strategy_selected=mode))
            assert result is not None, f"mode {mode!r} should be valid"

    def test_seek_validation_cast_to_bool(self):
        result = _parse_llm_output(self._valid(seek_validation=1))
        assert result["seek_validation"] is True

    def test_strategy_selected_uses_reasoning_mode(self):
        result = _parse_llm_output(self._valid(reasoning_mode="deliberative"))
        assert result["strategy_selected"] == "deliberative"


# ---------------------------------------------------------------------------
# _safe_default
# ---------------------------------------------------------------------------

class TestSafeDefault:
    def test_returns_mixed_mode(self):
        d = _safe_default("any query")
        assert d["reasoning_mode"] == "mixed"

    def test_budget_is_positive_int(self):
        d = _safe_default("x")
        assert isinstance(d["budget_allocated"], int)
        assert d["budget_allocated"] >= 1

    def test_threshold_in_range(self):
        d = _safe_default("x")
        assert 0.0 <= d["confidence_threshold"] <= 1.0

    def test_seek_validation_is_bool(self):
        d = _safe_default("x")
        assert isinstance(d["seek_validation"], bool)

    def test_rationale_is_heuristic(self):
        d = _safe_default("x")
        assert d["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# run_heuristic_async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunHeuristicAsync:
    async def test_simple_query_heuristic_mode(self):
        result = await run_heuristic_async("hello", "low", 0.6, 30)
        assert result["reasoning_mode"] in _VALID_MODES
        assert result["rationale"] == "heuristic"

    async def test_complex_query_deliberative_or_mixed(self):
        # Long multi-signal query: many tokens + complexity terms + irreversible → complexity near/above 0.75
        query = (
            "why did the deployment fail, compare the tradeoffs of each strategy, "
            "forecast the risk impact, and outline irreversible compliance consequences "
            "for the multi-step migration plan?"
        )
        result = await run_heuristic_async(query, "high", 0.3, 30)
        assert result["reasoning_mode"] in {"mixed", "deliberative", "escalate"}

    async def test_high_uncertainty_increases_complexity(self):
        q = "status?"
        r_low = await run_heuristic_async(q, "low", 0.6, 30)
        r_high = await run_heuristic_async(q, "high", 0.6, 30)
        assert r_high["complexity_score"] >= r_low["complexity_score"]

    async def test_budget_allocated_is_positive_int(self):
        result = await run_heuristic_async("test query", "low", 0.5, 30)
        assert isinstance(result["budget_allocated"], int)
        assert result["budget_allocated"] >= 6

    async def test_seek_validation_bool(self):
        result = await run_heuristic_async("test", "low", 0.5, 30)
        assert isinstance(result["seek_validation"], bool)

    async def test_confidence_threshold_in_range(self):
        result = await run_heuristic_async("optimize strategy", "medium", 0.4, 30)
        assert 0.50 <= result["confidence_threshold"] <= 0.95

    async def test_empty_query(self):
        result = await run_heuristic_async("", "low", 0.5, 30)
        assert result["reasoning_mode"] in _VALID_MODES

    async def test_short_time_budget_may_pick_heuristic(self):
        result = await run_heuristic_async("hello", "low", 0.5, 10)
        # short time budget on simple query → heuristic or mixed
        assert result["reasoning_mode"] in _VALID_MODES

    async def test_irreversible_query_escalates(self):
        # Requires high complexity (many tokens + signals) AND high_risk to trigger escalate
        query = (
            "irreversible compliance change with legal safety risk: compare tradeoffs, "
            "forecast outcomes, optimize multi-step strategy, prove compliance"
        )
        result = await run_heuristic_async(query, "high", 0.3, 30)
        assert result["reasoning_mode"] in {"deliberative", "escalate"}

    async def test_returns_all_required_keys(self):
        result = await run_heuristic_async("query", "low", 0.5, 30)
        for key in (
            "reasoning_mode", "budget_allocated", "confidence_threshold",
            "seek_validation", "complexity_score", "strategy_selected", "rationale",
        ):
            assert key in result, f"missing key: {key}"


# ---------------------------------------------------------------------------
# MetaCognitivePlanningAgent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMetaCognitivePlanningAgent:
    def _agent(self) -> MetaCognitivePlanningAgent:
        return MetaCognitivePlanningAgent()

    # --- metadata ---

    def test_name(self):
        assert self._agent().name == "MetaCognitivePlanningAgent"

    def test_version(self):
        assert self._agent().version == "v1"

    def test_dependencies(self):
        deps = self._agent().dependencies()
        assert "UncertaintyReportingAgent" in deps
        assert "AdaptiveEnrichmentBudgetAgent" in deps

    # --- heuristic path ---

    async def test_heuristic_run_returns_success(self):
        agent = self._agent()
        ctx = _mk_context()
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.agent_name == "MetaCognitivePlanningAgent"

    async def test_heuristic_outputs_valid_mode(self):
        agent = self._agent()
        ctx = _mk_context()
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["reasoning_mode"] in _VALID_MODES

    async def test_heuristic_budget_positive(self):
        agent = self._agent()
        ctx = _mk_context()
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["budget_allocated"] >= 1

    async def test_heuristic_confidence_threshold_range(self):
        agent = self._agent()
        ctx = _mk_context()
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        t = result.outputs["confidence_threshold"]
        assert 0.0 <= t <= 1.0

    async def test_heuristic_seek_validation_is_bool(self):
        agent = self._agent()
        ctx = _mk_context()
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert isinstance(result.outputs["seek_validation"], bool)

    async def test_heuristic_trace_id_propagated(self):
        agent = self._agent()
        ctx = _mk_context(job_id="job-xyz")
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-xyz"

    async def test_heuristic_no_job_id(self):
        agent = self._agent()
        ctx = _mk_context(job_id=None)
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id is None

    async def test_heuristic_high_uncertainty_context(self):
        agent = self._agent()
        ctx = _mk_context(enrichment=_mk_enrichment(uncertainty_level="high"))
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"

    async def test_heuristic_empty_enrichment(self):
        agent = self._agent()
        ctx = _mk_context(enrichment={})
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"

    async def test_heuristic_no_memory_key(self):
        agent = self._agent()
        ctx = {"tenant": {"org_id": "org-1"}, "runtime": {"job_id": "j1"}}
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.status == "success"

    async def test_started_before_finished(self):
        agent = self._agent()
        ctx = _mk_context()
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.started_at <= result.finished_at

    # --- LLM path ---

    async def test_llm_path_uses_parsed_output(self):
        agent = self._agent()
        ctx = _mk_context()

        llm_response = {
            "reasoning_mode": "deliberative",
            "budget_allocated": 35,
            "confidence_threshold": 0.80,
            "seek_validation": True,
            "complexity_score": 0.75,
            "strategy_selected": "deliberative",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.meta_cognitive_planning_agent.settings") as s, \
             patch("app.agents.meta_cognitive_planning_agent.create_ollama_client",
                   return_value=mock_client):
            s.AGENT_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "qwen2.5:7b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.outputs["reasoning_mode"] == "deliberative"
        assert result.outputs["rationale"] == "llm"

    async def test_llm_path_fallback_on_bad_response(self):
        agent = self._agent()
        ctx = _mk_context()

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "data"})

        with patch("app.agents.meta_cognitive_planning_agent.settings") as s, \
             patch("app.agents.meta_cognitive_planning_agent.create_ollama_client",
                   return_value=mock_client):
            s.AGENT_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "qwen2.5:7b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    async def test_llm_path_fallback_on_exception(self):
        agent = self._agent()
        ctx = _mk_context()

        with patch("app.agents.meta_cognitive_planning_agent.settings") as s, \
             patch("app.agents.meta_cognitive_planning_agent.create_ollama_client",
                   side_effect=Exception("connection refused")):
            s.AGENT_STRATEGY = "llm"
            s.OLLAMA_BASE_URL = "http://localhost:11434"
            s.OLLAMA_MODEL = "qwen2.5:7b"
            s.OLLAMA_TIMEOUT_SECONDS = 5.0
            s.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    async def test_llm_path_all_modes_accepted(self):
        agent = self._agent()

        for mode in _VALID_MODES:
            llm_response = {
                "reasoning_mode": mode,
                "budget_allocated": 20,
                "confidence_threshold": 0.70,
                "seek_validation": False,
                "complexity_score": 0.5,
                "strategy_selected": mode,
            }
            mock_client = AsyncMock()
            mock_client.complete_json = AsyncMock(return_value=llm_response)

            with patch("app.agents.meta_cognitive_planning_agent.settings") as s, \
                 patch("app.agents.meta_cognitive_planning_agent.create_ollama_client",
                       return_value=mock_client):
                s.AGENT_STRATEGY = "llm"
                s.OLLAMA_BASE_URL = "http://localhost:11434"
                s.OLLAMA_MODEL = "qwen2.5:7b"
                s.OLLAMA_TIMEOUT_SECONDS = 5.0
                s.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-1", _mk_context())

            assert result.outputs["reasoning_mode"] == mode

    # --- validate_outputs ---

    def test_validate_outputs_passes_on_valid(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "success"
        result.outputs = {
            "reasoning_mode": "mixed",
            "budget_allocated": 20,
            "confidence_threshold": 0.70,
            "seek_validation": False,
            "complexity_score": 0.5,
        }
        agent.validate_outputs(result)  # should not raise

    def test_validate_outputs_skips_non_success(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "failed"
        result.outputs = {}
        agent.validate_outputs(result)  # should not raise

    def test_validate_outputs_bad_mode_raises(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "success"
        result.outputs = {
            "reasoning_mode": "unknown",
            "budget_allocated": 20,
            "confidence_threshold": 0.70,
            "seek_validation": False,
            "complexity_score": 0.5,
        }
        with pytest.raises(ValueError, match="reasoning_mode"):
            agent.validate_outputs(result)

    def test_validate_outputs_bad_budget_raises(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "success"
        result.outputs = {
            "reasoning_mode": "mixed",
            "budget_allocated": 0,
            "confidence_threshold": 0.70,
            "seek_validation": False,
            "complexity_score": 0.5,
        }
        with pytest.raises(ValueError, match="budget_allocated"):
            agent.validate_outputs(result)

    def test_validate_outputs_bad_threshold_raises(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "success"
        result.outputs = {
            "reasoning_mode": "mixed",
            "budget_allocated": 20,
            "confidence_threshold": 1.5,
            "seek_validation": False,
            "complexity_score": 0.5,
        }
        with pytest.raises(ValueError, match="confidence_threshold"):
            agent.validate_outputs(result)

    def test_validate_outputs_bad_seek_validation_raises(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "success"
        result.outputs = {
            "reasoning_mode": "mixed",
            "budget_allocated": 20,
            "confidence_threshold": 0.70,
            "seek_validation": "yes",
            "complexity_score": 0.5,
        }
        with pytest.raises(ValueError, match="seek_validation"):
            agent.validate_outputs(result)

    def test_validate_outputs_bad_complexity_raises(self):
        agent = self._agent()
        result = MagicMock()
        result.status = "success"
        result.outputs = {
            "reasoning_mode": "mixed",
            "budget_allocated": 20,
            "confidence_threshold": 0.70,
            "seek_validation": False,
            "complexity_score": 1.5,
        }
        with pytest.raises(ValueError, match="complexity_score"):
            agent.validate_outputs(result)

    # --- query routing ---

    async def test_complex_irreversible_query_high_seek_validation(self):
        agent = self._agent()
        ctx = _mk_context(
            enrichment=_mk_enrichment(
                query="irreversible compliance change with legal risk",
                uncertainty_level="high",
            )
        )
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["seek_validation"] is True or \
               result.outputs["reasoning_mode"] in {"deliberative", "escalate"}

    async def test_simple_query_low_budget(self):
        agent = self._agent()
        ctx = _mk_context(
            enrichment=_mk_enrichment(query="hello", uncertainty_level="low"),
            time_budget_seconds=10,
        )
        with patch("app.agents.meta_cognitive_planning_agent.settings") as s:
            s.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.outputs["budget_allocated"] <= 50  # sanity cap

    # --- registry ---

    def test_registry_lookup(self):
        agent = get_agent("meta_cognitive_planning")
        assert isinstance(agent, MetaCognitivePlanningAgent)

    def test_registry_alias_camel(self):
        agent = get_agent("MetaCognitivePlanningAgent")
        assert isinstance(agent, MetaCognitivePlanningAgent)

    def test_registry_alias_short(self):
        agent = get_agent("meta_cognitive")
        assert isinstance(agent, MetaCognitivePlanningAgent)

    def test_registry_unknown_returns_none(self):
        assert get_agent("not_an_agent_xyz") is None
