"""Tests for Phase 49 — Cognitive Gateway Service.

Covers: CognitiveGatewayCapabilities, all five verb methods (write/read/decide/
plan/explain), capability gating, and pure helper functions.
"""

from __future__ import annotations

import pytest

from app.services.cognitive_gateway_service import (
    CognitiveGatewayService,
    CognitiveGatewayCapabilities,
    GatewayWriteResult,
    GatewayReadResult,
    GatewayDecideResult,
    GatewayPlanResult,
    GatewayExplainResult,
    _enrich_write_heuristic,
    _assemble_read_context,
    _heuristic_decide,
    _heuristic_plan,
    _heuristic_explain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _full_gateway() -> CognitiveGatewayService:
    return CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.full())


def _lite_gateway() -> CognitiveGatewayService:
    return CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.lite())


def _standard_gateway() -> CognitiveGatewayService:
    return CognitiveGatewayService(capabilities=CognitiveGatewayCapabilities.standard())


# ===========================================================================
# CognitiveGatewayCapabilities tests
# ===========================================================================

class TestCognitiveGatewayCapabilities:
    def test_full_has_all_verbs(self):
        caps = CognitiveGatewayCapabilities.full()
        for verb in ("write", "read", "decide", "plan", "explain"):
            assert caps.is_enabled(verb)

    def test_lite_only_read_write(self):
        caps = CognitiveGatewayCapabilities.lite()
        assert caps.is_enabled("read")
        assert caps.is_enabled("write")
        assert not caps.is_enabled("decide")
        assert not caps.is_enabled("plan")
        assert not caps.is_enabled("explain")

    def test_standard_has_no_plan(self):
        caps = CognitiveGatewayCapabilities.standard()
        assert not caps.is_enabled("plan")
        assert caps.is_enabled("decide")
        assert caps.is_enabled("explain")

    def test_custom_capabilities(self):
        caps = CognitiveGatewayCapabilities(enabled_verbs=frozenset({"decide"}))
        assert caps.is_enabled("decide")
        assert not caps.is_enabled("write")

    def test_unknown_verb_not_enabled(self):
        caps = CognitiveGatewayCapabilities.full()
        assert not caps.is_enabled("unknown_verb")


# ===========================================================================
# _enrich_write_heuristic tests
# ===========================================================================

class TestEnrichWriteHeuristic:
    def test_urgent_tone_for_critical(self):
        enr = _enrich_write_heuristic("critical outage in production", "", [], {})
        assert enr["tone"] == "urgent"

    def test_cautionary_tone_for_warning(self):
        enr = _enrich_write_heuristic("warning: latency increasing", "", [], {})
        assert enr["tone"] == "cautionary"

    def test_informational_tone_default(self):
        enr = _enrich_write_heuristic("deployment completed successfully", "", [], {})
        assert enr["tone"] == "informational"

    def test_security_domain_detected(self):
        enr = _enrich_write_heuristic("SSO authentication failure detected", "", [], {})
        assert enr["domain"] == "security"

    def test_devops_domain_detected(self):
        enr = _enrich_write_heuristic("CI pipeline build failed", "", [], {})
        assert enr["domain"] == "devops"

    def test_finance_domain_detected(self):
        enr = _enrich_write_heuristic("billing invoice not processed", "", [], {})
        assert enr["domain"] == "finance"

    def test_general_domain_default(self):
        enr = _enrich_write_heuristic("team meeting notes", "", [], {})
        assert enr["domain"] == "general"

    def test_word_count_computed(self):
        enr = _enrich_write_heuristic("one two three four five", "", [], {})
        assert enr["word_count"] == 5

    def test_has_metadata_flag(self):
        enr = _enrich_write_heuristic("content", "", [], {"key": "val"})
        assert enr["has_metadata"] is True


# ===========================================================================
# _assemble_read_context tests
# ===========================================================================

class TestAssembleReadContext:
    def test_empty_returns_empty(self):
        assert _assemble_read_context([], "query") == []

    def test_relevant_memory_ranked_first(self):
        mems = [
            {"id": "1", "content": "deploy pipeline failure"},
            {"id": "2", "content": "auth login sso failure auth sso"},
            {"id": "3", "content": "unrelated content here today"},
        ]
        ranked = _assemble_read_context(mems, "sso auth failure")
        # memory 2 has most matching tokens (auth, sso, failure)
        assert ranked[0]["id"] == "2"
        assert ranked[-1]["id"] == "3"

    def test_respects_limit(self):
        mems = [{"id": str(i), "content": f"item {i}"} for i in range(20)]
        ranked = _assemble_read_context(mems, "item")
        assert len(ranked) == 20  # _assemble doesn't limit; caller does via read()

    def test_all_match_preserves_count(self):
        mems = [{"id": "a", "content": "alpha"}, {"id": "b", "content": "beta"}]
        ranked = _assemble_read_context(mems, "gamma")
        assert len(ranked) == 2


# ===========================================================================
# _heuristic_decide tests
# ===========================================================================

class TestHeuristicDecide:
    def test_high_anomaly_escalate(self):
        result = _heuristic_decide("issue", {"anomaly_detected": True, "anomaly_score": 0.95})
        assert result.decision == "escalate"
        assert result.confidence >= 0.9
        assert result.action_recommended is not None

    def test_medium_anomaly_investigate(self):
        result = _heuristic_decide("issue", {"anomaly_detected": True, "anomaly_score": 0.75})
        assert result.decision == "investigate"

    def test_critical_keyword_escalate(self):
        result = _heuristic_decide("critical system outage detected", {})
        assert result.decision == "escalate"

    def test_warning_keyword_monitor(self):
        result = _heuristic_decide("warning: slow response times", {})
        assert result.decision == "monitor"

    def test_neutral_content_acknowledge(self):
        result = _heuristic_decide("user updated their profile", {})
        assert result.decision == "acknowledge"
        assert result.action_recommended is None

    def test_tone_from_enrichment(self):
        result = _heuristic_decide("issue", {"tone": "urgent"})
        assert result.tone == "urgent"

    def test_agents_run_list_non_empty(self):
        result = _heuristic_decide("content", {})
        assert len(result.agents_run) > 0

    def test_confidence_in_valid_range(self):
        result = _heuristic_decide("content", {"anomaly_detected": True, "anomaly_score": 1.0})
        assert 0.0 <= result.confidence <= 1.0


# ===========================================================================
# _heuristic_plan tests
# ===========================================================================

class TestHeuristicPlan:
    def test_investigate_goal_generates_steps(self):
        result = _heuristic_plan("investigate the login failure", {})
        assert result.step_count >= 3
        assert result.goal == "investigate the login failure"

    def test_escalate_goal_includes_playbook_step(self):
        result = _heuristic_plan("escalate P1 incident", {})
        tools = [s.get("tool") for s in result.steps]
        assert "playbook.match" in tools

    def test_report_goal_includes_narrative_step(self):
        result = _heuristic_plan("report on weekly incidents", {})
        tools = [s.get("tool") for s in result.steps]
        assert "narrative.synthesize" in tools

    def test_default_goal_produces_4_steps(self):
        result = _heuristic_plan("do something", {})
        assert result.step_count == 4

    def test_step_ids_unique(self):
        result = _heuristic_plan("investigate", {})
        ids = [s["step_id"] for s in result.steps]
        assert len(ids) == len(set(ids))

    def test_blocking_step_from_context(self):
        result = _heuristic_plan("investigate", {"blocking_subtask": "get credentials"})
        assert result.blocking_step == "get credentials"

    def test_confidence_reasonable(self):
        result = _heuristic_plan("investigate", {})
        assert 0.0 < result.confidence <= 1.0


# ===========================================================================
# _heuristic_explain tests
# ===========================================================================

class TestHeuristicExplain:
    def test_empty_records_no_crash(self):
        result = _heuristic_explain("mem-1", [])
        assert result.memory_id == "mem-1"
        assert "No audit" in result.explainability_summary
        assert result.confidence == 0.0

    def test_agents_extracted_from_records(self):
        records = [
            {"agent_name": "EntityResolutionAgent", "confidence": 0.9},
            {"agent_name": "AnomalyDetectionAgent", "confidence": 0.8},
        ]
        result = _heuristic_explain("mem-2", records)
        assert "EntityResolutionAgent" in result.agents or "AnomalyDetectionAgent" in result.agents

    def test_confidence_averaged(self):
        records = [
            {"agent_name": "AgentA", "confidence": 0.8},
            {"agent_name": "AgentB", "confidence": 0.6},
        ]
        result = _heuristic_explain("mem-3", records)
        assert abs(result.confidence - 0.7) < 0.01

    def test_summary_mentions_memory_id(self):
        records = [{"agent_name": "AgentA", "confidence": 0.9}]
        result = _heuristic_explain("mem-xyz", records)
        assert "mem-xyz" in result.explainability_summary

    def test_decisions_capped_at_5(self):
        records = [{"agent_name": f"Agent{i}", "confidence": 0.8} for i in range(10)]
        result = _heuristic_explain("m", records)
        assert len(result.decisions) <= 5


# ===========================================================================
# CognitiveGatewayService verb tests
# ===========================================================================

class TestGatewayWrite:
    def test_returns_write_result(self):
        gw = _full_gateway()
        result = gw.write(content="System alert: auth failure")
        assert isinstance(result, GatewayWriteResult)
        assert result.memory_id
        assert result.enriched

    def test_enrichment_summary_populated(self):
        gw = _full_gateway()
        result = gw.write(content="critical outage in prod")
        assert result.enrichment_summary.get("tone") == "urgent"

    def test_tags_preserved(self):
        gw = _full_gateway()
        result = gw.write(content="test", tags=["tag1", "tag2"])
        assert "tag1" in result.tags

    def test_lite_gateway_allows_write(self):
        gw = _lite_gateway()
        result = gw.write(content="test content")
        assert result.enriched

    def test_memory_id_unique(self):
        gw = _full_gateway()
        r1 = gw.write(content="a")
        r2 = gw.write(content="b")
        assert r1.memory_id != r2.memory_id


class TestGatewayRead:
    def test_returns_read_result(self):
        gw = _full_gateway()
        result = gw.read(query="auth failure")
        assert isinstance(result, GatewayReadResult)

    def test_empty_memories_returns_zero(self):
        gw = _full_gateway()
        result = gw.read(query="anything", memories=[])
        assert result.total == 0
        assert not result.context_assembled

    def test_memories_ranked_by_relevance(self):
        gw = _full_gateway()
        mems = [
            {"id": "1", "content": "unrelated content here"},
            {"id": "2", "content": "auth sso login token"},
        ]
        result = gw.read(query="auth sso", memories=mems)
        assert result.memories[0]["id"] == "2"

    def test_limit_respected(self):
        gw = _full_gateway()
        mems = [{"id": str(i), "content": "item"} for i in range(20)]
        result = gw.read(query="item", memories=mems, limit=5)
        assert result.total <= 5

    def test_lite_gateway_allows_read(self):
        gw = _lite_gateway()
        result = gw.read(query="anything")
        assert isinstance(result, GatewayReadResult)


class TestGatewayDecide:
    def test_returns_decide_result(self):
        gw = _full_gateway()
        result = gw.decide(content="critical system failure")
        assert isinstance(result, GatewayDecideResult)

    def test_decision_is_valid_string(self):
        gw = _full_gateway()
        result = gw.decide(content="test content")
        assert result.decision in ("escalate", "investigate", "monitor", "acknowledge")

    def test_confidence_in_range(self):
        gw = _full_gateway()
        result = gw.decide(content="test")
        assert 0.0 <= result.confidence <= 1.0

    def test_lite_gateway_denies_decide(self):
        gw = _lite_gateway()
        with pytest.raises(PermissionError, match="decide"):
            gw.decide(content="test")

    def test_enrichment_passed_through(self):
        gw = _full_gateway()
        result = gw.decide(content="test", enrichment={"anomaly_detected": True, "anomaly_score": 0.95})
        assert result.decision == "escalate"


class TestGatewayPlan:
    def test_returns_plan_result(self):
        gw = _full_gateway()
        result = gw.plan(goal="investigate login failure")
        assert isinstance(result, GatewayPlanResult)

    def test_steps_non_empty(self):
        gw = _full_gateway()
        result = gw.plan(goal="escalate P1 incident")
        assert result.step_count > 0

    def test_lite_gateway_denies_plan(self):
        gw = _lite_gateway()
        with pytest.raises(PermissionError, match="plan"):
            gw.plan(goal="test")

    def test_standard_gateway_denies_plan(self):
        gw = _standard_gateway()
        with pytest.raises(PermissionError, match="plan"):
            gw.plan(goal="test")

    def test_context_blocking_step_propagated(self):
        gw = _full_gateway()
        result = gw.plan(goal="investigate", context={"blocking_subtask": "get token"})
        assert result.blocking_step == "get token"


class TestGatewayExplain:
    def test_returns_explain_result(self):
        gw = _full_gateway()
        result = gw.explain(memory_id="mem-test")
        assert isinstance(result, GatewayExplainResult)
        assert result.memory_id == "mem-test"

    def test_empty_audit_returns_no_agents(self):
        gw = _full_gateway()
        result = gw.explain(memory_id="x", audit_records=[])
        assert result.agents == []

    def test_audit_records_parsed(self):
        gw = _full_gateway()
        result = gw.explain(
            memory_id="m1",
            audit_records=[
                {"agent_name": "EntityResolutionAgent", "confidence": 0.92},
                {"agent_name": "AnomalyDetectionAgent", "confidence": 0.88},
            ],
        )
        assert result.confidence > 0
        assert len(result.agents) == 2

    def test_lite_gateway_denies_explain(self):
        gw = _lite_gateway()
        with pytest.raises(PermissionError, match="explain"):
            gw.explain(memory_id="x")

    def test_standard_gateway_allows_explain(self):
        gw = _standard_gateway()
        result = gw.explain(memory_id="y")
        assert isinstance(result, GatewayExplainResult)
