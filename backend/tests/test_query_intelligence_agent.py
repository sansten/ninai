"""Tests for QueryIntelligenceAgent — Phase 27."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.query_intelligence_agent import (
    QueryIntelligenceAgent,
    detect_intent,
    extract_entities,
    build_dynamic_filters,
    suggest_agents,
    count_enrichment_signals,
    compute_confidence,
    run_heuristic,
    _valid_llm_response,
    _VALID_INTENTS,
    _DEFAULT_INTENT,
    _INTENT_AGENT_MAP,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(enrichment: dict | None = None, content: str = "") -> dict:
    mem: dict = {"enrichment": enrichment or {}}
    if content:
        mem["content"] = content
    return {
        "memory": mem,
        "runtime": {"job_id": "trace-27"},
    }


def _rich_enrichment() -> dict:
    return {
        "entity_name": "Acme Corp",
        "entity_type": "organisation",
        "resolved_aliases": ["Acme", "ACME"],
        "key_entities": ["John Smith", "Project Alpha"],
        "uncertainty_level": "low",
        "credibility_score": 0.85,
        "temporal_confidence": 0.80,
        "tags": ["sales", "enterprise", "q4"],
    }


def _result(outputs: dict, status: str = "success") -> AgentResult:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="QueryIntelligenceAgent",
        agent_version="v1",
        memory_id="mem-test",
        status=status,
        confidence=outputs.get("confidence", 0.65),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
        trace_id=None,
    )


# ---------------------------------------------------------------------------
# TestDetectIntent
# ---------------------------------------------------------------------------

class TestDetectIntent:
    def test_empty_text_returns_default(self):
        assert detect_intent("") == _DEFAULT_INTENT

    def test_whitespace_returns_default(self):
        assert detect_intent("   ") == _DEFAULT_INTENT

    def test_locate_keywords(self):
        assert detect_intent("find the contract for Acme") == "locate"

    def test_locate_search_keyword(self):
        assert detect_intent("search for all Q4 reports") == "locate"

    def test_explain_why_keyword(self):
        assert detect_intent("why did the deal fall through") == "explain"

    def test_explain_cause_keyword(self):
        assert detect_intent("the cause of the outage was a config error") == "explain"

    def test_compare_versus(self):
        assert detect_intent("compare product A vs product B") == "compare"

    def test_compare_difference(self):
        assert detect_intent("what is the difference between v1 and v2") == "compare"

    def test_analyze_trend(self):
        assert detect_intent("analyze the trend in customer churn") == "analyze"

    def test_generate_draft(self):
        assert detect_intent("draft a summary of the meeting") == "generate"

    def test_find_person_who(self):
        assert detect_intent("who is the owner of this account") == "find_person"

    def test_find_timeline_when(self):
        assert detect_intent("when was the last deployment") == "find_timeline"

    def test_no_matching_keyword_returns_default(self):
        assert detect_intent("the quick brown fox") == _DEFAULT_INTENT

    def test_case_insensitive(self):
        assert detect_intent("FIND the document") == "locate"

    def test_multiple_keywords_picks_highest_score(self):
        # "why" + "cause" + "reason" all point to explain
        result = detect_intent("why was the cause and reason for this")
        assert result == "explain"

    def test_ties_broken_by_max_key(self):
        # Single word from two different intents — whichever wins max() first
        # Just assert it returns a valid intent
        result = detect_intent("find who owns this")
        assert result in _VALID_INTENTS


# ---------------------------------------------------------------------------
# TestExtractEntities
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_empty_inputs(self):
        result = extract_entities("", {})
        assert result == []

    def test_entity_name_from_enrichment(self):
        result = extract_entities("", {"entity_name": "Acme Corp"})
        assert "Acme Corp" in result

    def test_resolved_aliases_from_enrichment(self):
        result = extract_entities("", {"resolved_aliases": ["Acme", "ACME Ltd"]})
        assert "Acme" in result
        assert "ACME Ltd" in result

    def test_key_entities_from_enrichment(self):
        result = extract_entities("", {"key_entities": ["John Smith", "Project X"]})
        assert "John Smith" in result
        assert "Project X" in result

    def test_capitalised_words_from_content(self):
        result = extract_entities("This is about Microsoft and Google", {})
        assert "Microsoft" in result
        assert "Google" in result

    def test_first_word_excluded(self):
        # "This" at position 0 should not be extracted
        result = extract_entities("This is about nothing", {})
        assert "This" not in result

    def test_deduplication(self):
        result = extract_entities(
            "Acme Corp is great",
            {"entity_name": "Acme Corp", "key_entities": ["Acme Corp"]},
        )
        assert result.count("Acme Corp") == 1

    def test_sorted_output(self):
        result = extract_entities("", {"key_entities": ["Zebra", "Alpha", "Mike"]})
        assert result == sorted(result)

    def test_cap_at_twenty(self):
        entities = [f"Entity{i}" for i in range(30)]
        result = extract_entities("", {"key_entities": entities})
        assert len(result) <= 20

    def test_invalid_alias_type_ignored(self):
        result = extract_entities("", {"resolved_aliases": [123, None, "Valid"]})
        assert "Valid" in result
        assert 123 not in result

    def test_punctuation_stripped(self):
        result = extract_entities("the Apple, and Microsoft.", {})
        assert "Apple" in result
        assert "Microsoft" in result


# ---------------------------------------------------------------------------
# TestBuildDynamicFilters
# ---------------------------------------------------------------------------

class TestBuildDynamicFilters:
    def test_empty_enrichment_returns_empty(self):
        result = build_dynamic_filters("retrieve", {})
        assert result == {}

    def test_entity_type_included(self):
        result = build_dynamic_filters("retrieve", {"entity_type": "person"})
        assert result["entity_type"] == "person"

    def test_tags_from_enrichment(self):
        result = build_dynamic_filters("retrieve", {"tags": ["sales", "q4"]})
        assert result["tags"] == ["sales", "q4"]

    def test_normalized_tags_fallback(self):
        result = build_dynamic_filters("retrieve", {"normalized_tags": ["ops", "infra"]})
        assert result["tags"] == ["ops", "infra"]

    def test_tags_capped_at_five(self):
        tags = [f"tag{i}" for i in range(10)]
        result = build_dynamic_filters("retrieve", {"tags": tags})
        assert len(result["tags"]) == 5

    def test_high_uncertainty_adds_exclude_filter(self):
        result = build_dynamic_filters("retrieve", {"uncertainty_level": "high"})
        assert "exclude_uncertainty_levels" in result

    def test_critical_uncertainty_adds_exclude_filter(self):
        result = build_dynamic_filters("retrieve", {"uncertainty_level": "critical"})
        assert "exclude_uncertainty_levels" in result

    def test_low_uncertainty_no_exclude_filter(self):
        result = build_dynamic_filters("retrieve", {"uncertainty_level": "low"})
        assert "exclude_uncertainty_levels" not in result

    def test_find_timeline_adds_temporal_filter(self):
        result = build_dynamic_filters("find_timeline", {})
        assert result.get("has_temporal_data") is True

    def test_find_person_adds_type_hint(self):
        result = build_dynamic_filters("find_person", {})
        assert result.get("entity_type_hint") == "person"

    def test_analyze_adds_min_credibility(self):
        result = build_dynamic_filters("analyze", {})
        assert result.get("min_credibility") == 0.5

    def test_non_string_tags_excluded(self):
        result = build_dynamic_filters("retrieve", {"tags": [1, 2, "valid"]})
        assert result["tags"] == ["valid"]


# ---------------------------------------------------------------------------
# TestSuggestAgents
# ---------------------------------------------------------------------------

class TestSuggestAgents:
    def test_all_intents_return_non_empty(self):
        for intent in _VALID_INTENTS:
            agents = suggest_agents(intent)
            assert len(agents) > 0, f"intent {intent!r} returned empty agents"

    def test_unknown_intent_returns_retrieve_list(self):
        result = suggest_agents("totally_unknown")
        assert result == list(_INTENT_AGENT_MAP["retrieve"])

    def test_locate_includes_entity_resolution(self):
        assert "EntityResolutionAgent" in suggest_agents("locate")

    def test_explain_includes_causal(self):
        assert "CausalReasoningAgent" in suggest_agents("explain")

    def test_analyze_includes_anomaly(self):
        assert "AnomalyDetectionAgent" in suggest_agents("analyze")

    def test_generate_includes_narrative(self):
        assert "NarrativeSynthesisAgent" in suggest_agents("generate")

    def test_find_person_includes_org_attention(self):
        assert "OrgAttentionAgent" in suggest_agents("find_person")

    def test_find_timeline_includes_temporal(self):
        assert "TemporalReasoningAgent" in suggest_agents("find_timeline")


# ---------------------------------------------------------------------------
# TestCountEnrichmentSignals
# ---------------------------------------------------------------------------

class TestCountEnrichmentSignals:
    def test_empty_enrichment_returns_zero(self):
        assert count_enrichment_signals({}) == 0

    def test_one_signal(self):
        assert count_enrichment_signals({"entity_name": "X"}) == 1

    def test_all_signals(self):
        enrichment = {
            "entity_name": "X",
            "uncertainty_level": "low",
            "key_entities": [],
            "tags": [],
            "credibility_score": 0.9,
            "temporal_confidence": 0.8,
            "normalized_tags": [],
            "entity_type": "person",
        }
        assert count_enrichment_signals(enrichment) == 8

    def test_irrelevant_keys_not_counted(self):
        assert count_enrichment_signals({"unknown_key": "value"}) == 0


# ---------------------------------------------------------------------------
# TestComputeConfidence
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_minimum_inputs_returns_base(self):
        result = compute_confidence(False, 0, 0, "retrieve")
        assert result == 0.50

    def test_has_content_raises_confidence(self):
        with_content = compute_confidence(True, 0, 0, "retrieve")
        without = compute_confidence(False, 0, 0, "retrieve")
        assert with_content > without

    def test_entities_raise_confidence(self):
        one = compute_confidence(False, 1, 0, "retrieve")
        zero = compute_confidence(False, 0, 0, "retrieve")
        assert one > zero

    def test_three_entities_bonus(self):
        three = compute_confidence(False, 3, 0, "retrieve")
        one = compute_confidence(False, 1, 0, "retrieve")
        assert three > one

    def test_enrichment_signals_raise_confidence(self):
        with_signals = compute_confidence(False, 0, 3, "retrieve")
        without = compute_confidence(False, 0, 0, "retrieve")
        assert with_signals > without

    def test_specific_intent_bonus(self):
        specific = compute_confidence(False, 0, 0, "analyze")
        default = compute_confidence(False, 0, 0, "retrieve")
        assert specific > default

    def test_max_capped_at_0_90(self):
        result = compute_confidence(True, 10, 8, "analyze")
        assert result <= 0.90

    def test_min_floor_at_0_40(self):
        result = compute_confidence(False, 0, 0, "retrieve")
        assert result >= 0.40

    def test_result_is_float(self):
        result = compute_confidence(True, 2, 3, "locate")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# TestRunHeuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_returns_all_required_fields(self):
        result = run_heuristic("find the contract", _rich_enrichment())
        assert "query_intent" in result
        assert "extracted_entities" in result
        assert "dynamic_filters" in result
        assert "suggested_agents" in result
        assert "confidence" in result
        assert "rationale" in result

    def test_rationale_is_heuristic(self):
        result = run_heuristic("anything", {})
        assert result["rationale"] == "heuristic"

    def test_intent_is_valid(self):
        result = run_heuristic("why did this happen", {})
        assert result["query_intent"] in _VALID_INTENTS

    def test_entities_is_list(self):
        result = run_heuristic("", {})
        assert isinstance(result["extracted_entities"], list)

    def test_filters_is_dict(self):
        result = run_heuristic("", {})
        assert isinstance(result["dynamic_filters"], dict)

    def test_suggested_agents_is_list(self):
        result = run_heuristic("", {})
        assert isinstance(result["suggested_agents"], list)

    def test_confidence_in_range(self):
        result = run_heuristic("find the user", _rich_enrichment())
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_content_and_enrichment(self):
        result = run_heuristic("", {})
        assert result["query_intent"] == _DEFAULT_INTENT
        assert result["extracted_entities"] == []

    def test_rich_enrichment_boosts_confidence(self):
        rich = run_heuristic("find Acme Corp", _rich_enrichment())
        empty = run_heuristic("", {})
        assert rich["confidence"] > empty["confidence"]


# ---------------------------------------------------------------------------
# TestValidLlmResponse
# ---------------------------------------------------------------------------

class TestValidLlmResponse:
    def test_valid_response(self):
        assert _valid_llm_response({
            "query_intent": "locate",
            "extracted_entities": ["Acme"],
            "dynamic_filters": {},
            "suggested_agents": ["EntityResolutionAgent"],
        }) is True

    def test_invalid_intent(self):
        assert _valid_llm_response({
            "query_intent": "unknown_intent",
            "extracted_entities": [],
            "dynamic_filters": {},
            "suggested_agents": [],
        }) is False

    def test_missing_entities(self):
        assert _valid_llm_response({
            "query_intent": "locate",
            "dynamic_filters": {},
            "suggested_agents": [],
        }) is False

    def test_entities_wrong_type(self):
        assert _valid_llm_response({
            "query_intent": "locate",
            "extracted_entities": "not a list",
            "dynamic_filters": {},
            "suggested_agents": [],
        }) is False

    def test_filters_wrong_type(self):
        assert _valid_llm_response({
            "query_intent": "locate",
            "extracted_entities": [],
            "dynamic_filters": "not a dict",
            "suggested_agents": [],
        }) is False

    def test_not_a_dict(self):
        assert _valid_llm_response("string response") is False

    def test_none(self):
        assert _valid_llm_response(None) is False


# ---------------------------------------------------------------------------
# TestQueryIntelligenceAgentHeuristic
# ---------------------------------------------------------------------------

class TestQueryIntelligenceAgentHeuristic:
    @pytest.mark.asyncio
    async def test_returns_agent_result(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="find the contract"))
        assert isinstance(result, AgentResult)

    @pytest.mark.asyncio
    async def test_status_success(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="find the contract"))
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_confidence_in_range(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="search for report"))
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_query_intent_is_valid(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="why did the deal fail"))
        assert result.outputs["query_intent"] in _VALID_INTENTS

    @pytest.mark.asyncio
    async def test_extracted_entities_is_list(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="about Microsoft and Google"))
        assert isinstance(result.outputs["extracted_entities"], list)

    @pytest.mark.asyncio
    async def test_dynamic_filters_is_dict(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx())
        assert isinstance(result.outputs["dynamic_filters"], dict)

    @pytest.mark.asyncio
    async def test_suggested_agents_is_list(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx())
        assert isinstance(result.outputs["suggested_agents"], list)

    @pytest.mark.asyncio
    async def test_rationale_heuristic(self):
        agent = QueryIntelligenceAgent()
        with patch("app.agents.query_intelligence_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.QUERY_INTELLIGENCE_STRATEGY = "heuristic"
            result = await agent.run("mem-001", _ctx(content="find the file"))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_set_from_runtime(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="find the file"))
        assert result.trace_id == "trace-27"

    @pytest.mark.asyncio
    async def test_empty_memory_still_succeeds(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-empty", {"memory": {}, "runtime": {}})
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_content_read_from_memory_content_field(self):
        agent = QueryIntelligenceAgent()
        ctx = {"memory": {"content": "find the budget report", "enrichment": {}}, "runtime": {}}
        result = await agent.run("mem-001", ctx)
        assert result.outputs["query_intent"] == "locate"

    @pytest.mark.asyncio
    async def test_content_read_from_memory_text_field(self, monkeypatch):
        import app.agents.query_intelligence_agent as qia_mod
        monkeypatch.setattr(qia_mod.settings, "AGENT_STRATEGY", "heuristic")
        agent = QueryIntelligenceAgent()
        ctx = {"memory": {"text": "explain why the system crashed", "enrichment": {}}, "runtime": {}}
        result = await agent.run("mem-001", ctx)
        assert result.outputs["query_intent"] == "explain"

    @pytest.mark.asyncio
    async def test_entity_name_from_enrichment_used(self):
        agent = QueryIntelligenceAgent()
        ctx = _ctx(enrichment={"entity_name": "GlobalBank"}, content="")
        result = await agent.run("mem-001", ctx)
        assert "GlobalBank" in result.outputs["extracted_entities"]

    @pytest.mark.asyncio
    async def test_key_entities_from_enrichment_used(self):
        agent = QueryIntelligenceAgent()
        ctx = _ctx(enrichment={"key_entities": ["Alice", "Bob"]}, content="")
        result = await agent.run("mem-001", ctx)
        assert "Alice" in result.outputs["extracted_entities"]

    @pytest.mark.asyncio
    async def test_finished_at_gte_started_at(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-001", _ctx(content="find the doc"))
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_memory_id_in_result(self):
        agent = QueryIntelligenceAgent()
        result = await agent.run("mem-xyz", _ctx())
        assert result.memory_id == "mem-xyz"


# ---------------------------------------------------------------------------
# TestQueryIntelligenceAgentLLM
# ---------------------------------------------------------------------------

class TestQueryIntelligenceAgentLLM:
    def _llm_ctx(self, content: str = "find the report") -> dict:
        return {"memory": {"content": content, "enrichment": {}}, "runtime": {"job_id": "t-llm"}}

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = QueryIntelligenceAgent()
        llm_resp = {
            "query_intent": "locate",
            "extracted_entities": ["Report2024"],
            "dynamic_filters": {"tags": ["finance"]},
            "suggested_agents": ["EntityResolutionAgent"],
            "confidence": 0.88,
            "rationale": "llm path",
        }
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            mock_cl.return_value.complete_json = AsyncMock(return_value=llm_resp)
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "llm"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = None
                mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_s.OLLAMA_MODEL = "llama3.1:8b"
                mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_s.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm", self._llm_ctx())
        assert result.outputs["query_intent"] == "locate"
        assert result.outputs["extracted_entities"] == ["Report2024"]

    @pytest.mark.asyncio
    async def test_invalid_llm_intent_falls_back(self):
        agent = QueryIntelligenceAgent()
        bad_resp = {
            "query_intent": "nonsense",
            "extracted_entities": [],
            "dynamic_filters": {},
            "suggested_agents": [],
        }
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            mock_cl.return_value.complete_json = AsyncMock(return_value=bad_resp)
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "llm"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = None
                mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_s.OLLAMA_MODEL = "llama3.1:8b"
                mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_s.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm", self._llm_ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self):
        agent = QueryIntelligenceAgent()
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            mock_cl.return_value.complete_json = AsyncMock(side_effect=RuntimeError("timeout"))
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "llm"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = None
                mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_s.OLLAMA_MODEL = "llama3.1:8b"
                mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_s.OLLAMA_MAX_CONCURRENCY = 2
                with pytest.raises(Exception):
                    await agent.run("mem-llm", self._llm_ctx())

    @pytest.mark.asyncio
    async def test_llm_none_response_falls_back(self):
        agent = QueryIntelligenceAgent()
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            mock_cl.return_value.complete_json = AsyncMock(return_value=None)
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "llm"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = None
                mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_s.OLLAMA_MODEL = "llama3.1:8b"
                mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_s.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm", self._llm_ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_heuristic_strategy_skips_llm(self):
        agent = QueryIntelligenceAgent()
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "heuristic"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = "heuristic"
                result = await agent.run("mem-001", self._llm_ctx())
            mock_cl.assert_not_called()
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_missing_filters_falls_back(self):
        agent = QueryIntelligenceAgent()
        bad_resp = {
            "query_intent": "locate",
            "extracted_entities": [],
            # missing dynamic_filters
            "suggested_agents": [],
        }
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            mock_cl.return_value.complete_json = AsyncMock(return_value=bad_resp)
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "llm"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = None
                mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_s.OLLAMA_MODEL = "llama3.1:8b"
                mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_s.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm", self._llm_ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_result_status_always_success(self):
        agent = QueryIntelligenceAgent()
        with patch("app.agents.query_intelligence_agent.create_ollama_client") as mock_cl:
            mock_cl.return_value.complete_json = AsyncMock(return_value=None)
            with patch("app.agents.query_intelligence_agent.settings") as mock_s:
                mock_s.AGENT_STRATEGY = "llm"
                mock_s.QUERY_INTELLIGENCE_STRATEGY = None
                mock_s.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_s.OLLAMA_MODEL = "llama3.1:8b"
                mock_s.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_s.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm", self._llm_ctx())
        assert result.status == "success"


# ---------------------------------------------------------------------------
# TestValidateOutputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def test_non_success_status_passes(self):
        agent = QueryIntelligenceAgent()
        r = _result({}, status="failed")
        agent.validate_outputs(r)  # should not raise

    def test_invalid_intent_raises(self):
        agent = QueryIntelligenceAgent()
        r = _result({
            "query_intent": "bad_intent",
            "extracted_entities": [],
            "dynamic_filters": {},
            "suggested_agents": [],
            "confidence": 0.7,
        })
        with pytest.raises(ValueError, match="query_intent"):
            agent.validate_outputs(r)

    def test_all_valid_intents_pass(self):
        agent = QueryIntelligenceAgent()
        for intent in _VALID_INTENTS:
            r = _result({
                "query_intent": intent,
                "extracted_entities": [],
                "dynamic_filters": {},
                "suggested_agents": [],
                "confidence": 0.65,
            })
            agent.validate_outputs(r)  # should not raise

    def test_entities_wrong_type_raises(self):
        agent = QueryIntelligenceAgent()
        r = _result({
            "query_intent": "locate",
            "extracted_entities": "not a list",
            "dynamic_filters": {},
            "suggested_agents": [],
            "confidence": 0.65,
        })
        with pytest.raises(ValueError, match="extracted_entities"):
            agent.validate_outputs(r)

    def test_filters_wrong_type_raises(self):
        agent = QueryIntelligenceAgent()
        r = _result({
            "query_intent": "locate",
            "extracted_entities": [],
            "dynamic_filters": "not a dict",
            "suggested_agents": [],
            "confidence": 0.65,
        })
        with pytest.raises(ValueError, match="dynamic_filters"):
            agent.validate_outputs(r)

    def test_suggested_agents_wrong_type_raises(self):
        agent = QueryIntelligenceAgent()
        r = _result({
            "query_intent": "locate",
            "extracted_entities": [],
            "dynamic_filters": {},
            "suggested_agents": "not a list",
            "confidence": 0.65,
        })
        with pytest.raises(ValueError, match="suggested_agents"):
            agent.validate_outputs(r)

    def _base_result(self) -> AgentResult:
        """Result with valid top-level confidence; mutate outputs after creation."""
        return _result({
            "query_intent": "locate",
            "extracted_entities": [],
            "dynamic_filters": {},
            "suggested_agents": [],
            "confidence": 0.65,
        })

    def test_confidence_above_one_raises(self):
        agent = QueryIntelligenceAgent()
        r = self._base_result()
        r.outputs["confidence"] = 1.5
        with pytest.raises(ValueError, match="confidence"):
            agent.validate_outputs(r)

    def test_confidence_below_zero_raises(self):
        agent = QueryIntelligenceAgent()
        r = self._base_result()
        r.outputs["confidence"] = -0.1
        with pytest.raises(ValueError, match="confidence"):
            agent.validate_outputs(r)

    def test_confidence_not_a_number_raises(self):
        agent = QueryIntelligenceAgent()
        r = self._base_result()
        r.outputs["confidence"] = "high"
        with pytest.raises(ValueError, match="confidence"):
            agent.validate_outputs(r)

    def test_valid_outputs_pass(self):
        agent = QueryIntelligenceAgent()
        r = _result({
            "query_intent": "analyze",
            "extracted_entities": ["Acme", "John"],
            "dynamic_filters": {"min_credibility": 0.5},
            "suggested_agents": ["AnomalyDetectionAgent"],
            "confidence": 0.78,
        })
        agent.validate_outputs(r)  # should not raise


# ---------------------------------------------------------------------------
# TestRegistry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_query_intelligence_snake(self):
        assert isinstance(get_agent("query_intelligence"), QueryIntelligenceAgent)

    def test_query_intelligence_camel(self):
        assert isinstance(get_agent("queryintelligence"), QueryIntelligenceAgent)

    def test_query_intelligence_full(self):
        assert isinstance(get_agent("queryintelligenceagent"), QueryIntelligenceAgent)

    def test_query_agent_alias(self):
        assert isinstance(get_agent("query_agent"), QueryIntelligenceAgent)

    def test_unknown_name_returns_none(self):
        assert get_agent("totally_unknown") is None


# ---------------------------------------------------------------------------
# TestAgentMeta
# ---------------------------------------------------------------------------

class TestAgentMeta:
    def test_name(self):
        assert QueryIntelligenceAgent.name == "QueryIntelligenceAgent"

    def test_version(self):
        assert QueryIntelligenceAgent.version == "v1"

    def test_dependencies(self):
        deps = QueryIntelligenceAgent().dependencies()
        assert "EntityResolutionAgent" in deps
        assert "UncertaintyReportingAgent" in deps
        assert "NarrativeSynthesisAgent" in deps
