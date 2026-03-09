"""Tests for NarrativeSynthesisAgent — Phase 23."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.narrative_synthesis_agent import (
    NarrativeSynthesisAgent,
    extract_key_entities,
    build_timeline_summary,
    collect_action_items,
    determine_tone,
    build_narrative_text,
    compute_narrative_confidence,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(enrichment: dict | None = None) -> dict:
    return {
        "memory": {"enrichment": enrichment or {}},
        "runtime": {"job_id": "trace-23"},
    }


def _full_enrichment() -> dict:
    return {
        "canonical_entity": "Acme Corp",
        "resolved_entities": [
            {"name": "Acme Corp", "type": "company"},
            {"name": "John Doe", "type": "person"},
        ],
        "episode_label": "Q3 budget review",
        "episode_id": "ep-001",
        "temporal_anchor": "2025-09-01",
        "sequence_position": 3,
        "goal_detected": True,
        "subtasks": ["Collect data", "Analyse spend", "Present to board"],
        "blocking_subtask": "Analyse spend",
        "completion_fraction": 0.33,
        "matched_playbook_id": "pb-budget-001",
        "playbook_confidence": 0.80,
        "uncertainty_level": "medium",
        "review_recommended": True,
        "unresolved_conflicts": [{"entity": "Budget", "severity": "high", "conflict_type": "value"}],
        "conflict_count": 1,
        "escalation_targets": ["finance-lead"],
        "credibility_score": 0.72,
        "source_tier": "internal",
    }


# ---------------------------------------------------------------------------
# TestExtractKeyEntities
# ---------------------------------------------------------------------------

class TestExtractKeyEntities:
    def test_canonical_entity_first(self):
        e = {"canonical_entity": "Acme", "resolved_entities": [{"name": "Acme"}, {"name": "Bob"}]}
        result = extract_key_entities(e)
        assert result[0] == "Acme"

    def test_resolved_entities_dict_list(self):
        e = {"resolved_entities": [{"name": "Alpha"}, {"name": "Beta"}]}
        result = extract_key_entities(e)
        assert "Alpha" in result
        assert "Beta" in result

    def test_resolved_entities_string_list(self):
        e = {"resolved_entities": ["Alice", "Bob", "Charlie"]}
        result = extract_key_entities(e)
        assert result == ["Alice", "Bob", "Charlie"]

    def test_no_duplicates(self):
        e = {
            "canonical_entity": "Acme",
            "resolved_entities": [{"name": "Acme"}, {"name": "Other"}],
        }
        result = extract_key_entities(e)
        assert result.count("Acme") == 1

    def test_cap_at_five(self):
        e = {
            "resolved_entities": [{"name": f"Entity{i}"} for i in range(10)]
        }
        result = extract_key_entities(e)
        assert len(result) <= 5

    def test_empty_enrichment(self):
        assert extract_key_entities({}) == []

    def test_none_values_skipped(self):
        e = {"resolved_entities": [{"name": None}, {"name": "Valid"}]}
        result = extract_key_entities(e)
        assert "Valid" in result
        assert None not in result

    def test_canonical_name_fallback(self):
        e = {"resolved_entities": [{"canonical_name": "CanonCorp"}]}
        result = extract_key_entities(e)
        assert "CanonCorp" in result


# ---------------------------------------------------------------------------
# TestBuildTimelineSummary
# ---------------------------------------------------------------------------

class TestBuildTimelineSummary:
    def test_anchor_only(self):
        result = build_timeline_summary({"temporal_anchor": "2025-01-01"})
        assert result is not None
        assert "2025-01-01" in result

    def test_sequence_only(self):
        result = build_timeline_summary({"sequence_position": 2})
        assert result is not None
        assert "2" in result

    def test_anchor_and_sequence(self):
        result = build_timeline_summary({"temporal_anchor": "2025-06-01", "sequence_position": 5})
        assert "2025-06-01" in result
        assert "5" in result

    def test_empty_enrichment_returns_none(self):
        assert build_timeline_summary({}) is None

    def test_none_anchor_no_seq_returns_none(self):
        assert build_timeline_summary({"temporal_anchor": None}) is None

    def test_result_ends_with_period(self):
        result = build_timeline_summary({"temporal_anchor": "T"})
        assert result.endswith(".")


# ---------------------------------------------------------------------------
# TestCollectActionItems
# ---------------------------------------------------------------------------

class TestCollectActionItems:
    def test_review_recommended(self):
        items = collect_action_items({"review_recommended": True})
        assert any("review" in i.lower() for i in items)

    def test_blocking_subtask(self):
        items = collect_action_items({"blocking_subtask": "Fix schema"})
        assert any("Fix schema" in i for i in items)

    def test_escalation_targets(self):
        items = collect_action_items({"escalation_targets": ["eng-lead", "cto"]})
        assert any("eng-lead" in i for i in items)

    def test_low_credibility(self):
        items = collect_action_items({"credibility_score": 0.20})
        assert any("credib" in i.lower() for i in items)

    def test_credibility_above_threshold_not_flagged(self):
        items = collect_action_items({"credibility_score": 0.75})
        assert not any("credib" in i.lower() for i in items)

    def test_all_signals(self):
        items = collect_action_items({
            "review_recommended": True,
            "blocking_subtask": "Do something",
            "escalation_targets": ["lead"],
            "credibility_score": 0.10,
        })
        assert len(items) == 4

    def test_empty_enrichment_no_items(self):
        assert collect_action_items({}) == []

    def test_escalation_capped_at_three(self):
        targets = ["target-alpha", "target-beta", "target-gamma", "target-delta", "target-epsilon"]
        items = collect_action_items({"escalation_targets": targets})
        escalation_item = next((i for i in items if "Escalate" in i), None)
        assert escalation_item is not None
        assert "target-delta" not in escalation_item
        assert "target-epsilon" not in escalation_item


# ---------------------------------------------------------------------------
# TestDetermineTone
# ---------------------------------------------------------------------------

class TestDetermineTone:
    def test_critical_uncertainty_is_urgent(self):
        assert determine_tone({"uncertainty_level": "critical"}) == "urgent"

    def test_high_conflict_count_is_urgent(self):
        assert determine_tone({"conflict_count": 3}) == "urgent"

    def test_high_uncertainty_is_cautionary(self):
        assert determine_tone({"uncertainty_level": "high"}) == "cautionary"

    def test_review_recommended_is_cautionary(self):
        assert determine_tone({"review_recommended": True}) == "cautionary"

    def test_one_conflict_is_cautionary(self):
        assert determine_tone({"conflict_count": 1}) == "cautionary"

    def test_clean_enrichment_is_informational(self):
        assert determine_tone({}) == "informational"


# ---------------------------------------------------------------------------
# TestBuildNarrativeText
# ---------------------------------------------------------------------------

class TestBuildNarrativeText:
    def test_uses_episode_label_as_subject(self):
        e = {"episode_label": "Q3 review"}
        text = build_narrative_text(e, [], None, "informational")
        assert "Q3 review" in text

    def test_uses_canonical_entity_when_no_label(self):
        e = {"canonical_entity": "Acme"}
        text = build_narrative_text(e, [], None, "informational")
        assert "Acme" in text

    def test_uses_first_entity_as_fallback(self):
        text = build_narrative_text({}, ["FallbackCo"], None, "informational")
        assert "FallbackCo" in text

    def test_default_opening_when_no_subject(self):
        text = build_narrative_text({}, [], None, "informational")
        assert "enterprise event" in text.lower()

    def test_timeline_included_when_provided(self):
        text = build_narrative_text({}, [], "Temporal context: anchored at 2025-01-01.", "informational")
        assert "2025-01-01" in text

    def test_goal_progress_included(self):
        e = {"goal_detected": True, "completion_fraction": 0.50, "blocking_subtask": "Step X"}
        text = build_narrative_text(e, [], None, "informational")
        assert "50%" in text
        assert "Step X" in text

    def test_playbook_included_above_threshold(self):
        e = {"matched_playbook_id": "pb-001", "playbook_confidence": 0.70}
        text = build_narrative_text(e, [], None, "informational")
        assert "pb-001" in text

    def test_playbook_excluded_below_threshold(self):
        e = {"matched_playbook_id": "pb-001", "playbook_confidence": 0.40}
        text = build_narrative_text(e, [], None, "informational")
        assert "pb-001" not in text

    def test_urgent_tone_with_conflicts(self):
        e = {"conflict_count": 4}
        text = build_narrative_text(e, [], None, "urgent")
        assert "urgent" in text.lower() or "immediate" in text.lower()

    def test_urgent_tone_no_conflicts(self):
        text = build_narrative_text({}, [], None, "urgent")
        assert "critical" in text.lower() or "immediate" in text.lower()


# ---------------------------------------------------------------------------
# TestComputeNarrativeConfidence
# ---------------------------------------------------------------------------

class TestComputeNarrativeConfidence:
    def test_rich_enrichment_boosts_confidence(self):
        e = {
            "episode_label": "X",
            "temporal_anchor": "T",
            "goal_detected": True,
            "uncertainty_level": "medium",
        }
        conf = compute_narrative_confidence(e, ["EntityA"])
        assert conf >= 0.75

    def test_empty_enrichment_penalised(self):
        conf = compute_narrative_confidence({}, [])
        assert conf < 0.60

    def test_always_in_range(self):
        for enrichment, entities in [
            ({}, []),
            ({"episode_label": "X", "temporal_anchor": "T", "goal_detected": True,
              "uncertainty_level": "low", "canonical_entity": "A"}, ["A", "B", "C"]),
        ]:
            conf = compute_narrative_confidence(enrichment, entities)
            assert 0.40 <= conf <= 0.90

    def test_key_entities_boost(self):
        base = compute_narrative_confidence({}, [])
        boosted = compute_narrative_confidence({}, ["EntityA"])
        assert boosted > base

    def test_uncertainty_level_boosts(self):
        base = compute_narrative_confidence({}, [])
        boosted = compute_narrative_confidence({"uncertainty_level": "high"}, [])
        assert boosted > base


# ---------------------------------------------------------------------------
# TestRunHeuristic
# ---------------------------------------------------------------------------

class TestRunHeuristic:
    def test_returns_required_keys(self):
        result = run_heuristic({})
        for key in ("narrative_text", "key_entities", "timeline_summary", "action_items", "tone", "confidence", "rationale"):
            assert key in result

    def test_rationale_is_heuristic(self):
        assert run_heuristic({})["rationale"] == "heuristic"

    def test_tone_valid(self):
        assert run_heuristic({})["tone"] in {"informational", "cautionary", "urgent"}

    def test_confidence_in_range(self):
        conf = run_heuristic(_full_enrichment())["confidence"]
        assert 0.40 <= conf <= 0.90

    def test_full_enrichment_produces_narrative(self):
        result = run_heuristic(_full_enrichment())
        assert len(result["narrative_text"]) > 20
        assert len(result["key_entities"]) >= 1

    def test_urgent_on_critical_enrichment(self):
        e = {"uncertainty_level": "critical", "conflict_count": 5}
        assert run_heuristic(e)["tone"] == "urgent"

    def test_action_items_list(self):
        result = run_heuristic({"review_recommended": True})
        assert isinstance(result["action_items"], list)
        assert len(result["action_items"]) >= 1

    def test_no_timeline_when_no_temporal(self):
        result = run_heuristic({"canonical_entity": "Corp"})
        assert result["timeline_summary"] is None


# ---------------------------------------------------------------------------
# TestNarrativeSynthesisAgentHeuristic
# ---------------------------------------------------------------------------

class TestNarrativeSynthesisAgentHeuristic:
    @pytest.mark.asyncio
    async def test_runs_heuristic_strategy(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_outputs_contain_required_fields(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        outputs = result.outputs
        for key in ("narrative_text", "key_entities", "timeline_summary", "action_items", "tone", "confidence"):
            assert key in outputs

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.trace_id == "trace-23"

    @pytest.mark.asyncio
    async def test_timestamps_set(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_memory_id_preserved(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-xyz", _ctx())
        assert result.memory_id == "mem-xyz"

    @pytest.mark.asyncio
    async def test_confidence_in_valid_range(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        assert 0.40 <= result.confidence <= 0.90

    @pytest.mark.asyncio
    async def test_empty_enrichment_succeeds(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx({}))
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_no_trace_id_when_missing(self):
        agent = NarrativeSynthesisAgent()
        ctx = {"memory": {"enrichment": {}}, "runtime": {}}
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_urgent_tone_on_crisis_enrichment(self):
        agent = NarrativeSynthesisAgent()
        enrichment = {"uncertainty_level": "critical", "conflict_count": 5}
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(enrichment))
        assert result.outputs["tone"] == "urgent"

    @pytest.mark.asyncio
    async def test_action_items_list_type(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        assert isinstance(result.outputs["action_items"], list)

    @pytest.mark.asyncio
    async def test_key_entities_list_type(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        assert isinstance(result.outputs["key_entities"], list)

    @pytest.mark.asyncio
    async def test_agent_strategy_fallback_used(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_narrative_text_non_empty(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        assert result.outputs["narrative_text"].strip()

    @pytest.mark.asyncio
    async def test_agent_name(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.agent_name == "NarrativeSynthesisAgent"

    @pytest.mark.asyncio
    async def test_warnings_empty(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx())
        assert result.warnings == []


# ---------------------------------------------------------------------------
# TestNarrativeSynthesisAgentLLM
# ---------------------------------------------------------------------------

def _valid_llm_response() -> dict:
    return {
        "narrative_text": "This memory concerns Acme Corp. A goal is 50% complete.",
        "key_entities": ["Acme Corp"],
        "timeline_summary": "Temporal context: anchored at 2025-01-01.",
        "action_items": ["Human review recommended."],
        "tone": "cautionary",
        "confidence": 0.75,
        "rationale": "LLM assessed enrichment signals.",
    }


class TestNarrativeSynthesisAgentLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = NarrativeSynthesisAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=_valid_llm_response())
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        assert result.status == "success"
        assert result.outputs["narrative_text"] == _valid_llm_response()["narrative_text"]

    @pytest.mark.asyncio
    async def test_llm_returns_none_falls_back(self):
        agent = NarrativeSynthesisAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=None)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_invalid_tone_falls_back(self):
        agent = NarrativeSynthesisAgent()
        bad = dict(_valid_llm_response(), tone="unknown_tone")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_missing_narrative_text_falls_back(self):
        agent = NarrativeSynthesisAgent()
        bad = {k: v for k, v in _valid_llm_response().items() if k != "narrative_text"}
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_empty_narrative_text_falls_back(self):
        agent = NarrativeSynthesisAgent()
        bad = dict(_valid_llm_response(), narrative_text="   ")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_wrong_key_entities_type_falls_back(self):
        agent = NarrativeSynthesisAgent()
        bad = dict(_valid_llm_response(), key_entities="not-a-list")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_wrong_action_items_type_falls_back(self):
        agent = NarrativeSynthesisAgent()
        bad = dict(_valid_llm_response(), action_items="not-a-list")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=bad)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_response_confidence_used(self):
        agent = NarrativeSynthesisAgent()
        resp = dict(_valid_llm_response(), confidence=0.88)
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=resp)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.confidence == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_llm_urgent_tone_accepted(self):
        agent = NarrativeSynthesisAgent()
        resp = dict(_valid_llm_response(), tone="urgent")
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=resp)
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["tone"] == "urgent"

    @pytest.mark.asyncio
    async def test_default_strategy_goes_llm_path(self):
        agent = NarrativeSynthesisAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=_valid_llm_response())
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        mock_client.complete_json.assert_awaited_once()
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_llm_non_dict_response_falls_back(self):
        agent = NarrativeSynthesisAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value="just a string")
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings, \
             patch("app.agents.narrative_synthesis_agent.create_ollama_client", return_value=mock_client):
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", _ctx())
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TestNarrativeSynthesisAgentValidateOutputs
# ---------------------------------------------------------------------------

class TestNarrativeSynthesisAgentValidateOutputs:
    def _agent(self):
        return NarrativeSynthesisAgent()

    def _result(self, outputs: dict) -> AgentResult:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="NarrativeSynthesisAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.70,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "narrative_text": "This memory concerns Acme Corp.",
            "key_entities": ["Acme Corp"],
            "timeline_summary": None,
            "action_items": [],
            "tone": "informational",
            "confidence": 0.70,
            "rationale": "heuristic",
        }

    def test_valid_outputs_pass(self):
        agent = self._agent()
        agent.validate_outputs(self._result(self._valid_outputs()))  # no exception

    def test_invalid_tone_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="tone"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), tone="weird")))

    def test_missing_narrative_text_raises(self):
        agent = self._agent()
        outputs = {k: v for k, v in self._valid_outputs().items() if k != "narrative_text"}
        with pytest.raises(ValueError, match="narrative_text"):
            agent.validate_outputs(self._result(outputs))

    def test_empty_narrative_text_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="narrative_text"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), narrative_text="")))

    def test_key_entities_not_list_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="key_entities"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), key_entities="oops")))

    def test_action_items_not_list_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="action_items"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), action_items="oops")))

    def test_timeline_summary_non_string_raises(self):
        agent = self._agent()
        with pytest.raises(ValueError, match="timeline_summary"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), timeline_summary=123)))

    def test_timeline_summary_none_passes(self):
        agent = self._agent()
        agent.validate_outputs(self._result(dict(self._valid_outputs(), timeline_summary=None)))

    def test_timeline_summary_string_passes(self):
        agent = self._agent()
        agent.validate_outputs(self._result(dict(self._valid_outputs(), timeline_summary="some text")))

    def test_failed_status_skips_validation(self):
        agent = self._agent()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="NarrativeSynthesisAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        agent.validate_outputs(result)  # should not raise

    def test_cautionary_tone_passes(self):
        agent = self._agent()
        agent.validate_outputs(self._result(dict(self._valid_outputs(), tone="cautionary")))

    def test_urgent_tone_passes(self):
        agent = self._agent()
        agent.validate_outputs(self._result(dict(self._valid_outputs(), tone="urgent")))


# ---------------------------------------------------------------------------
# TestNarrativeSynthesisAgentMetadata
# ---------------------------------------------------------------------------

class TestNarrativeSynthesisAgentMetadata:
    def test_name(self):
        assert NarrativeSynthesisAgent.name == "NarrativeSynthesisAgent"

    def test_version(self):
        assert NarrativeSynthesisAgent.version == "v1"

    def test_dependencies_include_uncertainty_reporting(self):
        agent = NarrativeSynthesisAgent()
        assert "UncertaintyReportingAgent" in agent.dependencies()

    def test_dependencies_include_entity_resolution(self):
        agent = NarrativeSynthesisAgent()
        assert "EntityResolutionAgent" in agent.dependencies()

    def test_dependencies_is_list(self):
        agent = NarrativeSynthesisAgent()
        assert isinstance(agent.dependencies(), list)


# ---------------------------------------------------------------------------
# TestNarrativeSynthesisRegistry
# ---------------------------------------------------------------------------

class TestNarrativeSynthesisRegistry:
    def test_narrative_synthesis(self):
        assert isinstance(get_agent("narrative_synthesis"), NarrativeSynthesisAgent)

    def test_narrativesynthesis(self):
        assert isinstance(get_agent("narrativesynthesis"), NarrativeSynthesisAgent)

    def test_narrativesynthesisagent(self):
        assert isinstance(get_agent("narrativesynthesisagent"), NarrativeSynthesisAgent)

    def test_narrative_agent(self):
        assert isinstance(get_agent("narrative_agent"), NarrativeSynthesisAgent)

    def test_unknown_returns_none(self):
        assert get_agent("not_a_real_agent_xyz") is None

    def test_mixed_case_insensitive(self):
        assert isinstance(get_agent("NarrativeSynthesis"), NarrativeSynthesisAgent)


# ---------------------------------------------------------------------------
# TestNarrativeSynthesisIntegration
# ---------------------------------------------------------------------------

class TestNarrativeSynthesisIntegration:
    @pytest.mark.asyncio
    async def test_briefing_ready_memory_informational(self):
        """Rich, clean memory → informational tone, no action items."""
        enrichment = {
            "canonical_entity": "Acme Corp",
            "resolved_entities": [{"name": "Acme Corp"}],
            "episode_label": "Annual strategy review",
            "temporal_anchor": "2025-01-10",
            "sequence_position": 1,
            "goal_detected": True,
            "completion_fraction": 1.0,
            "blocking_subtask": None,
            "matched_playbook_id": "pb-strategy-001",
            "playbook_confidence": 0.85,
            "uncertainty_level": "low",
            "review_recommended": False,
            "conflict_count": 0,
            "credibility_score": 0.90,
        }
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-brief", _ctx(enrichment))
        assert result.status == "success"
        assert result.outputs["tone"] == "informational"
        assert result.outputs["action_items"] == []

    @pytest.mark.asyncio
    async def test_crisis_memory_urgent_with_action_items(self):
        """Critical uncertainty + many conflicts → urgent tone + action items."""
        enrichment = {
            "uncertainty_level": "critical",
            "review_recommended": True,
            "conflict_count": 5,
            "escalation_targets": ["cto", "legal"],
            "blocking_subtask": "Resolve contract dispute",
            "credibility_score": 0.20,
        }
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-crisis", _ctx(enrichment))
        assert result.outputs["tone"] == "urgent"
        assert len(result.outputs["action_items"]) >= 3

    @pytest.mark.asyncio
    async def test_minimal_memory_still_produces_output(self):
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-empty", _ctx({}))
        assert result.status == "success"
        assert result.outputs["narrative_text"]
        assert result.outputs["tone"] == "informational"

    @pytest.mark.asyncio
    async def test_json_serialisable_outputs(self):
        import json
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_full_enrichment()))
        json.dumps(result.outputs)  # must not raise

    @pytest.mark.asyncio
    async def test_goal_progress_in_narrative(self):
        enrichment = {"goal_detected": True, "completion_fraction": 0.75, "blocking_subtask": None}
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-goal", _ctx(enrichment))
        assert "75%" in result.outputs["narrative_text"]

    @pytest.mark.asyncio
    async def test_playbook_in_narrative_when_high_confidence(self):
        enrichment = {"matched_playbook_id": "pb-deploy-001", "playbook_confidence": 0.90}
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-pb", _ctx(enrichment))
        assert "pb-deploy-001" in result.outputs["narrative_text"]

    @pytest.mark.asyncio
    async def test_playbook_absent_when_low_confidence(self):
        enrichment = {"matched_playbook_id": "pb-deploy-001", "playbook_confidence": 0.30}
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-pb", _ctx(enrichment))
        assert "pb-deploy-001" not in result.outputs["narrative_text"]

    @pytest.mark.asyncio
    async def test_timeline_in_narrative_when_anchor_present(self):
        enrichment = {"temporal_anchor": "2025-03-15"}
        agent = NarrativeSynthesisAgent()
        with patch("app.agents.narrative_synthesis_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_SYNTHESIS_STRATEGY = "heuristic"
            result = await agent.run("mem-time", _ctx(enrichment))
        assert result.outputs["timeline_summary"] is not None
        assert "2025-03-15" in result.outputs["timeline_summary"]
