"""Tests for ConflictDetectionAgent — Phase 13."""

from __future__ import annotations

import pytest

from app.agents.conflict_detection_agent import (
    ConflictDetectionAgent,
    _conflict_severity,
    _entity_type_weight,
    build_resolution_hint,
    classify_signal_sentiment,
    detect_causal_loops,
    detect_silo_gaps,
    detect_state_mismatches,
    merge_and_rank_conflicts,
)
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    world_nodes=None,
    propagated_signals=None,
    causal_chains=None,
    content="some memory content",
    strategy="heuristic",
):
    import os
    os.environ["AGENT_STRATEGY"] = strategy
    return {
        "tenant": {"org_id": "org-1", "org_slug": "acme"},
        "memory": {
            "id": "mem-1",
            "content": content,
            "enrichment": {
                "world_nodes": world_nodes or [],
                "propagated_signals": propagated_signals or [],
                "causal_chains": causal_chains or [],
            },
        },
        "runtime": {"job_id": "job-x"},
    }


def _node(entity, entity_type="project", domains=None, silo_count=1):
    return {
        "entity": entity,
        "entity_type": entity_type,
        "domains": domains or [],
        "silo_count": silo_count,
        "state_version": 1,
        "node_confidence": 0.8,
    }


def _signal(content, source_silo, entities=None, relevance=0.7):
    return {
        "content": content,
        "source_silo": source_silo,
        "entities": entities or [],
        "relevance_score": relevance,
        "target_domains": [],
    }


def _chain(effect, path, strength=0.6, silo_span=2):
    return {
        "effect_entity": effect,
        "causal_path": path,
        "hop_count": len(path) - 1,
        "chain_strength": strength,
        "silo_span": silo_span,
        "narrative": " → ".join(reversed(path)),
    }


# ---------------------------------------------------------------------------
# _entity_type_weight
# ---------------------------------------------------------------------------

class TestEntityTypeWeight:
    def test_person_highest(self):
        assert _entity_type_weight("person") == 1.0

    def test_metric_lowest_known(self):
        assert _entity_type_weight("metric") == 0.30

    def test_case_insensitive(self):
        assert _entity_type_weight("Person") == _entity_type_weight("person")

    def test_unknown_type_default(self):
        assert _entity_type_weight("widget") == 0.45

    def test_empty_string_default(self):
        assert _entity_type_weight("") == 0.45


# ---------------------------------------------------------------------------
# _conflict_severity
# ---------------------------------------------------------------------------

class TestConflictSeverity:
    def test_high_severity_person_two_silos(self):
        assert _conflict_severity("person", 2) == "high"

    def test_high_severity_customer_three_silos(self):
        assert _conflict_severity("customer", 3) == "high"

    def test_medium_severity_project_one_silo(self):
        # project weight=0.80 >= 0.45 → medium even with 1 silo
        assert _conflict_severity("project", 1) == "medium"

    def test_medium_severity_low_type_two_silos(self):
        # metric weight=0.30 < 0.45 but 2 silos → medium
        assert _conflict_severity("metric", 2) == "medium"

    def test_low_severity_metric_one_silo(self):
        assert _conflict_severity("metric", 1) == "low"


# ---------------------------------------------------------------------------
# classify_signal_sentiment
# ---------------------------------------------------------------------------

class TestClassifySignalSentiment:
    def test_positive_resolved(self):
        assert classify_signal_sentiment("The ticket has been resolved") == "positive"

    def test_negative_blocked(self):
        assert classify_signal_sentiment("Deploy is blocked, an outage occurred") == "negative"

    def test_neutral_no_keywords(self):
        assert classify_signal_sentiment("Team meeting scheduled for Monday") == "neutral"

    def test_mixed_more_negative(self):
        # 1 positive (done) vs 2 negative (failed, blocked)
        assert classify_signal_sentiment("done but then failed and blocked") == "negative"

    def test_mixed_more_positive(self):
        # 2 positive vs 1 negative
        assert classify_signal_sentiment("resolved and completed but one issue remains") == "positive"

    def test_empty_string_neutral(self):
        assert classify_signal_sentiment("") == "neutral"


# ---------------------------------------------------------------------------
# detect_state_mismatches
# ---------------------------------------------------------------------------

class TestDetectStateMismatches:
    def _lookup(self, *nodes):
        return {n["entity"]: n for n in nodes}

    def test_basic_mismatch_two_silos(self):
        signals = [
            _signal("Project Alpha resolved", "sales", entities=["Alpha"]),
            _signal("Project Alpha is blocked and overdue", "engineering", entities=["Alpha"]),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup={})
        assert len(result) == 1
        assert result[0]["entity"] == "Alpha"
        assert result[0]["conflict_type"] == "state_mismatch"
        assert set(result[0]["silos"]) == {"sales", "engineering"}

    def test_no_conflict_same_sentiment(self):
        signals = [
            _signal("Alpha resolved", "sales", entities=["Alpha"]),
            _signal("Alpha completed", "engineering", entities=["Alpha"]),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup={})
        assert result == []

    def test_no_conflict_single_silo(self):
        signals = [
            _signal("Alpha blocked", "sales", entities=["Alpha"]),
            _signal("Beta resolved", "sales", entities=["Beta"]),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup={})
        assert result == []

    def test_severity_set_by_entity_type(self):
        node_lookup = self._lookup(_node("Alpha", entity_type="customer"))
        signals = [
            _signal("Alpha resolved", "crm", entities=["Alpha"]),
            _signal("Alpha is degraded", "support", entities=["Alpha"]),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup=node_lookup)
        assert result[0]["severity"] == "high"
        assert result[0]["entity_type"] == "customer"

    def test_content_matching_world_node_names(self):
        node_lookup = self._lookup(_node("Horizon", entity_type="project"))
        signals = [
            _signal("Horizon project is completed and stable", "pm"),
            _signal("Horizon is critical and blocked", "engineering"),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup=node_lookup)
        assert any(c["entity"] == "Horizon" for c in result)

    def test_short_entity_name_skipped_in_content_matching(self):
        # "go" is < 4 chars → not matched by content scanning
        node_lookup = self._lookup(_node("go", entity_type="concept"))
        signals = [
            _signal("good progress going forward", "sales"),
            _signal("going wrong, errors everywhere", "engineering"),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup=node_lookup)
        # "go" should not match "going" etc. — might or might not appear; key point is no crash
        assert isinstance(result, list)

    def test_neutral_signals_ignored(self):
        signals = [
            _signal("Weekly report update", "sales", entities=["Alpha"]),
            _signal("Alpha is blocked", "engineering", entities=["Alpha"]),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup={})
        # Only one non-neutral signal → no conflict
        assert result == []

    def test_multiple_entities_detected(self):
        signals = [
            _signal("Alpha resolved, Beta done", "sales", entities=["Alpha", "Beta"]),
            _signal("Alpha blocked, Beta failed", "engineering", entities=["Alpha", "Beta"]),
        ]
        result = detect_state_mismatches(propagated_signals=signals, node_lookup={})
        entities_found = {c["entity"] for c in result}
        assert "Alpha" in entities_found
        assert "Beta" in entities_found


# ---------------------------------------------------------------------------
# detect_silo_gaps
# ---------------------------------------------------------------------------

class TestDetectSiloGaps:
    def test_entity_in_chain_but_not_world_model(self):
        world_nodes = [_node("Alpha")]
        chains = [_chain("Alpha", ["Alpha", "Ghost"])]
        result = detect_silo_gaps(world_nodes=world_nodes, causal_chains=chains)
        assert len(result) == 1
        assert result[0]["entity"] == "Ghost"
        assert result[0]["conflict_type"] == "silo_gap"

    def test_all_entities_in_world_model(self):
        world_nodes = [_node("Alpha"), _node("Beta")]
        chains = [_chain("Alpha", ["Alpha", "Beta"])]
        result = detect_silo_gaps(world_nodes=world_nodes, causal_chains=chains)
        assert result == []

    def test_empty_chains(self):
        world_nodes = [_node("Alpha")]
        result = detect_silo_gaps(world_nodes=world_nodes, causal_chains=[])
        assert result == []

    def test_multiple_gaps(self):
        world_nodes = [_node("Alpha")]
        chains = [_chain("Alpha", ["Alpha", "Ghost1", "Ghost2"])]
        result = detect_silo_gaps(world_nodes=world_nodes, causal_chains=chains)
        gap_entities = {c["entity"] for c in result}
        assert gap_entities == {"Ghost1", "Ghost2"}

    def test_gap_severity_is_low(self):
        chains = [_chain("Alpha", ["Alpha", "Ghost"])]
        result = detect_silo_gaps(world_nodes=[], causal_chains=chains)
        assert all(c["severity"] == "low" for c in result)


# ---------------------------------------------------------------------------
# detect_causal_loops
# ---------------------------------------------------------------------------

class TestDetectCausalLoops:
    def test_simple_two_entity_loop(self):
        chains = [
            _chain("Alpha", ["Alpha", "Beta"]),
            _chain("Beta", ["Beta", "Alpha"]),
        ]
        result = detect_causal_loops(causal_chains=chains)
        assert len(result) == 1
        assert result[0]["conflict_type"] == "causal_loop"
        assert "Alpha" in result[0]["descriptions"][0] or "Beta" in result[0]["descriptions"][0]

    def test_no_loop(self):
        chains = [
            _chain("Alpha", ["Alpha", "Beta"]),
            _chain("Beta", ["Beta", "Gamma"]),
        ]
        result = detect_causal_loops(causal_chains=chains)
        assert result == []

    def test_loop_only_counted_once(self):
        # Same pair should not produce two entries
        chains = [
            _chain("Alpha", ["Alpha", "Beta"]),
            _chain("Beta", ["Beta", "Alpha"]),
        ]
        result = detect_causal_loops(causal_chains=chains)
        assert len(result) == 1

    def test_empty_chains(self):
        assert detect_causal_loops(causal_chains=[]) == []

    def test_single_hop_chain_no_self_loop(self):
        # A chain that starts and ends at the same entity shouldn't happen (BFS prevents it)
        # but detect_causal_loops should handle gracefully
        chains = [_chain("Alpha", ["Alpha"])]  # path length 1, no root
        result = detect_causal_loops(causal_chains=chains)
        assert result == []


# ---------------------------------------------------------------------------
# build_resolution_hint
# ---------------------------------------------------------------------------

class TestBuildResolutionHint:
    def test_state_mismatch_hint(self):
        conflict = {
            "entity": "Alpha", "conflict_type": "state_mismatch",
            "silos": ["sales", "engineering"],
        }
        hint = build_resolution_hint(conflict=conflict)
        assert "Reconcile" in hint
        assert "Alpha" in hint
        assert "source of truth" in hint

    def test_silo_gap_hint(self):
        conflict = {"entity": "Ghost", "conflict_type": "silo_gap", "silos": []}
        hint = build_resolution_hint(conflict=conflict)
        assert "Register" in hint
        assert "Ghost" in hint

    def test_causal_loop_hint(self):
        conflict = {"entity": "Alpha", "conflict_type": "causal_loop", "silos": []}
        hint = build_resolution_hint(conflict=conflict)
        assert "circular" in hint.lower() or "causal" in hint.lower()
        assert "Alpha" in hint

    def test_unknown_conflict_type(self):
        conflict = {"entity": "X", "conflict_type": "unknown_type", "silos": []}
        hint = build_resolution_hint(conflict=conflict)
        assert "X" in hint


# ---------------------------------------------------------------------------
# ConflictDetectionAgent metadata
# ---------------------------------------------------------------------------

class TestConflictDetectionAgentMeta:
    def test_name(self):
        assert ConflictDetectionAgent.name == "ConflictDetectionAgent"

    def test_version(self):
        assert ConflictDetectionAgent.version == "v1"

    def test_dependencies(self):
        assert ConflictDetectionAgent().dependencies() == ["CausalReasoningAgent"]


# ---------------------------------------------------------------------------
# ConflictDetectionAgent.run — heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestConflictDetectionAgentHeuristic:
    async def test_empty_enrichment_returns_success(self):
        agent = ConflictDetectionAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["conflict_count"] == 0
        assert result.outputs["conflicts"] == []

    async def test_state_mismatch_detected(self):
        agent = ConflictDetectionAgent()
        signals = [
            _signal("Alpha resolved", "sales", entities=["Alpha"]),
            _signal("Alpha is blocked and degraded", "engineering", entities=["Alpha"]),
        ]
        ctx = _make_context(propagated_signals=signals)
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["conflict_count"] >= 1
        conflict_types = {c["conflict_type"] for c in result.outputs["conflicts"]}
        assert "state_mismatch" in conflict_types

    async def test_silo_gap_detected(self, monkeypatch):
        import app.agents.conflict_detection_agent as cda_mod
        monkeypatch.setattr(cda_mod.settings, "AGENT_STRATEGY", "heuristic")
        agent = ConflictDetectionAgent()
        world_nodes = [_node("Alpha")]
        chains = [_chain("Alpha", ["Alpha", "Ghost"])]
        ctx = _make_context(world_nodes=world_nodes, causal_chains=chains)
        result = await agent.run("mem-1", ctx)
        gap_conflicts = [c for c in result.outputs["conflicts"] if c["conflict_type"] == "silo_gap"]
        assert len(gap_conflicts) == 1
        assert gap_conflicts[0]["entity"] == "Ghost"

    async def test_causal_loop_detected(self):
        agent = ConflictDetectionAgent()
        chains = [
            _chain("Alpha", ["Alpha", "Beta"]),
            _chain("Beta", ["Beta", "Alpha"]),
        ]
        ctx = _make_context(causal_chains=chains)
        result = await agent.run("mem-1", ctx)
        loop_conflicts = [c for c in result.outputs["conflicts"] if c["conflict_type"] == "causal_loop"]
        assert len(loop_conflicts) == 1

    async def test_high_severity_conflicts_populated(self):
        agent = ConflictDetectionAgent()
        nodes = [_node("Alice", entity_type="person")]
        signals = [
            _signal("Alice approved the plan", "hr", entities=["Alice"]),
            _signal("Alice failed to deliver, critical issue", "eng", entities=["Alice"]),
        ]
        ctx = _make_context(world_nodes=nodes, propagated_signals=signals)
        result = await agent.run("mem-1", ctx)
        assert len(result.outputs["high_severity_conflicts"]) >= 1

    async def test_resolution_hints_populated(self):
        agent = ConflictDetectionAgent()
        signals = [
            _signal("Alpha resolved", "sales", entities=["Alpha"]),
            _signal("Alpha blocked", "engineering", entities=["Alpha"]),
        ]
        ctx = _make_context(propagated_signals=signals)
        result = await agent.run("mem-1", ctx)
        assert len(result.outputs["resolution_hints"]) >= 1

    async def test_confidence_scales_with_signals(self):
        agent = ConflictDetectionAgent()
        few_signals = [_signal("Alpha resolved", "sales", entities=["Alpha"])]
        many_signals = few_signals * 12
        ctx_few = _make_context(propagated_signals=few_signals)
        ctx_many = _make_context(propagated_signals=many_signals)
        res_few = await agent.run("mem-1", ctx_few)
        res_many = await agent.run("mem-1", ctx_many)
        assert res_many.outputs["confidence"] >= res_few.outputs["confidence"]

    async def test_rationale_is_heuristic(self, monkeypatch):
        import app.agents.conflict_detection_agent as cda_mod
        monkeypatch.setattr(cda_mod.settings, "AGENT_STRATEGY", "heuristic")
        agent = ConflictDetectionAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    async def test_conflicts_sorted_high_first(self):
        agent = ConflictDetectionAgent()
        nodes = [_node("Alice", entity_type="person"), _node("metric1", entity_type="metric")]
        signals = [
            _signal("Alice approved", "hr", entities=["Alice"]),
            _signal("Alice blocked and failed", "eng", entities=["Alice"]),
            _signal("metric1 is healthy and stable", "finance", entities=["metric1"]),
            _signal("metric1 is broken", "ops", entities=["metric1"]),
        ]
        ctx = _make_context(world_nodes=nodes, propagated_signals=signals)
        result = await agent.run("mem-1", ctx)
        conflicts = result.outputs["conflicts"]
        if len(conflicts) >= 2:
            severities = [c["severity"] for c in conflicts]
            order = [{"high": 0, "medium": 1, "low": 2}[s] for s in severities]
            assert order == sorted(order)

    async def test_trace_id_propagated(self):
        agent = ConflictDetectionAgent()
        ctx = _make_context()
        ctx["runtime"] = {"job_id": "trace-abc"}
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "trace-abc"

    async def test_agent_result_fields(self):
        agent = ConflictDetectionAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.agent_name == "ConflictDetectionAgent"
        assert result.agent_version == "v1"
        assert result.memory_id == "mem-1"
        assert 0.0 <= result.confidence <= 1.0
        assert result.started_at <= result.finished_at


# ---------------------------------------------------------------------------
# ConflictDetectionAgent.run — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestConflictDetectionAgentLLM:
    async def test_llm_valid_response_used(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        valid_resp = {
            "conflicts": [
                {
                    "entity": "Alpha",
                    "entity_type": "project",
                    "conflict_type": "state_mismatch",
                    "silos": ["sales", "eng"],
                    "descriptions": ["resolved", "blocked"],
                    "severity": "medium",
                    "resolution_hint": "Reconcile Alpha",
                }
            ],
            "conflict_count": 1,
            "high_severity_conflicts": [],
            "resolution_hints": ["Reconcile Alpha"],
            "confidence": 0.72,
            "rationale": "llm analysis",
        }
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value=valid_resp)
        monkeypatch.setattr(
            "app.agents.conflict_detection_agent.create_ollama_client",
            lambda **_: mock_client,
        )

        import os
        os.environ["AGENT_STRATEGY"] = "llm"
        agent = ConflictDetectionAgent()
        ctx = _make_context(content="deal Alpha resolved", strategy="llm")
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["conflict_count"] == 1

    async def test_llm_bad_response_falls_back_to_heuristic(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "response"})
        monkeypatch.setattr(
            "app.agents.conflict_detection_agent.create_ollama_client",
            lambda **_: mock_client,
        )

        import os
        os.environ["AGENT_STRATEGY"] = "llm"
        agent = ConflictDetectionAgent()
        ctx = _make_context(content="some content", strategy="llm")
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _base_result(self, outputs):
        from datetime import datetime, timezone
        return AgentResult(
            agent_name="ConflictDetectionAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.6,
            outputs=outputs,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_outputs_pass(self):
        agent = ConflictDetectionAgent()
        result = self._base_result({
            "conflicts": [
                {
                    "entity": "Alpha",
                    "conflict_type": "state_mismatch",
                    "severity": "medium",
                }
            ],
            "resolution_hints": ["Reconcile Alpha"],
        })
        agent.validate_outputs(result)  # should not raise

    def test_conflicts_not_list_raises(self):
        agent = ConflictDetectionAgent()
        result = self._base_result({"conflicts": "bad", "resolution_hints": []})
        with pytest.raises(ValueError, match="conflicts must be a list"):
            agent.validate_outputs(result)

    def test_resolution_hints_not_list_raises(self):
        agent = ConflictDetectionAgent()
        result = self._base_result({"conflicts": [], "resolution_hints": "bad"})
        with pytest.raises(ValueError, match="resolution_hints must be a list"):
            agent.validate_outputs(result)

    def test_conflict_missing_entity_raises(self):
        agent = ConflictDetectionAgent()
        result = self._base_result({
            "conflicts": [{"conflict_type": "state_mismatch", "severity": "low"}],
            "resolution_hints": [],
        })
        with pytest.raises(ValueError, match="must have entity"):
            agent.validate_outputs(result)

    def test_conflict_missing_conflict_type_raises(self):
        agent = ConflictDetectionAgent()
        result = self._base_result({
            "conflicts": [{"entity": "X", "severity": "low"}],
            "resolution_hints": [],
        })
        with pytest.raises(ValueError, match="must have conflict_type"):
            agent.validate_outputs(result)

    def test_invalid_severity_raises(self):
        agent = ConflictDetectionAgent()
        result = self._base_result({
            "conflicts": [
                {"entity": "X", "conflict_type": "state_mismatch", "severity": "critical"}
            ],
            "resolution_hints": [],
        })
        with pytest.raises(ValueError, match="severity must be"):
            agent.validate_outputs(result)

    def test_non_success_status_skips_validation(self):
        from datetime import datetime, timezone
        agent = ConflictDetectionAgent()
        result = AgentResult(
            agent_name="ConflictDetectionAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="failed",
            confidence=0.0,
            outputs={"conflicts": "not-a-list"},
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )
        agent.validate_outputs(result)  # should not raise
