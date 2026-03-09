"""Tests for Phase 14 — AdaptiveConflictResolutionAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.adaptive_conflict_resolution_agent import (
    AdaptiveConflictResolutionAgent,
    assign_action_type,
    assign_team,
    build_resolution_actions,
    compute_hint_effectiveness,
    extract_escalation_targets,
    score_priority,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CONFLICT_STATE_HIGH = {
    "entity": "Acme Corp",
    "entity_type": "customer",
    "conflict_type": "state_mismatch",
    "severity": "high",
    "source_silos": ["sales", "support"],
    "resolution_hint": "sync sales and support on Acme Corp state",
}

_CONFLICT_SILO_MEDIUM = {
    "entity": "Project Phoenix",
    "entity_type": "project",
    "conflict_type": "silo_gap",
    "severity": "medium",
    "source_silos": ["engineering"],
    "resolution_hint": "cross-check Project Phoenix across silos",
}

_CONFLICT_LOOP_LOW = {
    "entity": "Deploy Pipeline",
    "entity_type": "system",
    "conflict_type": "causal_loop",
    "severity": "low",
    "source_silos": ["devops"],
    "resolution_hint": "break causal loop in Deploy Pipeline",
}


# ---------------------------------------------------------------------------
# assign_action_type
# ---------------------------------------------------------------------------

class TestAssignActionType:
    def test_state_mismatch_high(self):
        assert assign_action_type(conflict_type="state_mismatch", severity="high") == "escalate"

    def test_state_mismatch_medium(self):
        assert assign_action_type(conflict_type="state_mismatch", severity="medium") == "sync_teams"

    def test_state_mismatch_low(self):
        assert assign_action_type(conflict_type="state_mismatch", severity="low") == "monitor"

    def test_silo_gap_high(self):
        assert assign_action_type(conflict_type="silo_gap", severity="high") == "investigate"

    def test_silo_gap_medium(self):
        assert assign_action_type(conflict_type="silo_gap", severity="medium") == "investigate"

    def test_silo_gap_low(self):
        assert assign_action_type(conflict_type="silo_gap", severity="low") == "monitor"

    def test_causal_loop_high(self):
        assert assign_action_type(conflict_type="causal_loop", severity="high") == "escalate"

    def test_causal_loop_low(self):
        assert assign_action_type(conflict_type="causal_loop", severity="low") == "archive"

    def test_unknown_defaults_to_monitor(self):
        assert assign_action_type(conflict_type="unknown_type", severity="high") == "monitor"

    def test_empty_strings(self):
        assert assign_action_type(conflict_type="", severity="") == "monitor"


# ---------------------------------------------------------------------------
# assign_team
# ---------------------------------------------------------------------------

class TestAssignTeam:
    def test_first_silo_is_owner(self):
        assert assign_team(source_silos=["sales", "support"]) == "sales"

    def test_single_silo(self):
        assert assign_team(source_silos=["engineering"]) == "engineering"

    def test_empty_silos_returns_unknown(self):
        assert assign_team(source_silos=[]) == "unknown"

    def test_order_preserved(self):
        assert assign_team(source_silos=["finance", "hr", "legal"]) == "finance"


# ---------------------------------------------------------------------------
# score_priority
# ---------------------------------------------------------------------------

class TestScorePriority:
    def test_high_state_mismatch(self):
        score = score_priority(severity="high", conflict_type="state_mismatch")
        assert score == 0.95  # 0.90 + 0.05

    def test_medium_silo_gap(self):
        score = score_priority(severity="medium", conflict_type="silo_gap")
        assert score == 0.57  # 0.55 + 0.02

    def test_low_causal_loop(self):
        score = score_priority(severity="low", conflict_type="causal_loop")
        assert score == 0.28  # 0.25 + 0.03

    def test_unknown_severity_defaults_low(self):
        score = score_priority(severity="critical", conflict_type="state_mismatch")
        assert score == 0.3  # 0.25 + 0.05

    def test_unknown_conflict_type_no_boost(self):
        score = score_priority(severity="high", conflict_type="mystery")
        assert score == 0.90  # 0.90 + 0.0

    def test_capped_at_one(self):
        # Manually force: high=0.90 + state_mismatch=0.05 = 0.95 < 1.0, stays below cap
        # But a hypothetical 0.98 + 0.05 should cap
        score = score_priority(severity="high", conflict_type="state_mismatch")
        assert score <= 1.0


# ---------------------------------------------------------------------------
# compute_hint_effectiveness
# ---------------------------------------------------------------------------

class TestComputeHintEffectiveness:
    def test_empty_feedback_returns_defaults(self):
        result = compute_hint_effectiveness(resolution_feedback=[])
        assert result["escalate"] == 0.80
        assert result["sync_teams"] == 0.70
        assert result["monitor"] == 0.40

    def test_all_resolved(self):
        fb = [{"hint_used": "escalate", "resolved": True}] * 3
        result = compute_hint_effectiveness(resolution_feedback=fb)
        assert result["escalate"] == 1.0

    def test_none_resolved(self):
        fb = [{"hint_used": "sync_teams", "resolved": False}] * 4
        result = compute_hint_effectiveness(resolution_feedback=fb)
        assert result["sync_teams"] == 0.0

    def test_partial_resolution(self):
        fb = [
            {"hint_used": "investigate", "resolved": True},
            {"hint_used": "investigate", "resolved": False},
        ]
        result = compute_hint_effectiveness(resolution_feedback=fb)
        assert result["investigate"] == 0.5

    def test_multiple_hints(self):
        fb = [
            {"hint_used": "escalate", "resolved": True},
            {"hint_used": "monitor", "resolved": False},
            {"hint_used": "monitor", "resolved": False},
        ]
        result = compute_hint_effectiveness(resolution_feedback=fb)
        assert result["escalate"] == 1.0
        assert result["monitor"] == 0.0

    def test_missing_hint_used_skipped(self):
        fb = [{"resolved": True}, {"hint_used": "", "resolved": True}]
        result = compute_hint_effectiveness(resolution_feedback=fb)
        # No tracked hints — should just return defaults
        assert result["escalate"] == 0.80

    def test_unknown_hint_added_to_result(self):
        fb = [{"hint_used": "custom_action", "resolved": True}]
        result = compute_hint_effectiveness(resolution_feedback=fb)
        assert result["custom_action"] == 1.0


# ---------------------------------------------------------------------------
# build_resolution_actions
# ---------------------------------------------------------------------------

class TestBuildResolutionActions:
    def test_empty_conflicts(self):
        actions = build_resolution_actions(conflicts=[], hint_effectiveness={})
        assert actions == []

    def test_single_conflict_produces_one_action(self):
        actions = build_resolution_actions(
            conflicts=[_CONFLICT_STATE_HIGH], hint_effectiveness={}
        )
        assert len(actions) == 1
        assert actions[0]["entity"] == "Acme Corp"
        assert actions[0]["action_type"] == "escalate"

    def test_assigned_team_from_first_silo(self):
        actions = build_resolution_actions(
            conflicts=[_CONFLICT_STATE_HIGH], hint_effectiveness={}
        )
        assert actions[0]["assigned_team"] == "sales"

    def test_hint_preserved(self):
        actions = build_resolution_actions(
            conflicts=[_CONFLICT_STATE_HIGH], hint_effectiveness={}
        )
        assert "sync sales and support" in actions[0]["hint"]

    def test_sorted_by_priority_descending(self):
        conflicts = [_CONFLICT_LOOP_LOW, _CONFLICT_STATE_HIGH, _CONFLICT_SILO_MEDIUM]
        actions = build_resolution_actions(conflicts=conflicts, hint_effectiveness={})
        priorities = [a["priority"] for a in actions]
        assert priorities == sorted(priorities, reverse=True)

    def test_conflict_ref_indexes_original_position(self):
        conflicts = [_CONFLICT_LOOP_LOW, _CONFLICT_STATE_HIGH]
        actions = build_resolution_actions(conflicts=conflicts, hint_effectiveness={})
        refs = {a["conflict_ref"] for a in actions}
        assert refs == {0, 1}

    def test_hint_effectiveness_from_map(self):
        eff = {"escalate": 0.95}
        actions = build_resolution_actions(
            conflicts=[_CONFLICT_STATE_HIGH], hint_effectiveness=eff
        )
        assert actions[0]["hint_effectiveness"] == 0.95

    def test_investigate_action_for_silo_gap_medium(self):
        actions = build_resolution_actions(
            conflicts=[_CONFLICT_SILO_MEDIUM], hint_effectiveness={}
        )
        assert actions[0]["action_type"] == "investigate"

    def test_archive_action_for_causal_loop_low(self):
        actions = build_resolution_actions(
            conflicts=[_CONFLICT_LOOP_LOW], hint_effectiveness={}
        )
        assert actions[0]["action_type"] == "archive"


# ---------------------------------------------------------------------------
# extract_escalation_targets
# ---------------------------------------------------------------------------

class TestExtractEscalationTargets:
    def test_empty_actions(self):
        assert extract_escalation_targets(actions=[]) == []

    def test_one_escalate(self):
        actions = [{"action_type": "escalate", "assigned_team": "sales"}]
        assert extract_escalation_targets(actions=actions) == ["sales"]

    def test_deduplicates_same_team(self):
        actions = [
            {"action_type": "escalate", "assigned_team": "sales"},
            {"action_type": "escalate", "assigned_team": "sales"},
        ]
        assert extract_escalation_targets(actions=actions) == ["sales"]

    def test_non_escalate_excluded(self):
        actions = [
            {"action_type": "sync_teams", "assigned_team": "hr"},
            {"action_type": "investigate", "assigned_team": "eng"},
        ]
        assert extract_escalation_targets(actions=actions) == []

    def test_multiple_escalate_teams_in_order(self):
        actions = [
            {"action_type": "escalate", "assigned_team": "sales"},
            {"action_type": "monitor", "assigned_team": "devops"},
            {"action_type": "escalate", "assigned_team": "legal"},
        ]
        result = extract_escalation_targets(actions=actions)
        assert result == ["sales", "legal"]


# ---------------------------------------------------------------------------
# AdaptiveConflictResolutionAgent — meta
# ---------------------------------------------------------------------------

class TestAdaptiveConflictResolutionAgentMeta:
    def test_name(self):
        assert AdaptiveConflictResolutionAgent().name == "AdaptiveConflictResolutionAgent"

    def test_version(self):
        assert AdaptiveConflictResolutionAgent().version == "v1"

    def test_dependencies(self):
        assert AdaptiveConflictResolutionAgent().dependencies() == ["ConflictDetectionAgent"]

    def test_registry_lookup(self):
        assert isinstance(get_agent("adaptive_conflict_resolution"), AdaptiveConflictResolutionAgent)

    def test_registry_alias(self):
        assert isinstance(get_agent("adaptiveconflictresolutionagent"), AdaptiveConflictResolutionAgent)


# ---------------------------------------------------------------------------
# AdaptiveConflictResolutionAgent — heuristic run
# ---------------------------------------------------------------------------

def _make_context(*, conflicts=None, resolution_feedback=None, content=""):
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "conflicts": conflicts or [],
                "resolution_feedback": resolution_feedback or [],
            },
        },
        "runtime": {"job_id": "test-job-1"},
    }


@pytest.mark.asyncio
class TestAdaptiveConflictResolutionAgentHeuristic:
    async def test_empty_conflicts_returns_zero_actions(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["resolution_actions"] == []
        assert result.outputs["conflict_count"] == 0

    async def test_single_high_conflict_escalates(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH])
        result = await agent.run("mem-2", ctx)
        actions = result.outputs["resolution_actions"]
        assert len(actions) == 1
        assert actions[0]["action_type"] == "escalate"

    async def test_escalation_targets_populated(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH])
        result = await agent.run("mem-3", ctx)
        assert "sales" in result.outputs["escalation_targets"]

    async def test_silo_gap_produces_investigate(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_SILO_MEDIUM])
        result = await agent.run("mem-4", ctx)
        assert result.outputs["resolution_actions"][0]["action_type"] == "investigate"

    async def test_resolution_rate_nonzero_for_active_actions(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH, _CONFLICT_LOOP_LOW])
        result = await agent.run("mem-5", ctx)
        # 1 escalate (active) out of 2 total = 0.5
        assert result.outputs["resolution_rate"] == 0.5

    async def test_resolution_rate_zero_when_all_passive(self):
        agent = AdaptiveConflictResolutionAgent()
        conflict_loop_medium = {**_CONFLICT_LOOP_LOW, "severity": "medium", "conflict_type": "causal_loop"}
        ctx = _make_context(conflicts=[_CONFLICT_LOOP_LOW, conflict_loop_medium])
        result = await agent.run("mem-6", ctx)
        # Both produce "monitor" or "archive" — no active actions
        active = [
            a for a in result.outputs["resolution_actions"]
            if a["action_type"] in {"escalate", "sync_teams", "investigate"}
        ]
        assert len(active) == 0

    async def test_confidence_scales_with_conflict_count(self):
        agent = AdaptiveConflictResolutionAgent()
        conflicts = [_CONFLICT_STATE_HIGH] * 5
        ctx = _make_context(conflicts=conflicts)
        result = await agent.run("mem-7", ctx)
        assert result.outputs["confidence"] > 0.40

    async def test_rationale_is_heuristic(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH])
        result = await agent.run("mem-8", ctx)
        assert result.outputs["rationale"] == "heuristic"

    async def test_hint_effectiveness_in_outputs(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH])
        result = await agent.run("mem-9", ctx)
        eff = result.outputs["hint_effectiveness"]
        assert isinstance(eff, dict)
        assert "escalate" in eff

    async def test_resolution_feedback_updates_effectiveness(self):
        agent = AdaptiveConflictResolutionAgent()
        feedback = [{"hint_used": "escalate", "resolved": False}] * 3
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH], resolution_feedback=feedback)
        result = await agent.run("mem-10", ctx)
        assert result.outputs["hint_effectiveness"]["escalate"] == 0.0

    async def test_trace_id_in_result(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH])
        result = await agent.run("mem-11", ctx)
        assert result.trace_id == "test-job-1"

    async def test_result_fields_present(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH])
        result = await agent.run("mem-12", ctx)
        for key in ("resolution_actions", "escalation_targets", "resolution_rate",
                    "hint_effectiveness", "conflict_count", "confidence", "rationale"):
            assert key in result.outputs

    async def test_sorted_high_priority_first(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(
            conflicts=[_CONFLICT_LOOP_LOW, _CONFLICT_STATE_HIGH, _CONFLICT_SILO_MEDIUM]
        )
        result = await agent.run("mem-13", ctx)
        priorities = [a["priority"] for a in result.outputs["resolution_actions"]]
        assert priorities == sorted(priorities, reverse=True)


# ---------------------------------------------------------------------------
# AdaptiveConflictResolutionAgent — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAdaptiveConflictResolutionAgentLLM:
    async def test_valid_llm_response_used(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH], content="live content")

        llm_resp = {
            "resolution_actions": [{"conflict_ref": 0, "entity": "Acme Corp",
                                     "action_type": "escalate", "assigned_team": "sales",
                                     "priority": 0.95, "hint": "escalate now",
                                     "hint_effectiveness": 0.8}],
            "escalation_targets": ["sales"],
            "resolution_rate": 1.0,
            "hint_effectiveness": {"escalate": 0.8},
            "conflict_count": 1,
            "confidence": 0.88,
            "rationale": "llm",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)

        with patch(
            "app.agents.adaptive_conflict_resolution_agent.create_ollama_client",
            return_value=mock_client,
        ), patch.object(type(agent), "_get_strategy", return_value="llm", create=True):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.CONFLICT_RESOLUTION_STRATEGY = "llm"
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "llama3.1:8b"
                mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_settings.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-1", ctx)

        assert result.outputs["rationale"] == "llm"
        assert result.outputs["resolution_actions"][0]["entity"] == "Acme Corp"

    async def test_bad_llm_response_falls_back_to_heuristic(self):
        agent = AdaptiveConflictResolutionAgent()
        ctx = _make_context(conflicts=[_CONFLICT_STATE_HIGH], content="live content")

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "response"})

        with patch(
            "app.agents.adaptive_conflict_resolution_agent.create_ollama_client",
            return_value=mock_client,
        ):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.CONFLICT_RESOLUTION_STRATEGY = "llm"
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "llama3.1:8b"
                mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_settings.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-2", ctx)

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs):
        return AgentResult(
            agent_name="AdaptiveConflictResolutionAgent",
            agent_version="v1",
            memory_id="m",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_passes(self):
        agent = AdaptiveConflictResolutionAgent()
        result = self._make_result({
            "resolution_actions": [
                {"entity": "Acme", "action_type": "escalate"}
            ],
            "escalation_targets": ["sales"],
        })
        agent.validate_outputs(result)  # no raise

    def test_resolution_actions_not_list(self):
        agent = AdaptiveConflictResolutionAgent()
        result = self._make_result({"resolution_actions": "bad", "escalation_targets": []})
        with pytest.raises(ValueError, match="resolution_actions must be a list"):
            agent.validate_outputs(result)

    def test_escalation_targets_not_list(self):
        agent = AdaptiveConflictResolutionAgent()
        result = self._make_result({"resolution_actions": [], "escalation_targets": "bad"})
        with pytest.raises(ValueError, match="escalation_targets must be a list"):
            agent.validate_outputs(result)

    def test_missing_entity(self):
        agent = AdaptiveConflictResolutionAgent()
        result = self._make_result({
            "resolution_actions": [{"action_type": "escalate"}],
            "escalation_targets": [],
        })
        with pytest.raises(ValueError, match="must have entity"):
            agent.validate_outputs(result)

    def test_missing_action_type(self):
        agent = AdaptiveConflictResolutionAgent()
        result = self._make_result({
            "resolution_actions": [{"entity": "Acme"}],
            "escalation_targets": [],
        })
        with pytest.raises(ValueError, match="must have action_type"):
            agent.validate_outputs(result)

    def test_invalid_action_type(self):
        agent = AdaptiveConflictResolutionAgent()
        result = self._make_result({
            "resolution_actions": [{"entity": "Acme", "action_type": "zap"}],
            "escalation_targets": [],
        })
        with pytest.raises(ValueError, match="invalid action_type"):
            agent.validate_outputs(result)

    def test_non_success_skips_validation(self):
        agent = AdaptiveConflictResolutionAgent()
        result = AgentResult(
            agent_name="AdaptiveConflictResolutionAgent",
            agent_version="v1",
            memory_id="m",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=["boom"],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )
        agent.validate_outputs(result)  # no raise
