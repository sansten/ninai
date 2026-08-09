"""Tests for the expanded OrchestrationBusAgent agent roster.

35 agent classes were registered and fully unit-tested (in their own
test_*.py files) but had zero production invocation path: not in
WRITE_TIME_AGENT_SPECS, not in OrchestrationBusAgent's _DEFAULT_AGENTS, and
not referenced by any Celery task or endpoint that actually calls .run() on
them (some endpoints only *read* AgentRun rows filtered by these names,
assuming something else produced them — nothing did).

_DEFAULT_AGENTS / _SELECTABLE_AGENTS now include all of them, and
_heuristic_agent_names has new content-keyword branches so 33 of the 34 are
reachable via the default (non-LLM) write-time path without unconditionally
running all ~58 agents on every single memory write. MemoryTierManagerAgent
is intentionally selectable-only (session working-set capacity management,
not a per-memory content signal) — not asserted as heuristically triggered.
"""
from __future__ import annotations

import pytest

from app.agents.orchestration_bus_agent import (
    _DEFAULT_AGENTS,
    _SELECTABLE_AGENTS,
    _heuristic_agent_names,
)
from app.agents.registry import get_agent

_PREVIOUSLY_UNREACHABLE = [
    "ActiveKnowledgeSeekerAgent",
    "AdaptiveEnrichmentBudgetAgent",
    "AdaptivePersonaAgent",
    "AnalogicalReasoningAgent",
    "AuditTrailAgent",
    "AutoResearchAgent",
    "AutonomousActionAgent",
    "AutonomousGoalGenerationAgent",
    "CompositionalGeneralizationAgent",
    "ConceptLearningAgent",
    "CounterfactualMemoryAgent",
    "CrossModalReasoningAgent",
    "DebateEnsembleAgent",
    "EmotionalAffectiveMemoryAgent",
    "EpisodicFutureSimulationAgent",
    "ErrorRecoveryAgent",
    "ErrorRemediationAgent",
    "FederatedMemoryAgent",
    "HierarchicalGoalPlannerAgent",
    "HumanReviewQueueAgent",
    "MemoryTierManagerAgent",
    "MetaCognitivePlanningAgent",
    "MultiTurnGoalTrackingAgent",
    "MultimodalDeepMemoryAgent",
    "NarrativeCompressionAgent",
    "PlaybookAutoSynthesisAgent",
    "PlaybookExecutionTrackerAgent",
    "ProspectiveMemoryAgent",
    "QueryIntelligenceAgent",
    "SelfImprovementPlannerAgent",
    "SemanticChangeDetectionAgent",
    "SemanticRoleInferenceAgent",
    "SocialMemoryAgent",
    "TemporalPatternMinerAgent",
    "TheoryOfMindAgent",
]

# One representative content sample per new heuristic branch, mapped to the
# agent(s) it must select.
_TRIGGER_SAMPLES: dict[str, list[str]] = {
    "relationship trust rapport, worked with Bob": ["SocialMemoryAgent", "TheoryOfMindAgent"],
    "I am frustrated and worried about this": ["EmotionalAffectiveMemoryAgent"],
    "please research and investigate this further": ["AutoResearchAgent", "ActiveKnowledgeSeekerAgent"],
    "this reminds me of the same pattern as before": ["AnalogicalReasoningAgent", "CompositionalGeneralizationAgent"],
    "see the attached screenshot and diagram": ["MultimodalDeepMemoryAgent", "CrossModalReasoningAgent"],
    "needs review and audit before we approve": ["HumanReviewQueueAgent", "AuditTrailAgent"],
    "there was an error, the system crashed with an outage": ["ErrorRemediationAgent", "ErrorRecoveryAgent", "AutonomousActionAgent"],
    "should we do this? pros and cons of the tradeoff": ["DebateEnsembleAgent", "CounterfactualMemoryAgent"],
    "please summarize and recap the tl;dr": ["NarrativeCompressionAgent"],
    "remind me to follow up next time": ["ProspectiveMemoryAgent", "EpisodicFutureSimulationAgent"],
    "search for and find all memories about this": ["QueryIntelligenceAgent"],
    "we need a runbook procedure with steps to automate": ["PlaybookAutoSynthesisAgent", "PlaybookExecutionTrackerAgent"],
    "this is a recurring trend, pattern of drift over time": ["TemporalPatternMinerAgent", "SemanticChangeDetectionAgent"],
    "industry benchmark best practice from peers": ["FederatedMemoryAgent"],
    "I prefer this communication style and tone": ["AdaptivePersonaAgent"],
    "retrospective lessons learned, mistake to reflect on": ["MetaCognitivePlanningAgent", "SelfImprovementPlannerAgent", "AdaptiveEnrichmentBudgetAgent"],
    "new goal, task objective and milestone plan": ["MultiTurnGoalTrackingAgent", "HierarchicalGoalPlannerAgent", "AutonomousGoalGenerationAgent"],
    "new concept, define the terminology": ["ConceptLearningAgent"],
    "who did this, responsible for the action item": ["SemanticRoleInferenceAgent"],
}


class TestExpandedAgentRosterRegistered:
    @pytest.mark.parametrize("name", _PREVIOUSLY_UNREACHABLE)
    def test_in_default_and_selectable(self, name):
        assert name in _DEFAULT_AGENTS
        assert name in _SELECTABLE_AGENTS

    @pytest.mark.parametrize("name", _PREVIOUSLY_UNREACHABLE)
    def test_resolves_via_get_agent(self, name):
        agent = get_agent(name.lower())
        assert agent is not None, f"{name} did not resolve via get_agent()"
        assert agent.name == name

    def test_no_duplicate_entries_in_default_agents(self):
        assert len(_DEFAULT_AGENTS) == len(set(_DEFAULT_AGENTS))

    @pytest.mark.parametrize("name", _DEFAULT_AGENTS)
    def test_agent_name_and_dependencies_are_pascal_case(self, name):
        """Regression: AutonomousActionAgent's .name was 'autonomous_action_agent'
        (snake_case) and its dependencies() returned snake_case strings too.
        The bus's topo-sort keys the agents dict by agent.name and matches
        dependencies() entries against those keys by exact string — a casing
        mismatch means the dependency lookup silently misses (treated as
        "satisfied externally"), breaking execution ordering with no error.
        Every agent's name and declared dependencies must be PascalCase to
        match how every other agent in the roster identifies itself."""
        agent = get_agent(name.lower())
        assert agent is not None
        assert agent.name == name
        assert agent.name[0].isupper(), f"{name}: agent.name is not PascalCase"
        for dep in agent.dependencies():
            assert dep[0].isupper(), f"{name}: dependency {dep!r} is not PascalCase"


class TestHeuristicTriggersReachNewAgents:
    @pytest.mark.parametrize("content,expected_agents", list(_TRIGGER_SAMPLES.items()))
    def test_content_triggers_expected_agents(self, content, expected_agents):
        selected = _heuristic_agent_names(content, {})
        for agent_name in expected_agents:
            assert agent_name in selected, (
                f"content {content!r} should select {agent_name!r}, got {selected!r}"
            )

    def test_unrelated_content_does_not_trigger_everything(self):
        """A plain memory with no special signals should stay to the always-on
        base pipeline + synthesis tail, not the full 58-agent roster —
        otherwise every write pays for all 58 agent runs."""
        selected = _heuristic_agent_names("the weather was nice today", {})
        assert "SocialMemoryAgent" not in selected
        assert "ErrorRemediationAgent" not in selected
        assert "PlaybookAutoSynthesisAgent" not in selected
        assert len(selected) < 15

    def test_memory_tier_manager_is_selectable_only_not_heuristically_triggered(self):
        """Deliberately excluded from all heuristic branches — it manages
        session working-set capacity, not a per-memory content signal."""
        all_selected: set[str] = set()
        for content in list(_TRIGGER_SAMPLES.keys()) + [
            "conflict dispute", "new project kickoff", "deadline schedule",
            "team department silo",
        ]:
            all_selected.update(_heuristic_agent_names(content, {}))
        assert "MemoryTierManagerAgent" not in all_selected
        assert "MemoryTierManagerAgent" in _SELECTABLE_AGENTS
