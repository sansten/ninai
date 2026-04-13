from __future__ import annotations

from typing import Any

import pytest

from app.agents.auto_research_agent import AutoResearchAgent, _candidate_values
from app.agents.registry import get_agent
from app.services.auto_research_benchmark_harness import BenchmarkSummary
from app.services.config_snapshot_service import AUTO_RESEARCH_PARAMETER_REGISTRY, ParameterSpec


class FakeConfigService:
    def __init__(self, initial: dict[str, float]):
        self.values = dict(initial)
        self.restore_calls: list[dict[str, float]] = []

    async def snapshot(self, org_id: str, param_keys: list[str] | None = None) -> dict[str, float]:
        keys = param_keys or list(self.values)
        return {key: self.values[key] for key in keys}

    async def set_parameter_value(self, org_id: str, key: str, value: float) -> float:
        self.values[key] = value
        return value

    async def restore_snapshot(self, org_id: str, values: dict[str, float]) -> None:
        self.restore_calls.append(dict(values))
        self.values.update(values)


class FakeBenchmarkHarness:
    def __init__(self, scores: list[float]):
        self.scores = list(scores)
        self.calls: list[str] = []

    async def evaluate(self, org_id: str, *, label: str = "candidate") -> BenchmarkSummary:
        self.calls.append(label)
        score = self.scores.pop(0)
        return BenchmarkSummary(
            composite_score=score,
            gate_scores={"decide": score, "plan": score, "explain": score},
            gates_completed=3,
            duration_seconds=0.01,
            budget_exceeded=False,
            results=[{"name": "decide", "score": score, "details": {}}],
        )


class FakeSession:
    def __init__(self):
        self.added: list[Any] = []
        self.flush_calls = 0

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        self.flush_calls += 1


def _ctx(**enrichment: Any) -> dict[str, Any]:
    return {
        "tenant": {"org_id": "org-1", "org_slug": "acme"},
        "memory": {"enrichment": enrichment},
        "runtime": {"job_id": "trace-auto-research"},
    }


class TestCandidateValues:
    def test_moves_up_first_below_midpoint(self):
        spec = ParameterSpec("x", "meta_learning", "confidence_floor", 0.4, 0.1, 0.9, 0.05)
        assert _candidate_values(0.2, spec) == [0.25, 0.15]

    def test_moves_down_first_above_midpoint(self):
        spec = ParameterSpec("x", "meta_learning", "confidence_floor", 0.4, 0.1, 0.9, 0.05)
        assert _candidate_values(0.8, spec) == [0.75, 0.85]


class TestAutoResearchAgent:
    @pytest.mark.asyncio
    async def test_accepts_improving_candidate(self):
        key = "meta_learning.confidence_floor"
        config_service = FakeConfigService({key: 0.4})
        harness = FakeBenchmarkHarness([0.60, 0.67])
        session = FakeSession()
        agent = AutoResearchAgent(session=session, config_service=config_service, benchmark_harness=harness)

        result = await agent.run("m1", _ctx(parameter_key=key, min_improvement=0.02))

        assert result.status == "success"
        assert result.outputs["accepted"] is True
        assert result.outputs["applied_value"] == 0.45
        assert result.outputs["score_delta"] == 0.07
        assert config_service.values[key] == 0.45
        assert len(session.added) == 1
        assert session.added[0].status == "accepted"

    @pytest.mark.asyncio
    async def test_reverts_when_candidate_does_not_improve(self):
        key = "meta_learning.confidence_floor"
        config_service = FakeConfigService({key: 0.4})
        harness = FakeBenchmarkHarness([0.60, 0.605, 0.59])
        session = FakeSession()
        agent = AutoResearchAgent(session=session, config_service=config_service, benchmark_harness=harness)

        result = await agent.run("m1", _ctx(parameter_key=key, min_improvement=0.02))

        assert result.status == "success"
        assert result.outputs["accepted"] is False
        assert result.outputs["applied_value"] == 0.4
        assert result.outputs["final_score"] == 0.6
        assert len(result.outputs["experiments"]) == 2
        assert config_service.values[key] == 0.4
        assert len(config_service.restore_calls) >= 2
        assert [item.status for item in session.added] == ["reverted", "reverted"]

    @pytest.mark.asyncio
    async def test_requires_org_id(self):
        key = "meta_learning.confidence_floor"
        agent = AutoResearchAgent(
            config_service=FakeConfigService({key: 0.4}),
            benchmark_harness=FakeBenchmarkHarness([0.6, 0.7]),
        )

        result = await agent.run("m1", {"memory": {"enrichment": {"parameter_key": key}}})

        assert result.status == "failed"
        assert "org_id is required" in result.errors[0]

    @pytest.mark.asyncio
    async def test_records_trace_id_in_ledger_summary(self):
        key = "meta_learning.confidence_floor"
        session = FakeSession()
        agent = AutoResearchAgent(
            session=session,
            config_service=FakeConfigService({key: 0.4}),
            benchmark_harness=FakeBenchmarkHarness([0.60, 0.67]),
        )

        await agent.run("m1", _ctx(parameter_key=key, min_improvement=0.02))

        assert session.added[0].benchmark_summary["trace_id"] == "trace-auto-research"


class TestRegistryIntegration:
    def test_registry_returns_auto_research_agent(self):
        agent = get_agent("auto_research")
        assert agent is not None
        assert agent.name == "AutoResearchAgent"

    def test_registry_keys_are_realistic(self):
        assert "meta_learning.confidence_floor" in AUTO_RESEARCH_PARAMETER_REGISTRY
        assert "org_llm.temperature" in AUTO_RESEARCH_PARAMETER_REGISTRY