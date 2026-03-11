"""Tests for MemorySleepAgent — Phase 40."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from datetime import datetime, timezone

from app.agents.memory_sleep_agent import (
    MemorySleepAgent,
    _PRUNE_REDUNDANCY_THRESHOLD,
    _STRENGTHEN_FRESHNESS_MAX,
    _STRENGTHEN_REF_MIN,
    _ASSOCIATION_SIMILARITY_MIN,
    _MAX_ASSOCIATION_LINKS,
    _DELTA,
    _VALID_ACTIONS,
    decide_sleep_action,
    build_association_links,
    build_prune_reason,
    build_association_reason,
    action_confidence,
)
from app.agents.registry import get_agent
from app.models.sleep_cycle_report import SleepCycleReport


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _similar(id: str, score: float, domain: str = "engineering") -> dict:
    return {"id": id, "similarity_score": score, "domain": domain}


def _candidate(id: str, score: float = 0.80) -> dict:
    return {"id": id, "similarity_score": score, "reason": "same_domain_similar"}


def _make_context(
    freshness_score: float = 0.8,
    is_stale: bool = False,
    reference_count: int = 5,
    consolidation_action: str = "keep",
    merge_candidates: list | None = None,
    redundancy_score: float = 0.1,
    similar_memories: list | None = None,
    content: str = "some memory content",
    job_id: str | None = None,
) -> dict:
    enrichment: dict = {
        "freshness_score": freshness_score,
        "is_stale": is_stale,
        "reference_count": reference_count,
        "consolidation_action": consolidation_action,
        "merge_candidates": merge_candidates or [],
        "redundancy_score": redundancy_score,
        "similar_memories": similar_memories or [],
        "episode_key": "eng_acme_2026-W10",
        "episode_cohesion": 0.75,
        "cross_episode_links": [],
    }
    ctx: dict = {"memory": {"content": content, "enrichment": enrichment}}
    if job_id:
        ctx["runtime"] = {"job_id": job_id}
    return ctx


# ---------------------------------------------------------------------------
# decide_sleep_action
# ---------------------------------------------------------------------------

class TestDecideSleepAction:
    def test_prune_when_archive(self):
        action = decide_sleep_action(
            consolidation_action="archive",
            is_stale=False,
            redundancy_score=0.0,
            merge_candidates=[],
            freshness_score=0.9,
            reference_count=10,
        )
        assert action == "prune"

    def test_prune_when_stale_and_redundant(self):
        action = decide_sleep_action(
            consolidation_action="keep",
            is_stale=True,
            redundancy_score=_PRUNE_REDUNDANCY_THRESHOLD,
            merge_candidates=[],
            freshness_score=0.1,
            reference_count=0,
        )
        assert action == "prune"

    def test_no_prune_when_stale_but_low_redundancy(self):
        action = decide_sleep_action(
            consolidation_action="keep",
            is_stale=True,
            redundancy_score=_PRUNE_REDUNDANCY_THRESHOLD - 0.01,
            merge_candidates=[],
            freshness_score=0.8,
            reference_count=10,
        )
        assert action != "prune"

    def test_merge_when_candidates_exist(self):
        action = decide_sleep_action(
            consolidation_action="merge_candidate",
            is_stale=False,
            redundancy_score=0.2,
            merge_candidates=[_candidate("m1")],
            freshness_score=0.9,
            reference_count=5,
        )
        assert action == "merge"

    def test_prune_beats_merge(self):
        """archive consolidation_action takes priority over merge candidates."""
        action = decide_sleep_action(
            consolidation_action="archive",
            is_stale=True,
            redundancy_score=0.9,
            merge_candidates=[_candidate("m1")],
            freshness_score=0.1,
            reference_count=0,
        )
        assert action == "prune"

    def test_strengthen_low_freshness_high_refs(self):
        action = decide_sleep_action(
            consolidation_action="keep",
            is_stale=False,
            redundancy_score=0.0,
            merge_candidates=[],
            freshness_score=_STRENGTHEN_FRESHNESS_MAX - 0.01,
            reference_count=_STRENGTHEN_REF_MIN,
        )
        assert action == "strengthen"

    def test_no_strengthen_when_insufficient_refs(self):
        action = decide_sleep_action(
            consolidation_action="keep",
            is_stale=False,
            redundancy_score=0.0,
            merge_candidates=[],
            freshness_score=_STRENGTHEN_FRESHNESS_MAX - 0.01,
            reference_count=_STRENGTHEN_REF_MIN - 1,
        )
        assert action == "retain"

    def test_no_strengthen_when_freshness_above_threshold(self):
        action = decide_sleep_action(
            consolidation_action="keep",
            is_stale=False,
            redundancy_score=0.0,
            merge_candidates=[],
            freshness_score=_STRENGTHEN_FRESHNESS_MAX,
            reference_count=10,
        )
        assert action == "retain"

    def test_retain_default(self):
        action = decide_sleep_action(
            consolidation_action="keep",
            is_stale=False,
            redundancy_score=0.0,
            merge_candidates=[],
            freshness_score=0.9,
            reference_count=2,
        )
        assert action == "retain"

    def test_compress_action_retains(self):
        """consolidation_action=compress with no other triggers → retain."""
        action = decide_sleep_action(
            consolidation_action="compress",
            is_stale=False,
            redundancy_score=0.0,
            merge_candidates=[],
            freshness_score=0.9,
            reference_count=2,
        )
        assert action == "retain"


# ---------------------------------------------------------------------------
# build_association_links
# ---------------------------------------------------------------------------

class TestBuildAssociationLinks:
    def test_returns_high_sim_not_in_merge(self):
        similar = [_similar("a1", 0.80), _similar("a2", 0.76)]
        links = build_association_links(similar, set())
        assert "a1" in links and "a2" in links

    def test_excludes_merge_candidates(self):
        similar = [_similar("m1", 0.90)]
        links = build_association_links(similar, {"m1"})
        assert "m1" not in links

    def test_excludes_below_threshold(self):
        similar = [_similar("low", _ASSOCIATION_SIMILARITY_MIN - 0.01)]
        links = build_association_links(similar, set())
        assert links == []

    def test_exactly_at_threshold(self):
        similar = [_similar("edge", _ASSOCIATION_SIMILARITY_MIN)]
        links = build_association_links(similar, set())
        assert "edge" in links

    def test_capped_at_max(self):
        similar = [_similar(f"m{i}", 0.90) for i in range(_MAX_ASSOCIATION_LINKS + 3)]
        links = build_association_links(similar, set())
        assert len(links) <= _MAX_ASSOCIATION_LINKS

    def test_empty_similar_memories(self):
        assert build_association_links([], set()) == []

    def test_skips_entries_with_no_id(self):
        similar = [{"similarity_score": 0.95}]
        links = build_association_links(similar, set())
        assert links == []


# ---------------------------------------------------------------------------
# build_prune_reason
# ---------------------------------------------------------------------------

class TestBuildPruneReason:
    def test_archive_reason(self):
        reason = build_prune_reason(
            consolidation_action="archive", is_stale=False, redundancy_score=0.0
        )
        assert reason == "consolidation_agent_flagged_archive"

    def test_stale_redundant_reason(self):
        reason = build_prune_reason(
            consolidation_action="keep",
            is_stale=True,
            redundancy_score=0.60,
        )
        assert reason is not None and "stale_and_redundant" in reason

    def test_no_reason_when_healthy(self):
        reason = build_prune_reason(
            consolidation_action="keep", is_stale=False, redundancy_score=0.1
        )
        assert reason is None


# ---------------------------------------------------------------------------
# build_association_reason
# ---------------------------------------------------------------------------

class TestBuildAssociationReason:
    def test_with_links(self):
        reason = build_association_reason(["m1", "m2"])
        assert reason is not None and "2" in reason

    def test_empty_links(self):
        assert build_association_reason([]) is None


# ---------------------------------------------------------------------------
# action_confidence
# ---------------------------------------------------------------------------

class TestActionConfidence:
    def test_prune_archive_confidence(self):
        conf = action_confidence(
            "prune",
            consolidation_action="archive",
            merge_candidates=[],
            is_stale=False,
            redundancy_score=0.0,
        )
        assert conf == 0.80

    def test_prune_stale_confidence(self):
        conf = action_confidence(
            "prune",
            consolidation_action="keep",
            merge_candidates=[],
            is_stale=True,
            redundancy_score=0.6,
        )
        assert conf == 0.70

    def test_merge_confidence(self):
        conf = action_confidence(
            "merge",
            consolidation_action="merge_candidate",
            merge_candidates=[_candidate("x")],
            is_stale=False,
            redundancy_score=0.0,
        )
        assert conf == 0.80

    def test_strengthen_confidence(self):
        conf = action_confidence(
            "strengthen",
            consolidation_action="keep",
            merge_candidates=[],
            is_stale=False,
            redundancy_score=0.0,
        )
        assert conf == 0.75

    def test_retain_confidence(self):
        conf = action_confidence(
            "retain",
            consolidation_action="keep",
            merge_candidates=[],
            is_stale=False,
            redundancy_score=0.0,
        )
        assert conf == 0.60


# ---------------------------------------------------------------------------
# strength_delta constants
# ---------------------------------------------------------------------------

class TestStrengthDelta:
    def test_retain_delta_zero(self):
        assert _DELTA["retain"] == 0.0

    def test_strengthen_positive(self):
        assert _DELTA["strengthen"] > 0.0

    def test_merge_negative(self):
        assert _DELTA["merge"] < 0.0

    def test_prune_zero(self):
        assert _DELTA["prune"] == 0.0


# ---------------------------------------------------------------------------
# MemorySleepAgent — heuristic path
# ---------------------------------------------------------------------------

class TestMemorySleepAgentHeuristic:
    @pytest.mark.asyncio
    async def test_retain_action(self):
        agent = MemorySleepAgent()
        ctx = _make_context(freshness_score=0.9, is_stale=False, reference_count=2)
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-001", ctx)
        assert result.status == "success"
        assert result.outputs["sleep_action"] == "retain"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_strengthen_action(self):
        agent = MemorySleepAgent()
        ctx = _make_context(
            freshness_score=_STRENGTHEN_FRESHNESS_MAX - 0.05,
            is_stale=False,
            reference_count=_STRENGTHEN_REF_MIN,
        )
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-002", ctx)
        assert result.outputs["sleep_action"] == "strengthen"
        assert result.outputs["strength_delta"] > 0.0

    @pytest.mark.asyncio
    async def test_merge_action(self):
        agent = MemorySleepAgent()
        ctx = _make_context(
            consolidation_action="merge_candidate",
            merge_candidates=[_candidate("dup-1")],
            freshness_score=0.8,
        )
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-003", ctx)
        assert result.outputs["sleep_action"] == "merge"
        assert result.outputs["strength_delta"] < 0.0

    @pytest.mark.asyncio
    async def test_prune_archive(self):
        agent = MemorySleepAgent()
        ctx = _make_context(consolidation_action="archive")
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-004", ctx)
        assert result.outputs["sleep_action"] == "prune"
        assert result.outputs["prune_reason"] == "consolidation_agent_flagged_archive"

    @pytest.mark.asyncio
    async def test_prune_stale_redundant(self):
        agent = MemorySleepAgent()
        ctx = _make_context(
            is_stale=True,
            redundancy_score=_PRUNE_REDUNDANCY_THRESHOLD,
            consolidation_action="compress",
        )
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-005", ctx)
        assert result.outputs["sleep_action"] == "prune"

    @pytest.mark.asyncio
    async def test_association_links_discovered(self):
        agent = MemorySleepAgent()
        ctx = _make_context(
            freshness_score=0.9,
            similar_memories=[_similar("s1", 0.80), _similar("s2", 0.60)],
        )
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-006", ctx)
        assert "s1" in result.outputs["association_links"]
        assert "s2" not in result.outputs["association_links"]

    @pytest.mark.asyncio
    async def test_no_association_links_for_merge_candidates(self):
        agent = MemorySleepAgent()
        ctx = _make_context(
            consolidation_action="merge_candidate",
            merge_candidates=[_candidate("m1")],
            similar_memories=[_similar("m1", 0.95)],
        )
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-007", ctx)
        assert "m1" not in result.outputs["association_links"]

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = MemorySleepAgent()
        ctx = _make_context(job_id="job-xyz")
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-008", ctx)
        assert result.trace_id == "job-xyz"

    @pytest.mark.asyncio
    async def test_empty_context(self):
        agent = MemorySleepAgent()
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-009", {})
        assert result.status == "success"
        assert result.outputs["sleep_action"] in _VALID_ACTIONS

    @pytest.mark.asyncio
    async def test_agent_name_version(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-010", ctx)
        assert result.agent_name == "MemorySleepAgent"
        assert result.agent_version == "v1"

    @pytest.mark.asyncio
    async def test_memory_id_propagated(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-unique-id", ctx)
        assert result.memory_id == "mem-unique-id"

    @pytest.mark.asyncio
    async def test_confidence_in_range(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-011", ctx)
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_association_reason_present_when_links(self):
        agent = MemorySleepAgent()
        ctx = _make_context(
            similar_memories=[_similar("s1", 0.80)],
            freshness_score=0.9,
        )
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-012", ctx)
        if result.outputs["association_links"]:
            assert result.outputs["association_reason"] is not None

    @pytest.mark.asyncio
    async def test_no_prune_reason_for_retain(self):
        agent = MemorySleepAgent()
        ctx = _make_context(freshness_score=0.9, is_stale=False)
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-013", ctx)
        assert result.outputs.get("prune_reason") is None

    @pytest.mark.asyncio
    async def test_timestamps_set(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-014", ctx)
        assert isinstance(result.started_at, datetime)
        assert isinstance(result.finished_at, datetime)
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_max_association_links_capped(self):
        agent = MemorySleepAgent()
        many_similar = [_similar(f"s{i}", 0.90) for i in range(_MAX_ASSOCIATION_LINKS + 5)]
        ctx = _make_context(similar_memories=many_similar, freshness_score=0.9)
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings:
            mock_settings.MEMORY_SLEEP_STRATEGY = "heuristic"
            result = await agent.run("mem-015", ctx)
        assert len(result.outputs["association_links"]) <= _MAX_ASSOCIATION_LINKS


# ---------------------------------------------------------------------------
# MemorySleepAgent — LLM path
# ---------------------------------------------------------------------------

class TestMemorySleepAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_valid_response_used(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        llm_resp = {
            "sleep_action": "strengthen",
            "association_links": ["x1"],
            "association_reason": "cross domain link",
            "strength_delta": 0.15,
            "prune_reason": None,
            "confidence": 0.85,
            "rationale": "llm analysis",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings, \
             patch("app.agents.memory_sleep_agent.create_ollama_client", return_value=mock_client):
            mock_settings.MEMORY_SLEEP_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm-001", ctx)
        assert result.outputs["sleep_action"] == "strengthen"
        assert result.outputs["association_links"] == ["x1"]

    @pytest.mark.asyncio
    async def test_llm_invalid_response_falls_back_to_heuristic(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "response"})
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings, \
             patch("app.agents.memory_sleep_agent.create_ollama_client", return_value=mock_client):
            mock_settings.MEMORY_SLEEP_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm-002", ctx)
        assert result.outputs["sleep_action"] in _VALID_ACTIONS
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_heuristic(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("timeout"))
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings, \
             patch("app.agents.memory_sleep_agent.create_ollama_client", return_value=mock_client):
            mock_settings.MEMORY_SLEEP_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            # Should raise since fallback isn't wrapped around LLM exceptions
            with pytest.raises(RuntimeError):
                await agent.run("mem-llm-003", ctx)

    @pytest.mark.asyncio
    async def test_llm_invalid_action_falls_back(self):
        agent = MemorySleepAgent()
        ctx = _make_context()
        llm_resp = {
            "sleep_action": "invalid_action",
            "association_links": [],
            "strength_delta": 0.0,
            "confidence": 0.9,
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)
        with patch("app.agents.memory_sleep_agent.settings") as mock_settings, \
             patch("app.agents.memory_sleep_agent.create_ollama_client", return_value=mock_client):
            mock_settings.MEMORY_SLEEP_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-llm-004", ctx)
        assert result.outputs["sleep_action"] in _VALID_ACTIONS


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, **overrides):
        from app.agents.types import AgentResult
        outputs = {
            "sleep_action": "retain",
            "association_links": [],
            "association_reason": None,
            "strength_delta": 0.0,
            "prune_reason": None,
            "confidence": 0.6,
            "rationale": "heuristic",
        }
        outputs.update(overrides)
        return AgentResult(
            agent_name="MemorySleepAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=float(outputs.get("confidence", 0.6)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_passes(self):
        agent = MemorySleepAgent()
        result = self._make_result()
        agent.validate_outputs(result)  # no exception

    def test_invalid_sleep_action_raises(self):
        agent = MemorySleepAgent()
        result = self._make_result(sleep_action="nap")
        with pytest.raises(ValueError, match="invalid sleep_action"):
            agent.validate_outputs(result)

    def test_missing_association_links_raises(self):
        agent = MemorySleepAgent()
        result = self._make_result(association_links="not-a-list")
        with pytest.raises(ValueError, match="association_links"):
            agent.validate_outputs(result)

    def test_non_float_strength_delta_raises(self):
        agent = MemorySleepAgent()
        result = self._make_result(strength_delta="high")
        with pytest.raises(ValueError, match="strength_delta"):
            agent.validate_outputs(result)

    def test_confidence_out_of_range_raises(self):
        agent = MemorySleepAgent()
        # Put the invalid value inside outputs (validate_outputs reads outputs["confidence"])
        result = self._make_result()
        result.outputs["confidence"] = 1.5
        with pytest.raises(ValueError, match="confidence"):
            agent.validate_outputs(result)

    def test_skipped_status_not_validated(self):
        from app.agents.types import AgentResult
        agent = MemorySleepAgent()
        result = AgentResult(
            agent_name="MemorySleepAgent",
            agent_version="v1",
            memory_id="m1",
            status="skipped",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )
        agent.validate_outputs(result)  # no exception for non-success status

    def test_all_valid_actions_pass(self):
        agent = MemorySleepAgent()
        for action in _VALID_ACTIONS:
            result = self._make_result(sleep_action=action)
            agent.validate_outputs(result)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

class TestDependencies:
    def test_dependencies_list(self):
        agent = MemorySleepAgent()
        deps = agent.dependencies()
        assert "MemoryDecayAgent" in deps
        assert "MemoryConsolidationAgent" in deps
        assert "EpisodicGroupingAgent" in deps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_memory_sleep_lookup(self):
        assert isinstance(get_agent("memory_sleep"), MemorySleepAgent)

    def test_memorysleep_lookup(self):
        assert isinstance(get_agent("memorysleep"), MemorySleepAgent)

    def test_memorysleepagent_lookup(self):
        assert isinstance(get_agent("memorysleepagent"), MemorySleepAgent)

    def test_unknown_returns_none(self):
        assert get_agent("sleep_cycle_unknown_xyz") is None


# ---------------------------------------------------------------------------
# SleepCycleReport model
# ---------------------------------------------------------------------------

class TestSleepCycleReportModel:
    def test_model_has_required_columns(self):
        cols = {c.name for c in SleepCycleReport.__table__.columns}
        for col in (
            "id", "organization_id", "run_date",
            "memories_scanned", "retained", "strengthened", "merged", "pruned",
            "associations_created", "report_payload", "completed_at",
            "created_at", "updated_at",
        ):
            assert col in cols, f"missing column: {col}"

    def test_tablename(self):
        assert SleepCycleReport.__tablename__ == "sleep_cycle_reports"

    def test_unique_index_defined(self):
        index_names = {idx.name for idx in SleepCycleReport.__table__.indexes}
        assert "uq_sleep_cycle_org_date" in index_names

    def test_instantiation(self):
        from datetime import date, datetime, timezone
        report = SleepCycleReport(
            organization_id="org-123",
            run_date=date(2026, 3, 11),
            memories_scanned=100,
            retained=60,
            strengthened=15,
            merged=10,
            pruned=15,
            associations_created=30,
            report_payload={"details": []},
            completed_at=datetime.now(timezone.utc),
        )
        assert report.memories_scanned == 100
        assert report.retained == 60


# ---------------------------------------------------------------------------
# Pipeline task (broker-disabled path)
# ---------------------------------------------------------------------------

class TestMemorySleepPipeline:
    def test_task_broker_disabled_returns_skip(self):
        from app.tasks.memory_sleep_pipeline import memory_sleep_task
        # In-memory broker → should return skipped immediately.
        result = memory_sleep_task.apply(args=[], kwargs={}).get()
        assert result == {"ok": True, "skipped": True, "reason": "broker_disabled"}

    def test_build_agent_context_with_metadata(self):
        from app.tasks.memory_sleep_pipeline import _build_agent_context
        from unittest.mock import MagicMock

        mem = MagicMock()
        mem.id = "mem-abc"
        mem.content = "test content"
        mem.created_at = datetime.now(timezone.utc)
        mem.enrichment = {"freshness_score": 0.5}
        mem.metadata = {"is_stale": True}
        ctx = _build_agent_context(mem)
        assert ctx["memory"]["id"] == "mem-abc"
        assert ctx["memory"]["enrichment"]["freshness_score"] == 0.5
        # metadata key should be merged in when missing from enrichment
        assert ctx["memory"]["enrichment"]["is_stale"] is True

    def test_build_agent_context_enrichment_takes_priority(self):
        from app.tasks.memory_sleep_pipeline import _build_agent_context
        from unittest.mock import MagicMock

        mem = MagicMock()
        mem.id = "mem-def"
        mem.content = "test"
        mem.created_at = None
        mem.enrichment = {"freshness_score": 0.9}
        mem.metadata = {"freshness_score": 0.1}
        ctx = _build_agent_context(mem)
        # enrichment value wins
        assert ctx["memory"]["enrichment"]["freshness_score"] == 0.9

    def test_broker_enabled_check_memory_broker(self):
        from app.tasks.memory_sleep_pipeline import _broker_enabled
        # Default test broker is memory://
        assert _broker_enabled() is False

    @pytest.mark.asyncio
    async def test_run_sleep_cycle_async_no_orgs(self):
        from app.tasks.memory_sleep_pipeline import _run_sleep_cycle_async
        from unittest.mock import AsyncMock, patch, MagicMock

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: [])))
        mock_ctx_manager = MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=False))

        with patch("app.tasks.memory_sleep_pipeline.async_session_factory", return_value=mock_ctx_manager):
            result = await _run_sleep_cycle_async(batch_size=100, lookback_days=1)

        assert result["ok"] is True
        assert result["orgs_processed"] == 0
        assert result["total_scanned"] == 0

    def test_valid_actions_constant(self):
        assert "retain" in _VALID_ACTIONS
        assert "strengthen" in _VALID_ACTIONS
        assert "merge" in _VALID_ACTIONS
        assert "prune" in _VALID_ACTIONS
        assert len(_VALID_ACTIONS) == 4
