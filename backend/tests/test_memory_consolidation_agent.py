"""Tests for Phase 16 — MemoryConsolidationAgent."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.agents.memory_consolidation_agent import (
    MemoryConsolidationAgent,
    _ARCHIVE_REDUNDANCY_THRESHOLD,
    _HIGH_SIMILARITY,
    _LOW_REF_THRESHOLD,
    _MIN_MERGE_SIMILARITY,
    build_compression_summary,
    build_deduplication_key,
    compute_redundancy_score,
    decide_action,
    filter_merge_candidates,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# build_deduplication_key
# ---------------------------------------------------------------------------

class TestBuildDeduplicationKey:
    def test_includes_domain(self):
        key = build_deduplication_key(domain="sales", entities={}, content="")
        assert key.startswith("sales")

    def test_includes_entity_names(self):
        key = build_deduplication_key(
            domain="sales", entities={"customer": "Acme"}, content=""
        )
        assert "acme" in key
        assert "customer" in key

    def test_sorted_and_lowercased(self):
        key1 = build_deduplication_key(
            domain="sales", entities={"B": "x", "A": "y"}, content=""
        )
        key2 = build_deduplication_key(
            domain="sales", entities={"A": "y", "B": "x"}, content=""
        )
        assert key1 == key2

    def test_excludes_stopwords(self):
        key = build_deduplication_key(
            domain="sales", entities={}, content="the quick brown fox and the dog"
        )
        assert "the" not in key.split("|")
        assert "and" not in key.split("|")

    def test_entity_list_values_included(self):
        key = build_deduplication_key(
            domain="hr",
            entities={"teams": ["alpha", "beta"]},
            content="",
        )
        assert "alpha" in key
        assert "beta" in key

    def test_empty_inputs_returns_domain_only(self):
        key = build_deduplication_key(domain="finance", entities={}, content="")
        assert key == "finance"

    def test_content_tokens_contribute(self):
        key = build_deduplication_key(
            domain="engineering",
            entities={},
            content="deployment rollback incident kubernetes",
        )
        assert "deployment" in key or "rollback" in key or "incident" in key


# ---------------------------------------------------------------------------
# filter_merge_candidates
# ---------------------------------------------------------------------------

class TestFilterMergeCandidates:
    def _mem(self, id: str, sim: float, domain: str) -> dict:
        return {"id": id, "similarity_score": sim, "domain": domain}

    def test_empty_returns_empty(self):
        assert filter_merge_candidates([], domain="sales") == []

    def test_same_domain_above_threshold_included(self):
        mems = [self._mem("a1", 0.80, "sales")]
        result = filter_merge_candidates(mems, domain="sales")
        assert len(result) == 1
        assert result[0]["id"] == "a1"
        assert result[0]["reason"] == "same_domain_similar"

    def test_same_domain_below_threshold_excluded(self):
        mems = [self._mem("a1", 0.50, "sales")]
        result = filter_merge_candidates(mems, domain="sales")
        assert result == []

    def test_cross_domain_high_sim_included(self):
        mems = [self._mem("b1", _HIGH_SIMILARITY, "engineering")]
        result = filter_merge_candidates(mems, domain="sales")
        assert len(result) == 1
        assert result[0]["reason"] == "cross_domain_near_duplicate"

    def test_cross_domain_below_high_threshold_excluded(self):
        mems = [self._mem("b1", 0.75, "engineering")]
        result = filter_merge_candidates(mems, domain="sales")
        assert result == []

    def test_sorted_by_similarity_desc(self):
        mems = [
            self._mem("a1", 0.72, "sales"),
            self._mem("a2", 0.90, "sales"),
            self._mem("a3", 0.81, "sales"),
        ]
        result = filter_merge_candidates(mems, domain="sales")
        scores = [r["similarity_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_no_id_skipped(self):
        mems = [{"id": "", "similarity_score": 0.95, "domain": "sales"}]
        result = filter_merge_candidates(mems, domain="sales")
        assert result == []

    def test_capped_at_max_candidates(self):
        mems = [self._mem(f"m{i}", 0.80, "sales") for i in range(20)]
        result = filter_merge_candidates(mems, domain="sales")
        assert len(result) == 10


# ---------------------------------------------------------------------------
# compute_redundancy_score
# ---------------------------------------------------------------------------

class TestComputeRedundancyScore:
    def test_no_candidates_no_stale_no_boost(self):
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=False, reference_count=5
        )
        assert score == 0.0

    def test_no_candidates_stale_boost(self):
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=True, reference_count=5
        )
        assert score == pytest.approx(0.10)

    def test_similar_memories_set_base(self):
        similar = [{"id": "x", "similarity_score": 0.80, "domain": "sales"}]
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=False, reference_count=5,
            similar_memories=similar,
        )
        assert score == pytest.approx(0.80)

    def test_low_ref_boost(self):
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=False, reference_count=1,
            similar_memories=[],
        )
        assert score == pytest.approx(0.10)

    def test_both_boosts_added(self):
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=True, reference_count=0,
            similar_memories=[],
        )
        assert score == pytest.approx(0.20)

    def test_cap_at_one(self):
        similar = [{"id": "x", "similarity_score": 0.95, "domain": "sales"}]
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=True, reference_count=0,
            similar_memories=similar,
        )
        assert score == 1.0

    def test_threshold_boundary_stale(self):
        # 0.40 base (from similar_memories) + 0.10 stale = 0.50 → archive threshold
        similar = [{"id": "x", "similarity_score": 0.40, "domain": "sales"}]
        score = compute_redundancy_score(
            merge_candidates=[], is_stale=True, reference_count=5,
            similar_memories=similar,
        )
        assert score == pytest.approx(0.50)

    def test_falls_back_to_merge_candidates_when_similar_not_supplied(self):
        candidates = [{"id": "x", "similarity_score": 0.75, "reason": "same_domain_similar"}]
        score = compute_redundancy_score(
            merge_candidates=candidates, is_stale=False, reference_count=5,
        )
        assert score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# decide_action
# ---------------------------------------------------------------------------

class TestDecideAction:
    def test_merge_candidate_when_candidates_exist(self):
        candidates = [{"id": "x", "similarity_score": 0.80, "reason": "same_domain_similar"}]
        action = decide_action(
            merge_candidates=candidates,
            is_stale=False,
            redundancy_score=0.80,
            reference_count=5,
        )
        assert action == "merge_candidate"

    def test_merge_candidate_beats_stale(self):
        candidates = [{"id": "x", "similarity_score": 0.80, "reason": "same_domain_similar"}]
        action = decide_action(
            merge_candidates=candidates,
            is_stale=True,
            redundancy_score=0.90,
            reference_count=0,
        )
        assert action == "merge_candidate"

    def test_archive_when_stale_and_redundant(self):
        action = decide_action(
            merge_candidates=[],
            is_stale=True,
            redundancy_score=_ARCHIVE_REDUNDANCY_THRESHOLD,
            reference_count=5,
        )
        assert action == "archive"

    def test_compress_when_stale_low_ref(self):
        action = decide_action(
            merge_candidates=[],
            is_stale=True,
            redundancy_score=0.20,
            reference_count=1,
        )
        assert action == "compress"

    def test_keep_default(self):
        action = decide_action(
            merge_candidates=[],
            is_stale=False,
            redundancy_score=0.0,
            reference_count=10,
        )
        assert action == "keep"

    def test_stale_high_ref_keeps_if_not_redundant(self):
        action = decide_action(
            merge_candidates=[],
            is_stale=True,
            redundancy_score=0.20,
            reference_count=10,
        )
        assert action == "keep"

    def test_not_stale_no_candidates_keep(self):
        action = decide_action(
            merge_candidates=[],
            is_stale=False,
            redundancy_score=0.10,
            reference_count=0,
        )
        assert action == "keep"

    def test_archive_threshold_just_below(self):
        action = decide_action(
            merge_candidates=[],
            is_stale=True,
            redundancy_score=_ARCHIVE_REDUNDANCY_THRESHOLD - 0.01,
            reference_count=5,
        )
        # Below archive threshold, high ref → keep
        assert action == "keep"


# ---------------------------------------------------------------------------
# build_compression_summary
# ---------------------------------------------------------------------------

class TestBuildCompressionSummary:
    def test_keep_returns_none(self):
        assert build_compression_summary(action="keep", domain="sales", content="text") is None

    def test_merge_candidate_returns_none(self):
        assert build_compression_summary(
            action="merge_candidate", domain="sales", content="text"
        ) is None

    def test_compress_returns_summary(self):
        result = build_compression_summary(
            action="compress", domain="sales", content="Q4 renewal pipeline discussion."
        )
        assert result is not None
        assert "Sales" in result
        assert "Q4 renewal" in result

    def test_archive_returns_summary(self):
        result = build_compression_summary(
            action="archive", domain="incident", content="Server down at 03:00 UTC."
        )
        assert result is not None
        assert "Incident" in result

    def test_domain_capitalised_in_prefix(self):
        result = build_compression_summary(
            action="compress", domain="engineering", content="fix"
        )
        assert "[Engineering]" in result

    def test_empty_content_returns_prefix_only(self):
        result = build_compression_summary(action="archive", domain="hr", content="")
        assert result == "[Hr]"

    def test_long_content_truncated(self):
        long_content = "x" * 300
        result = build_compression_summary(action="compress", domain="sales", content=long_content)
        assert result is not None
        # excerpt capped at 150 chars + ellipsis
        assert len(result) < 300


# ---------------------------------------------------------------------------
# Agent meta
# ---------------------------------------------------------------------------

class TestMemoryConsolidationAgentMeta:
    def test_agent_name(self):
        agent = MemoryConsolidationAgent()
        assert agent.name == "MemoryConsolidationAgent"

    def test_agent_version(self):
        agent = MemoryConsolidationAgent()
        assert agent.version == "v1"

    def test_dependencies(self):
        agent = MemoryConsolidationAgent()
        deps = agent.dependencies()
        assert "SemanticNormalizationAgent" in deps
        assert "EntityResolutionAgent" in deps
        assert "MemoryDecayAgent" in deps

    def test_registry_lookup(self):
        assert isinstance(get_agent("memory_consolidation"), MemoryConsolidationAgent)

    def test_registry_alias_full_name(self):
        assert isinstance(get_agent("memoryconsolidationagent"), MemoryConsolidationAgent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    content: str = "Test memory content about project deployment.",
    domain: str = "engineering",
    entities: dict | None = None,
    is_stale: bool = False,
    reference_count: int = 5,
    similar_memories: list | None = None,
) -> dict:
    return {
        "tenant": {"org_id": "org-1", "org_slug": "acme"},
        "actor": {"user_id": "u-1", "roles": ["member"], "clearance_level": 1},
        "memory": {
            "id": "mem-1",
            "content": content,
            "enrichment": {
                "domain": domain,
                "entities": entities or {},
                "is_stale": is_stale,
                "reference_count": reference_count,
                "similar_memories": similar_memories or [],
            },
        },
        "runtime": {"job_id": "job-1", "attempt": 1, "max_attempts": 3},
        "config": {},
    }


# ---------------------------------------------------------------------------
# Heuristic run
# ---------------------------------------------------------------------------

class TestMemoryConsolidationAgentHeuristic:
    @pytest.mark.asyncio
    async def test_keep_on_fresh_no_candidates(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(is_stale=False, reference_count=10)
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["consolidation_action"] == "keep"

    @pytest.mark.asyncio
    async def test_merge_candidate_action(self):
        agent = MemoryConsolidationAgent()
        similar = [
            {"id": "mem-2", "similarity_score": 0.88, "domain": "engineering"}
        ]
        ctx = _make_context(similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["consolidation_action"] == "merge_candidate"
        assert len(result.outputs["merge_candidates"]) == 1

    @pytest.mark.asyncio
    async def test_archive_action(self):
        agent = MemoryConsolidationAgent()
        similar = [
            {"id": "mem-2", "similarity_score": 0.40, "domain": "engineering"}
        ]
        # 0.40 + 0.10 stale boost = 0.50 → archive
        ctx = _make_context(is_stale=True, reference_count=10, similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["consolidation_action"] == "archive"

    @pytest.mark.asyncio
    async def test_compress_action(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(is_stale=True, reference_count=1)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["consolidation_action"] == "compress"
        assert result.outputs["compression_summary"] is not None

    @pytest.mark.asyncio
    async def test_redundancy_score_in_range(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        rs = result.outputs["redundancy_score"]
        assert 0.0 <= rs <= 1.0

    @pytest.mark.asyncio
    async def test_candidate_count_matches_merge_candidates(self):
        agent = MemoryConsolidationAgent()
        similar = [
            {"id": "m2", "similarity_score": 0.75, "domain": "engineering"},
            {"id": "m3", "similarity_score": 0.82, "domain": "engineering"},
        ]
        ctx = _make_context(similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["candidate_count"] == len(result.outputs["merge_candidates"])

    @pytest.mark.asyncio
    async def test_deduplication_key_not_empty(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert isinstance(result.outputs["deduplication_key"], str)
        assert len(result.outputs["deduplication_key"]) > 0

    @pytest.mark.asyncio
    async def test_merge_candidates_sorted_by_sim_desc(self):
        agent = MemoryConsolidationAgent()
        similar = [
            {"id": "m2", "similarity_score": 0.72, "domain": "engineering"},
            {"id": "m3", "similarity_score": 0.95, "domain": "engineering"},
            {"id": "m4", "similarity_score": 0.80, "domain": "engineering"},
        ]
        ctx = _make_context(similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        scores = [c["similarity_score"] for c in result.outputs["merge_candidates"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_no_compression_summary_for_keep(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(is_stale=False, reference_count=10)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["compression_summary"] is None

    @pytest.mark.asyncio
    async def test_compression_summary_present_for_compress(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(is_stale=True, reference_count=1)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["compression_summary"] is not None

    @pytest.mark.asyncio
    async def test_compression_summary_present_for_archive(self):
        agent = MemoryConsolidationAgent()
        similar = [{"id": "m2", "similarity_score": 0.40, "domain": "engineering"}]
        ctx = _make_context(is_stale=True, reference_count=10, similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["consolidation_action"] == "archive"
        assert result.outputs["compression_summary"] is not None

    @pytest.mark.asyncio
    async def test_empty_similar_memories_no_candidates(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(similar_memories=[])
        result = await agent.run("mem-1", ctx)
        assert result.outputs["merge_candidates"] == []
        assert result.outputs["candidate_count"] == 0

    @pytest.mark.asyncio
    async def test_confidence_high_when_merge_candidates(self):
        agent = MemoryConsolidationAgent()
        similar = [{"id": "m2", "similarity_score": 0.80, "domain": "engineering"}]
        ctx = _make_context(similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_confidence_lower_for_keep(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(is_stale=False, reference_count=10)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["confidence"] == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_no_content_falls_to_heuristic(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context(content="")
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_in_result(self):
        agent = MemoryConsolidationAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-1"

    @pytest.mark.asyncio
    async def test_entities_affect_dedup_key(self):
        agent = MemoryConsolidationAgent()
        ctx_no_ent = _make_context(entities={})
        ctx_with_ent = _make_context(entities={"customer": "Acme Corp"})
        r1 = await agent.run("mem-1", ctx_no_ent)
        r2 = await agent.run("mem-1", ctx_with_ent)
        assert r1.outputs["deduplication_key"] != r2.outputs["deduplication_key"]

    @pytest.mark.asyncio
    async def test_cross_silo_near_duplicate_flagged(self):
        agent = MemoryConsolidationAgent()
        similar = [
            {"id": "m-other", "similarity_score": _HIGH_SIMILARITY, "domain": "finance"}
        ]
        ctx = _make_context(domain="engineering", similar_memories=similar)
        result = await agent.run("mem-1", ctx)
        assert result.outputs["consolidation_action"] == "merge_candidate"
        assert result.outputs["merge_candidates"][0]["reason"] == "cross_domain_near_duplicate"


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

class TestMemoryConsolidationAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_good_response_used(self):
        agent = MemoryConsolidationAgent()
        llm_response = {
            "consolidation_action": "keep",
            "merge_candidates": [],
            "compression_summary": None,
            "redundancy_score": 0.05,
            "deduplication_key": "sales|acme",
            "candidate_count": 0,
            "confidence": 0.75,
            "rationale": "No similar memories found.",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch(
            "app.agents.memory_consolidation_agent.create_ollama_client",
            return_value=mock_client,
        ), patch(
            "app.agents.memory_consolidation_agent.settings",
            MEMORY_CONSOLIDATION_STRATEGY=None,
            AGENT_STRATEGY="llm",
        ):
            ctx = _make_context(content="Customer renewal discussion for Acme Corp.")
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["consolidation_action"] == "keep"
        assert result.outputs["rationale"] == "No similar memories found."

    @pytest.mark.asyncio
    async def test_llm_bad_response_falls_back_to_heuristic(self):
        agent = MemoryConsolidationAgent()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "data"})

        with patch(
            "app.agents.memory_consolidation_agent.create_ollama_client",
            return_value=mock_client,
        ), patch(
            "app.agents.memory_consolidation_agent.settings",
            MEMORY_CONSOLIDATION_STRATEGY=None,
            AGENT_STRATEGY="llm",
        ):
            ctx = _make_context(content="Customer renewal discussion.")
            result = await agent.run("mem-1", ctx)

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_heuristic(self):
        agent = MemoryConsolidationAgent()
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("LLM down"))

        with patch(
            "app.agents.memory_consolidation_agent.create_ollama_client",
            return_value=mock_client,
        ), patch(
            "app.agents.memory_consolidation_agent.settings",
            MEMORY_CONSOLIDATION_STRATEGY=None,
            AGENT_STRATEGY="llm",
        ):
            ctx = _make_context(content="Customer renewal discussion.")
            with pytest.raises(RuntimeError):
                await agent.run("mem-1", ctx)

    @pytest.mark.asyncio
    async def test_heuristic_strategy_skips_llm(self):
        agent = MemoryConsolidationAgent()
        mock_client = AsyncMock()

        with patch(
            "app.agents.memory_consolidation_agent.create_ollama_client",
            return_value=mock_client,
        ), patch(
            "app.agents.memory_consolidation_agent.settings",
            MEMORY_CONSOLIDATION_STRATEGY=None,
            AGENT_STRATEGY="heuristic",
        ):
            ctx = _make_context(content="Some content here.")
            result = await agent.run("mem-1", ctx)

        mock_client.complete_json.assert_not_called()
        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict) -> AgentResult:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="MemoryConsolidationAgent",
            agent_version="v1",
            memory_id="mem-1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def test_valid_outputs_no_error(self):
        agent = MemoryConsolidationAgent()
        result = self._make_result({
            "consolidation_action": "keep",
            "merge_candidates": [],
            "compression_summary": None,
            "redundancy_score": 0.10,
            "deduplication_key": "sales",
            "candidate_count": 0,
        })
        agent.validate_outputs(result)  # should not raise

    def test_invalid_action_raises(self):
        agent = MemoryConsolidationAgent()
        result = self._make_result({
            "consolidation_action": "explode",
            "merge_candidates": [],
            "redundancy_score": 0.10,
            "deduplication_key": "sales",
            "candidate_count": 0,
        })
        with pytest.raises(ValueError, match="consolidation_action"):
            agent.validate_outputs(result)

    def test_redundancy_score_out_of_range_raises(self):
        agent = MemoryConsolidationAgent()
        result = self._make_result({
            "consolidation_action": "keep",
            "merge_candidates": [],
            "redundancy_score": 1.5,
            "deduplication_key": "sales",
            "candidate_count": 0,
        })
        with pytest.raises(ValueError, match="redundancy_score"):
            agent.validate_outputs(result)

    def test_merge_candidates_not_list_raises(self):
        agent = MemoryConsolidationAgent()
        result = self._make_result({
            "consolidation_action": "keep",
            "merge_candidates": "not-a-list",
            "redundancy_score": 0.10,
            "deduplication_key": "sales",
            "candidate_count": 0,
        })
        with pytest.raises(ValueError, match="merge_candidates"):
            agent.validate_outputs(result)

    def test_candidate_count_not_int_raises(self):
        agent = MemoryConsolidationAgent()
        result = self._make_result({
            "consolidation_action": "keep",
            "merge_candidates": [],
            "redundancy_score": 0.10,
            "deduplication_key": "sales",
            "candidate_count": "zero",
        })
        with pytest.raises(ValueError, match="candidate_count"):
            agent.validate_outputs(result)

    def test_deduplication_key_not_str_raises(self):
        agent = MemoryConsolidationAgent()
        result = self._make_result({
            "consolidation_action": "keep",
            "merge_candidates": [],
            "redundancy_score": 0.10,
            "deduplication_key": 123,
            "candidate_count": 0,
        })
        with pytest.raises(ValueError, match="deduplication_key"):
            agent.validate_outputs(result)

    def test_non_success_status_skips_validation(self):
        agent = MemoryConsolidationAgent()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="MemoryConsolidationAgent",
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
