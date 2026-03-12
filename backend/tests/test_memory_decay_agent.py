"""Tests for Phase 15 — MemoryDecayAgent."""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from app.agents.memory_decay_agent import (
    MemoryDecayAgent,
    _DEFAULT_HALF_LIFE,
    compute_base_freshness,
    compute_decay_rate,
    compute_expires_at,
    compute_freshness_score,
    compute_recency_boost,
    compute_reference_boost,
    domain_half_life,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# domain_half_life
# ---------------------------------------------------------------------------

class TestDomainHalfLife:
    def test_incident(self):
        assert domain_half_life("incident") == 7.0

    def test_document(self):
        assert domain_half_life("document") == 365.0

    def test_sprint(self):
        assert domain_half_life("sprint") == 14.0

    def test_unknown_returns_default(self):
        assert domain_half_life("zork") == _DEFAULT_HALF_LIFE

    def test_empty_returns_default(self):
        assert domain_half_life("") == _DEFAULT_HALF_LIFE

    def test_case_insensitive(self):
        assert domain_half_life("INCIDENT") == 7.0

    def test_finance(self):
        assert domain_half_life("finance") == 90.0

    def test_strategy(self):
        assert domain_half_life("strategy") == 180.0


# ---------------------------------------------------------------------------
# compute_decay_rate
# ---------------------------------------------------------------------------

class TestComputeDecayRate:
    def test_30_day_half_life(self):
        rate = compute_decay_rate(half_life=30.0)
        # λ = ln(2)/30 ≈ 0.023105
        assert abs(rate - math.log(2) / 30.0) < 1e-5

    def test_7_day_half_life_faster(self):
        rate7 = compute_decay_rate(half_life=7.0)
        rate30 = compute_decay_rate(half_life=30.0)
        assert rate7 > rate30

    def test_at_half_life_freshness_is_half(self):
        hl = 30.0
        rate = compute_decay_rate(half_life=hl)
        freshness = compute_base_freshness(age_days=hl, decay_rate=rate)
        assert abs(freshness - 0.5) < 0.01

    def test_very_small_half_life_does_not_crash(self):
        rate = compute_decay_rate(half_life=0.0)
        assert rate > 0


# ---------------------------------------------------------------------------
# compute_base_freshness
# ---------------------------------------------------------------------------

class TestComputeBaseFreshness:
    def _rate(self, hl=30.0):
        return compute_decay_rate(half_life=hl)

    def test_age_zero_is_one(self):
        assert compute_base_freshness(age_days=0.0, decay_rate=self._rate()) == 1.0

    def test_at_half_life_is_half(self):
        rate = self._rate(30.0)
        fs = compute_base_freshness(age_days=30.0, decay_rate=rate)
        assert abs(fs - 0.5) < 0.01

    def test_large_age_near_zero(self):
        rate = self._rate(7.0)
        fs = compute_base_freshness(age_days=365.0, decay_rate=rate)
        assert fs < 0.01

    def test_negative_age_clamped(self):
        rate = self._rate()
        fs = compute_base_freshness(age_days=-5.0, decay_rate=rate)
        assert fs == 1.0

    def test_monotone_decreasing(self):
        rate = self._rate()
        scores = [compute_base_freshness(age_days=float(d), decay_rate=rate) for d in range(0, 100, 10)]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_result_between_0_and_1(self):
        rate = self._rate()
        assert 0.0 <= compute_base_freshness(age_days=15.0, decay_rate=rate) <= 1.0


# ---------------------------------------------------------------------------
# compute_reference_boost
# ---------------------------------------------------------------------------

class TestComputeReferenceBoost:
    def test_zero_refs(self):
        assert compute_reference_boost(reference_count=0) == 0.0

    def test_one_ref(self):
        assert compute_reference_boost(reference_count=1) == 0.02

    def test_five_refs(self):
        assert compute_reference_boost(reference_count=5) == 0.10

    def test_ten_refs_at_cap(self):
        assert compute_reference_boost(reference_count=10) == 0.20

    def test_over_cap_stays_capped(self):
        assert compute_reference_boost(reference_count=100) == 0.20

    def test_negative_treated_as_zero(self):
        assert compute_reference_boost(reference_count=-3) == 0.0


# ---------------------------------------------------------------------------
# compute_recency_boost
# ---------------------------------------------------------------------------

class TestComputeRecencyBoost:
    def test_accessed_now_is_max(self):
        boost = compute_recency_boost(last_accessed_at=_NOW, now=_NOW)
        assert boost == 0.10

    def test_accessed_3_days_ago(self):
        laa = _NOW - timedelta(days=3)
        boost = compute_recency_boost(last_accessed_at=laa, now=_NOW)
        # 0.10 * (1 - 3/7) ≈ 0.0571
        assert 0.05 < boost < 0.10

    def test_accessed_7_days_ago_is_zero(self):
        laa = _NOW - timedelta(days=7)
        boost = compute_recency_boost(last_accessed_at=laa, now=_NOW)
        assert boost == 0.0

    def test_accessed_14_days_ago_is_zero(self):
        laa = _NOW - timedelta(days=14)
        boost = compute_recency_boost(last_accessed_at=laa, now=_NOW)
        assert boost == 0.0

    def test_none_last_accessed_is_zero(self):
        boost = compute_recency_boost(last_accessed_at=None, now=_NOW)
        assert boost == 0.0

    def test_future_access_clamped_to_zero(self):
        laa = _NOW + timedelta(days=1)
        boost = compute_recency_boost(last_accessed_at=laa, now=_NOW)
        assert boost == 0.0


# ---------------------------------------------------------------------------
# compute_freshness_score
# ---------------------------------------------------------------------------

class TestComputeFreshnessScore:
    def test_all_components(self):
        score = compute_freshness_score(base=0.60, reference_boost=0.10, recency_boost=0.05)
        assert score == 0.75

    def test_capped_at_one(self):
        score = compute_freshness_score(base=0.90, reference_boost=0.20, recency_boost=0.10)
        assert score == 1.0

    def test_all_zero(self):
        score = compute_freshness_score(base=0.0, reference_boost=0.0, recency_boost=0.0)
        assert score == 0.0

    def test_just_base(self):
        score = compute_freshness_score(base=0.45, reference_boost=0.0, recency_boost=0.0)
        assert score == 0.45


# ---------------------------------------------------------------------------
# compute_expires_at
# ---------------------------------------------------------------------------

class TestComputeExpiresAt:
    def test_30_day_half_life_expires_at_90(self):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        expires = compute_expires_at(created_at=created, half_life=30.0)
        assert expires == datetime(2026, 4, 1, tzinfo=timezone.utc)

    def test_incident_7_day_expires_at_21(self):
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        expires = compute_expires_at(created_at=created, half_life=7.0)
        assert expires == created + timedelta(days=21)

    def test_expires_after_created(self):
        created = _NOW
        expires = compute_expires_at(created_at=created, half_life=30.0)
        assert expires > created


# ---------------------------------------------------------------------------
# MemoryDecayAgent — meta
# ---------------------------------------------------------------------------

class TestMemoryDecayAgentMeta:
    def test_name(self):
        assert MemoryDecayAgent().name == "MemoryDecayAgent"

    def test_version(self):
        assert MemoryDecayAgent().version == "v1"

    def test_dependencies(self):
        assert MemoryDecayAgent().dependencies() == ["SemanticNormalizationAgent"]

    def test_registry_lookup(self):
        assert isinstance(get_agent("memory_decay"), MemoryDecayAgent)

    def test_registry_alias(self):
        assert isinstance(get_agent("memorydecayagent"), MemoryDecayAgent)


# ---------------------------------------------------------------------------
# MemoryDecayAgent — heuristic run
# ---------------------------------------------------------------------------

def _make_context(
    *,
    created_at=None,
    domain="engineering",
    reference_count=0,
    last_accessed_at=None,
    content="",
):
    return {
        "memory": {
            "content": content,
            "created_at": created_at or _NOW.isoformat(),
            "enrichment": {
                "domain": domain,
                "reference_count": reference_count,
                "last_accessed_at": last_accessed_at,
            },
        },
        "runtime": {"job_id": "test-decay-1"},
    }


@pytest.mark.asyncio
class TestMemoryDecayAgentHeuristic:
    async def test_brand_new_memory_is_fresh(self):
        agent = MemoryDecayAgent()
        ctx = _make_context(created_at=_NOW.isoformat())
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        # Recently created memories should still be very fresh.
        assert result.outputs["freshness_score"] > 0.94

    async def test_old_incident_memory_is_stale(self):
        agent = MemoryDecayAgent()
        old = (_NOW - timedelta(days=60)).isoformat()
        ctx = _make_context(created_at=old, domain="incident")
        result = await agent.run("mem-2", ctx)
        assert result.outputs["is_stale"] is True

    async def test_fresh_document_not_stale(self):
        agent = MemoryDecayAgent()
        ctx = _make_context(created_at=_NOW.isoformat(), domain="document")
        result = await agent.run("mem-3", ctx)
        assert result.outputs["is_stale"] is False

    async def test_reference_boost_applied(self):
        agent = MemoryDecayAgent()
        # 30-day old engineering memory with 10 references
        old = (_NOW - timedelta(days=30)).isoformat()
        ctx_no_ref = _make_context(created_at=old, domain="engineering", reference_count=0)
        ctx_with_ref = _make_context(created_at=old, domain="engineering", reference_count=10)
        r1 = await agent.run("mem-4a", ctx_no_ref)
        r2 = await agent.run("mem-4b", ctx_with_ref)
        assert r2.outputs["freshness_score"] > r1.outputs["freshness_score"]

    async def test_recency_boost_applied(self):
        agent = MemoryDecayAgent()
        old = (_NOW - timedelta(days=20)).isoformat()
        laa = (_NOW - timedelta(days=1)).isoformat()
        ctx_no_laa = _make_context(created_at=old, domain="sales")
        ctx_with_laa = _make_context(created_at=old, domain="sales", last_accessed_at=laa)
        r1 = await agent.run("mem-5a", ctx_no_laa)
        r2 = await agent.run("mem-5b", ctx_with_laa)
        assert r2.outputs["freshness_score"] > r1.outputs["freshness_score"]

    async def test_domain_sets_half_life(self):
        agent = MemoryDecayAgent()
        ctx = _make_context(created_at=_NOW.isoformat(), domain="finance")
        result = await agent.run("mem-6", ctx)
        assert result.outputs["half_life"] == 90.0

    async def test_age_days_computed(self):
        agent = MemoryDecayAgent()
        # Use a created_at that is 10 days before the actual run time
        created = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        ctx = _make_context(created_at=created)
        result = await agent.run("mem-7", ctx)
        assert abs(result.outputs["age_days"] - 10.0) < 0.1

    async def test_expires_at_is_iso_string(self):
        agent = MemoryDecayAgent()
        ctx = _make_context()
        result = await agent.run("mem-8", ctx)
        # Should parse without error
        datetime.fromisoformat(result.outputs["expires_at"])

    async def test_rationale_is_heuristic(self):
        agent = MemoryDecayAgent()
        ctx = _make_context()
        result = await agent.run("mem-9", ctx)
        assert result.outputs["rationale"] == "heuristic"

    async def test_missing_created_at_defaults_to_now(self):
        agent = MemoryDecayAgent()
        ctx = {
            "memory": {"content": "", "enrichment": {"domain": "sales"}},
            "runtime": {},
        }
        result = await agent.run("mem-10", ctx)
        assert result.status == "success"
        assert result.outputs["age_days"] < 0.01

    async def test_trace_id_in_result(self):
        agent = MemoryDecayAgent()
        ctx = _make_context()
        ctx["runtime"] = {"job_id": "trace-xyz"}
        result = await agent.run("mem-11", ctx)
        assert result.trace_id == "trace-xyz"

    async def test_all_output_keys_present(self):
        agent = MemoryDecayAgent()
        ctx = _make_context()
        result = await agent.run("mem-12", ctx)
        for key in (
            "freshness_score", "decay_rate", "age_days", "half_life",
            "reference_boost", "recency_boost", "expires_at", "is_stale",
            "domain", "confidence", "rationale",
        ):
            assert key in result.outputs, f"missing key: {key}"

    async def test_confidence_scales_with_freshness(self):
        agent = MemoryDecayAgent()
        # Fresh memory → high confidence
        fresh_ctx = _make_context(created_at=_NOW.isoformat(), domain="document")
        stale_ctx = _make_context(
            created_at=(_NOW - timedelta(days=365)).isoformat(), domain="incident"
        )
        r_fresh = await agent.run("mem-13a", fresh_ctx)
        r_stale = await agent.run("mem-13b", stale_ctx)
        assert r_fresh.outputs["confidence"] > r_stale.outputs["confidence"]


# ---------------------------------------------------------------------------
# MemoryDecayAgent — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestMemoryDecayAgentLLM:
    async def test_valid_llm_response_used(self):
        agent = MemoryDecayAgent()
        ctx = _make_context(content="live content about the Q4 deployment")

        llm_resp = {
            "freshness_score": 0.72,
            "decay_rate": 0.023105,
            "age_days": 5.0,
            "half_life": 30.0,
            "reference_boost": 0.04,
            "recency_boost": 0.05,
            "expires_at": "2026-06-15T12:00:00+00:00",
            "is_stale": False,
            "domain": "engineering",
            "confidence": 0.78,
            "rationale": "llm",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)

        with patch(
            "app.agents.memory_decay_agent.create_ollama_client",
            return_value=mock_client,
        ):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.MEMORY_DECAY_STRATEGY = "llm"
                mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
                mock_settings.OLLAMA_MODEL = "llama3.1:8b"
                mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
                mock_settings.OLLAMA_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-1", ctx)

        assert result.outputs["rationale"] == "llm"
        assert result.outputs["freshness_score"] == 0.72

    async def test_bad_llm_response_falls_back_to_heuristic(self):
        agent = MemoryDecayAgent()
        ctx = _make_context(content="live content")

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"invalid": "response"})

        with patch(
            "app.agents.memory_decay_agent.create_ollama_client",
            return_value=mock_client,
        ):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.MEMORY_DECAY_STRATEGY = "llm"
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
    def _make_result(self, outputs, status="success"):
        return AgentResult(
            agent_name="MemoryDecayAgent",
            agent_version="v1",
            memory_id="m",
            status=status,
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
        )

    def test_valid_passes(self):
        agent = MemoryDecayAgent()
        result = self._make_result({
            "freshness_score": 0.75,
            "is_stale": False,
            "age_days": 5.0,
            "expires_at": "2026-06-01T00:00:00+00:00",
        })
        agent.validate_outputs(result)  # no raise

    def test_freshness_score_not_float(self):
        agent = MemoryDecayAgent()
        result = self._make_result({
            "freshness_score": "high",
            "is_stale": False,
            "age_days": 5.0,
            "expires_at": "2026-06-01T00:00:00+00:00",
        })
        with pytest.raises(ValueError, match="freshness_score must be a float"):
            agent.validate_outputs(result)

    def test_freshness_score_out_of_range(self):
        agent = MemoryDecayAgent()
        result = self._make_result({
            "freshness_score": 1.5,
            "is_stale": False,
            "age_days": 5.0,
            "expires_at": "2026-06-01T00:00:00+00:00",
        })
        with pytest.raises(ValueError, match="between 0 and 1"):
            agent.validate_outputs(result)

    def test_is_stale_not_bool(self):
        agent = MemoryDecayAgent()
        result = self._make_result({
            "freshness_score": 0.5,
            "is_stale": "yes",
            "age_days": 5.0,
            "expires_at": "2026-06-01T00:00:00+00:00",
        })
        with pytest.raises(ValueError, match="is_stale must be a bool"):
            agent.validate_outputs(result)

    def test_age_days_negative(self):
        agent = MemoryDecayAgent()
        result = self._make_result({
            "freshness_score": 0.5,
            "is_stale": False,
            "age_days": -1.0,
            "expires_at": "2026-06-01T00:00:00+00:00",
        })
        with pytest.raises(ValueError, match="age_days must be a non-negative float"):
            agent.validate_outputs(result)

    def test_expires_at_not_string(self):
        agent = MemoryDecayAgent()
        result = self._make_result({
            "freshness_score": 0.5,
            "is_stale": False,
            "age_days": 5.0,
            "expires_at": 12345,
        })
        with pytest.raises(ValueError, match="expires_at must be a string"):
            agent.validate_outputs(result)

    def test_non_success_skips_validation(self):
        agent = MemoryDecayAgent()
        result = self._make_result(outputs={}, status="failed")
        agent.validate_outputs(result)  # no raise
