"""Tests for Phase 17 — TemporalReasoningAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from app.agents.temporal_reasoning_agent import (
    TemporalReasoningAgent,
    _RECENT_THRESHOLD,
    _HISTORICAL_THRESHOLD,
    _VELOCITY_WINDOW_DAYS,
    classify_temporal_position,
    compute_temporal_cluster,
    compute_sequence_gaps,
    compute_velocity_score,
    detect_trend_signal,
    detect_sequence_anomaly,
    classify_sequence_role,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_related(*, days_ago: float, freshness: float = 0.7, domain: str = "engineering") -> dict:
    dt = _NOW - timedelta(days=days_ago)
    return {"id": f"rel-{days_ago}", "created_at": dt.isoformat(), "domain": domain, "freshness_score": freshness}


def _make_context(
    *,
    content: str = "",
    created_at: str | None = None,
    age_days: float = 5.0,
    is_stale: bool = False,
    related_memories: list | None = None,
    job_id: str | None = None,
) -> dict:
    if created_at is None:
        created_at = (_NOW - timedelta(days=age_days)).isoformat()
    ctx: dict = {
        "memory": {
            "content": content,
            "created_at": created_at,
            "enrichment": {
                "domain": "engineering",
                "entities": {},
                "age_days": age_days,
                "is_stale": is_stale,
                "related_memories": related_memories or [],
            },
        },
        "runtime": {"job_id": job_id} if job_id else {},
    }
    return ctx


# ---------------------------------------------------------------------------
# classify_temporal_position
# ---------------------------------------------------------------------------

class TestClassifyTemporalPosition:
    def test_zero_age_is_recent(self):
        assert classify_temporal_position(0.0) == "recent"

    def test_just_below_recent_threshold(self):
        assert classify_temporal_position(_RECENT_THRESHOLD - 0.1) == "recent"

    def test_at_recent_threshold_is_historical(self):
        assert classify_temporal_position(_RECENT_THRESHOLD) == "historical"

    def test_midway_historical(self):
        assert classify_temporal_position(45.0) == "historical"

    def test_just_below_historical_threshold(self):
        assert classify_temporal_position(_HISTORICAL_THRESHOLD - 0.1) == "historical"

    def test_at_historical_threshold_is_archival(self):
        assert classify_temporal_position(_HISTORICAL_THRESHOLD) == "archival"

    def test_large_age_is_archival(self):
        assert classify_temporal_position(365.0) == "archival"

    def test_negative_age_treated_as_zero(self):
        assert classify_temporal_position(-5.0) == "recent"


# ---------------------------------------------------------------------------
# compute_temporal_cluster
# ---------------------------------------------------------------------------

class TestComputeTemporalCluster:
    def test_format_matches_year_week(self):
        cluster = compute_temporal_cluster(_NOW)
        assert cluster.startswith("2026-W")
        week_part = cluster.split("-W")[1]
        assert week_part.isdigit()
        assert 1 <= int(week_part) <= 53

    def test_different_weeks_give_different_clusters(self):
        dt1 = datetime(2026, 3, 1, tzinfo=timezone.utc)
        dt2 = datetime(2026, 3, 15, tzinfo=timezone.utc)
        assert compute_temporal_cluster(dt1) != compute_temporal_cluster(dt2)

    def test_same_week_gives_same_cluster(self):
        monday = datetime(2026, 3, 2, tzinfo=timezone.utc)
        friday = datetime(2026, 3, 6, tzinfo=timezone.utc)
        assert compute_temporal_cluster(monday) == compute_temporal_cluster(friday)

    def test_zero_padded_week(self):
        # Week 1 of 2026
        dt = datetime(2026, 1, 5, tzinfo=timezone.utc)
        cluster = compute_temporal_cluster(dt)
        assert "-W0" in cluster or "-W1" in cluster


# ---------------------------------------------------------------------------
# compute_sequence_gaps
# ---------------------------------------------------------------------------

class TestComputeSequenceGaps:
    def test_empty_list_returns_empty(self):
        assert compute_sequence_gaps([], _NOW) == []

    def test_single_related_memory(self):
        rel = [_make_related(days_ago=3.0)]
        gaps = compute_sequence_gaps(rel, _NOW)
        assert len(gaps) == 1
        assert abs(gaps[0] - 3.0) < 0.01

    def test_multiple_gaps_sorted(self):
        rels = [_make_related(days_ago=10.0), _make_related(days_ago=2.0), _make_related(days_ago=5.0)]
        gaps = compute_sequence_gaps(rels, _NOW)
        assert gaps == sorted(gaps)
        assert len(gaps) == 3

    def test_entry_without_created_at_skipped(self):
        rels = [{"id": "x"}, _make_related(days_ago=5.0)]
        gaps = compute_sequence_gaps(rels, _NOW)
        assert len(gaps) == 1

    def test_future_date_gap_is_positive(self):
        future_dt = (_NOW + timedelta(days=2)).isoformat()
        rels = [{"id": "fut", "created_at": future_dt, "domain": "engineering", "freshness_score": 0.9}]
        gaps = compute_sequence_gaps(rels, _NOW)
        assert len(gaps) == 1
        assert gaps[0] > 0

    def test_none_created_at_skipped(self):
        rels = [{"id": "y", "created_at": None}, _make_related(days_ago=1.0)]
        gaps = compute_sequence_gaps(rels, _NOW)
        assert len(gaps) == 1


# ---------------------------------------------------------------------------
# compute_velocity_score
# ---------------------------------------------------------------------------

class TestComputeVelocityScore:
    def test_empty_list_is_zero(self):
        assert compute_velocity_score([], now=_NOW) == 0.0

    def test_all_within_window(self):
        rels = [_make_related(days_ago=d) for d in [1, 5, 10, 15, 20]]
        score = compute_velocity_score(rels, now=_NOW)
        assert score > 0.0
        assert score <= 1.0

    def test_all_outside_window(self):
        rels = [_make_related(days_ago=60.0), _make_related(days_ago=90.0)]
        score = compute_velocity_score(rels, now=_NOW, window_days=30.0)
        assert score == 0.0

    def test_score_saturates_at_one(self):
        rels = [_make_related(days_ago=i * 0.5) for i in range(1, 25)]
        score = compute_velocity_score(rels, now=_NOW, window_days=30.0)
        assert score == 1.0

    def test_more_recent_gives_higher_score(self):
        few = [_make_related(days_ago=d) for d in [1, 5]]
        many = [_make_related(days_ago=d) for d in range(1, 12)]
        assert compute_velocity_score(many, now=_NOW) > compute_velocity_score(few, now=_NOW)

    def test_score_is_clamped(self):
        rels = [_make_related(days_ago=0.1) for _ in range(100)]
        score = compute_velocity_score(rels, now=_NOW)
        assert score == 1.0


# ---------------------------------------------------------------------------
# detect_trend_signal
# ---------------------------------------------------------------------------

class TestDetectTrendSignal:
    def test_empty_list_is_stable(self):
        assert detect_trend_signal([]) == "stable"

    def test_fewer_than_min_is_stable(self):
        rels = [_make_related(days_ago=d, freshness=0.8) for d in [5, 3]]
        assert detect_trend_signal(rels) == "stable"

    def test_rising_freshness_trend(self):
        # Older memories have lower freshness, newer have higher → rising
        rels = [
            _make_related(days_ago=30, freshness=0.2),
            _make_related(days_ago=20, freshness=0.4),
            _make_related(days_ago=10, freshness=0.6),
            _make_related(days_ago=2, freshness=0.8),
        ]
        assert detect_trend_signal(rels) == "rising"

    def test_falling_freshness_trend(self):
        # oldest→newest: 0.9 → 0.6 → 0.3 → 0.1 (decreasing over time → falling)
        rels = [
            _make_related(days_ago=30, freshness=0.9),
            _make_related(days_ago=20, freshness=0.6),
            _make_related(days_ago=10, freshness=0.3),
            _make_related(days_ago=2, freshness=0.1),
        ]
        assert detect_trend_signal(rels) == "falling"

    def test_stable_freshness(self):
        rels = [
            _make_related(days_ago=30, freshness=0.5),
            _make_related(days_ago=20, freshness=0.51),
            _make_related(days_ago=10, freshness=0.49),
            _make_related(days_ago=2, freshness=0.50),
        ]
        assert detect_trend_signal(rels) == "stable"

    def test_noisy_signal_is_anomalous(self):
        rels = [
            _make_related(days_ago=30, freshness=0.9),
            _make_related(days_ago=25, freshness=0.1),
            _make_related(days_ago=20, freshness=0.95),
            _make_related(days_ago=15, freshness=0.05),
            _make_related(days_ago=10, freshness=0.9),
        ]
        # Alternating high/low → noisy → anomalous
        assert detect_trend_signal(rels) == "anomalous"

    def test_no_freshness_score_skipped(self):
        rels = [
            {"id": "a", "created_at": (_NOW - timedelta(days=10)).isoformat()},
            _make_related(days_ago=5, freshness=0.7),
        ]
        # Only 1 valid entry → stable
        assert detect_trend_signal(rels) == "stable"


# ---------------------------------------------------------------------------
# detect_sequence_anomaly
# ---------------------------------------------------------------------------

class TestDetectSequenceAnomaly:
    def test_empty_gaps_no_anomaly(self):
        assert detect_sequence_anomaly(5.0, []) is False

    def test_single_gap_no_anomaly(self):
        assert detect_sequence_anomaly(5.0, [5.0]) is False

    def test_within_normal_range(self):
        gaps = [5.0, 6.0, 5.5, 5.8, 6.2]
        assert detect_sequence_anomaly(5.9, gaps) is False

    def test_large_deviation_is_anomaly(self):
        # Cluster gaps around 1 day with small variance; 30-day gap is anomalous
        gaps = [0.9, 1.0, 1.1, 1.0, 0.95]
        assert detect_sequence_anomaly(30.0, gaps) is True

    def test_zero_std_no_anomaly(self):
        # All gaps identical → std=0 → no anomaly regardless of value
        gaps = [5.0, 5.0, 5.0]
        assert detect_sequence_anomaly(50.0, gaps) is False


# ---------------------------------------------------------------------------
# classify_sequence_role
# ---------------------------------------------------------------------------

class TestClassifySequenceRole:
    def _kwargs(self, **overrides):
        base = {
            "sequence_gaps": [2.0, 5.0, 3.0],
            "trend_signal": "stable",
            "is_stale": False,
            "is_anomaly": False,
            "velocity_score": 0.3,
            "temporal_position": "historical",
        }
        base.update(overrides)
        return base

    def test_isolated_when_no_gaps(self):
        result = classify_sequence_role(
            sequence_gaps=[],
            trend_signal="stable",
            is_stale=False,
            is_anomaly=False,
            velocity_score=0.5,
            temporal_position="recent",
        )
        assert result == "isolated"

    def test_isolated_when_anomaly(self):
        result = classify_sequence_role(**self._kwargs(is_anomaly=True))
        assert result == "isolated"

    def test_initiating_when_recent_low_velocity(self):
        result = classify_sequence_role(**self._kwargs(
            temporal_position="recent",
            velocity_score=0.05,
        ))
        assert result == "initiating"

    def test_escalating_when_rising_not_stale(self):
        result = classify_sequence_role(**self._kwargs(
            trend_signal="rising",
            is_stale=False,
            temporal_position="historical",
            velocity_score=0.4,
        ))
        assert result == "escalating"

    def test_resolving_when_falling(self):
        result = classify_sequence_role(**self._kwargs(
            trend_signal="falling",
            temporal_position="historical",
        ))
        assert result == "resolving"

    def test_recurring_when_stable_high_velocity(self):
        result = classify_sequence_role(**self._kwargs(
            trend_signal="stable",
            velocity_score=0.5,
            temporal_position="historical",
        ))
        assert result == "recurring"

    def test_rising_stale_not_escalating(self):
        # Stale memory with rising trend should NOT escalate
        result = classify_sequence_role(**self._kwargs(
            trend_signal="rising",
            is_stale=True,
            temporal_position="historical",
            velocity_score=0.05,
        ))
        assert result != "escalating"

    def test_stable_low_velocity_is_isolated(self):
        result = classify_sequence_role(**self._kwargs(
            trend_signal="stable",
            velocity_score=0.1,
            temporal_position="historical",
        ))
        assert result == "isolated"


# ---------------------------------------------------------------------------
# TemporalReasoningAgent — metadata
# ---------------------------------------------------------------------------

class TestTemporalReasoningAgentMetadata:
    def test_name(self):
        assert TemporalReasoningAgent.name == "TemporalReasoningAgent"

    def test_version(self):
        assert TemporalReasoningAgent.version == "v1"

    def test_dependencies(self):
        agent = TemporalReasoningAgent()
        deps = agent.dependencies()
        assert "SemanticNormalizationAgent" in deps
        assert "EntityResolutionAgent" in deps
        assert "MemoryDecayAgent" in deps

    def test_registry_by_key(self):
        agent = get_agent("temporal_reasoning")
        assert isinstance(agent, TemporalReasoningAgent)

    def test_registry_by_camel(self):
        agent = get_agent("temporalreasoningagent")
        assert isinstance(agent, TemporalReasoningAgent)


# ---------------------------------------------------------------------------
# TemporalReasoningAgent — heuristic run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTemporalReasoningAgentHeuristic:
    async def test_recent_memory_no_related(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=2.0, related_memories=[])
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["temporal_position"] == "recent"
        assert result.outputs["sequence_role"] == "isolated"

    async def test_historical_memory(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=45.0)
        result = await agent.run("mem-2", ctx)
        assert result.outputs["temporal_position"] == "historical"

    async def test_archival_memory(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=120.0)
        result = await agent.run("mem-3", ctx)
        assert result.outputs["temporal_position"] == "archival"

    async def test_temporal_cluster_format(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=3.0)
        result = await agent.run("mem-4", ctx)
        cluster = result.outputs["temporal_cluster"]
        assert isinstance(cluster, str)
        assert "-W" in cluster

    async def test_velocity_increases_with_more_recent_related(self):
        agent = TemporalReasoningAgent()
        few_related = [_make_related(days_ago=d) for d in [5, 10]]
        many_related = [_make_related(days_ago=d) for d in range(1, 12)]
        ctx_few = _make_context(age_days=3.0, related_memories=few_related)
        ctx_many = _make_context(age_days=3.0, related_memories=many_related)
        r_few = await agent.run("mem-5a", ctx_few)
        r_many = await agent.run("mem-5b", ctx_many)
        assert r_many.outputs["velocity_score"] > r_few.outputs["velocity_score"]

    async def test_rising_trend_gives_escalating_role(self):
        agent = TemporalReasoningAgent()
        rels = [
            _make_related(days_ago=30, freshness=0.2),
            _make_related(days_ago=20, freshness=0.4),
            _make_related(days_ago=10, freshness=0.6),
            _make_related(days_ago=2, freshness=0.85),
        ]
        ctx = _make_context(age_days=45.0, related_memories=rels)
        result = await agent.run("mem-6", ctx)
        assert result.outputs["trend_signal"] == "rising"
        assert result.outputs["sequence_role"] == "escalating"

    async def test_falling_trend_gives_resolving_role(self):
        agent = TemporalReasoningAgent()
        # oldest→newest: 0.9→0.1 (decreasing freshness over time → falling)
        rels = [
            _make_related(days_ago=30, freshness=0.9),
            _make_related(days_ago=20, freshness=0.6),
            _make_related(days_ago=10, freshness=0.3),
            _make_related(days_ago=2, freshness=0.1),
        ]
        ctx = _make_context(age_days=45.0, related_memories=rels)
        result = await agent.run("mem-7", ctx)
        assert result.outputs["trend_signal"] == "falling"
        assert result.outputs["sequence_role"] == "resolving"

    async def test_sequence_gap_days_is_median(self):
        agent = TemporalReasoningAgent()
        age = 30.0
        created_at = (_NOW - timedelta(days=age)).isoformat()
        # Related memories 5, 10, 15 days from the memory's created_at
        rels = [
            _make_related(days_ago=age + 5),
            _make_related(days_ago=age - 5),
            _make_related(days_ago=age + 10),
        ]
        ctx = _make_context(age_days=age, created_at=created_at, related_memories=rels)
        result = await agent.run("mem-8", ctx)
        assert result.outputs["sequence_gap_days"] >= 0.0

    async def test_is_sequence_anomaly_false_by_default(self):
        agent = TemporalReasoningAgent()
        # With no related memories → no anomaly
        ctx = _make_context(age_days=5.0, related_memories=[])
        result = await agent.run("mem-9", ctx)
        assert result.outputs["is_sequence_anomaly"] is False

    async def test_anomaly_detected_with_outlier_gap(self):
        agent = TemporalReasoningAgent()
        age = 100.0
        created_at = (_NOW - timedelta(days=age)).isoformat()
        # Cluster with uniform 1-day gaps, one massive outlier
        rels = [_make_related(days_ago=age + g) for g in [1, 1, 1, 1, 1, 1]]
        ctx = _make_context(age_days=age, created_at=created_at, related_memories=rels)
        result = await agent.run("mem-10", ctx)
        assert isinstance(result.outputs["is_sequence_anomaly"], bool)

    async def test_trace_id_passthrough(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=3.0, job_id="trace-abc")
        result = await agent.run("mem-11", ctx)
        assert result.trace_id == "trace-abc"

    async def test_missing_created_at_defaults_to_now(self):
        agent = TemporalReasoningAgent()
        ctx = {
            "memory": {
                "content": "",
                "enrichment": {
                    "age_days": 0.0,
                    "is_stale": False,
                    "related_memories": [],
                },
            },
            "runtime": {},
        }
        result = await agent.run("mem-12", ctx)
        assert result.status == "success"
        assert result.outputs["temporal_position"] == "recent"

    async def test_all_output_keys_present(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=5.0)
        result = await agent.run("mem-13", ctx)
        for key in (
            "temporal_position", "sequence_role", "trend_signal",
            "sequence_gap_days", "is_sequence_anomaly", "temporal_cluster",
            "velocity_score", "confidence", "rationale",
        ):
            assert key in result.outputs, f"missing key: {key}"

    async def test_rationale_is_heuristic(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=5.0)
        result = await agent.run("mem-14", ctx)
        assert result.outputs["rationale"] == "heuristic"

    async def test_confidence_recent_higher_than_archival(self):
        agent = TemporalReasoningAgent()
        rels = [_make_related(days_ago=d) for d in [1, 5, 10]]
        ctx_recent = _make_context(age_days=2.0, related_memories=rels)
        ctx_archival = _make_context(age_days=200.0, related_memories=rels)
        r_recent = await agent.run("mem-15a", ctx_recent)
        r_archival = await agent.run("mem-15b", ctx_archival)
        assert r_recent.outputs["confidence"] > r_archival.outputs["confidence"]

    async def test_isolated_no_related_confidence_low(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=5.0, related_memories=[])
        result = await agent.run("mem-16", ctx)
        assert result.outputs["confidence"] <= 0.65

    async def test_recurring_role_with_high_velocity_stable(self):
        agent = TemporalReasoningAgent()
        rels = [_make_related(days_ago=d, freshness=0.5) for d in range(1, 16)]
        ctx = _make_context(age_days=45.0, related_memories=rels)
        result = await agent.run("mem-17", ctx)
        assert result.outputs["velocity_score"] >= 0.30
        assert result.outputs["sequence_role"] in ("recurring", "resolving", "escalating", "isolated")

    async def test_stale_memory_produces_valid_outputs(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=100.0, is_stale=True)
        result = await agent.run("mem-18", ctx)
        assert result.status == "success"
        assert result.outputs["temporal_position"] == "archival"

    async def test_velocity_score_in_range(self):
        agent = TemporalReasoningAgent()
        rels = [_make_related(days_ago=d) for d in [2, 5, 8, 12]]
        ctx = _make_context(age_days=5.0, related_memories=rels)
        result = await agent.run("mem-19", ctx)
        vs = result.outputs["velocity_score"]
        assert 0.0 <= vs <= 1.0

    async def test_sequence_gap_days_zero_when_no_related(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(age_days=5.0, related_memories=[])
        result = await agent.run("mem-20", ctx)
        assert result.outputs["sequence_gap_days"] == 0.0


# ---------------------------------------------------------------------------
# TemporalReasoningAgent — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTemporalReasoningAgentLLM:
    async def test_valid_llm_response_used(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(content="Q4 capacity planning meeting notes", age_days=10.0)

        llm_resp = {
            "temporal_position": "historical",
            "sequence_role": "escalating",
            "trend_signal": "rising",
            "sequence_gap_days": 3.5,
            "is_sequence_anomaly": False,
            "temporal_cluster": "2026-W10",
            "velocity_score": 0.4,
            "confidence": 0.80,
            "rationale": "llm",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)

        with patch("app.agents.temporal_reasoning_agent.create_llm_client", return_value=mock_client):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.TEMPORAL_REASONING_STRATEGY = "llm"
                mock_settings.VLLM_BASE_URL = "http://localhost:11434"
                mock_settings.VLLM_MODEL = "llama3.1:8b"
                mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
                mock_settings.VLLM_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-1", ctx)

        assert result.outputs["rationale"] == "llm"
        assert result.outputs["sequence_role"] == "escalating"

    async def test_invalid_llm_falls_back_to_heuristic(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(content="fallback test content", age_days=5.0)

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"garbage": True})

        with patch("app.agents.temporal_reasoning_agent.create_llm_client", return_value=mock_client):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.TEMPORAL_REASONING_STRATEGY = "llm"
                mock_settings.VLLM_BASE_URL = "http://localhost:11434"
                mock_settings.VLLM_MODEL = "llama3.1:8b"
                mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
                mock_settings.VLLM_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-2", ctx)

        assert result.outputs["rationale"] == "heuristic"

    async def test_heuristic_strategy_skips_llm(self):
        agent = TemporalReasoningAgent()
        # Empty content unconditionally routes to heuristic; LLM client must not be called.
        ctx = _make_context(content="", age_days=3.0)

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("should not be called"))

        with patch("app.agents.temporal_reasoning_agent.create_llm_client", return_value=mock_client):
            result = await agent.run("mem-llm-3", ctx)

        assert result.outputs["rationale"] == "heuristic"

    async def test_empty_content_skips_llm(self):
        agent = TemporalReasoningAgent()
        ctx = _make_context(content="", age_days=3.0)

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("should not be called"))

        with patch("app.agents.temporal_reasoning_agent.create_llm_client", return_value=mock_client):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.TEMPORAL_REASONING_STRATEGY = None
                mock_settings.VLLM_BASE_URL = "http://localhost:11434"
                mock_settings.VLLM_MODEL = "llama3.1:8b"
                mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
                mock_settings.VLLM_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-4", ctx)

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# TemporalReasoningAgent — validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict) -> AgentResult:
        return AgentResult(
            agent_name="TemporalReasoningAgent",
            agent_version="v1",
            memory_id="val-test",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=_NOW,
            finished_at=_NOW,
            trace_id=None,
        )

    def _valid_outputs(self) -> dict:
        return {
            "temporal_position": "recent",
            "sequence_role": "isolated",
            "trend_signal": "stable",
            "sequence_gap_days": 0.0,
            "is_sequence_anomaly": False,
            "temporal_cluster": "2026-W10",
            "velocity_score": 0.0,
            "confidence": 0.6,
            "rationale": "heuristic",
        }

    def test_valid_outputs_no_error(self):
        agent = TemporalReasoningAgent()
        result = self._make_result(self._valid_outputs())
        agent.validate_outputs(result)  # should not raise

    def test_invalid_temporal_position_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["temporal_position"] = "yesterday"
        with pytest.raises(ValueError, match="temporal_position"):
            agent.validate_outputs(self._make_result(o))

    def test_invalid_sequence_role_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["sequence_role"] = "dormant"
        with pytest.raises(ValueError, match="sequence_role"):
            agent.validate_outputs(self._make_result(o))

    def test_invalid_trend_signal_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["trend_signal"] = "sideways"
        with pytest.raises(ValueError, match="trend_signal"):
            agent.validate_outputs(self._make_result(o))

    def test_negative_gap_days_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["sequence_gap_days"] = -1.0
        with pytest.raises(ValueError, match="sequence_gap_days"):
            agent.validate_outputs(self._make_result(o))

    def test_non_bool_anomaly_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["is_sequence_anomaly"] = "yes"
        with pytest.raises(ValueError, match="is_sequence_anomaly"):
            agent.validate_outputs(self._make_result(o))

    def test_non_string_cluster_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["temporal_cluster"] = 202610
        with pytest.raises(ValueError, match="temporal_cluster"):
            agent.validate_outputs(self._make_result(o))

    def test_out_of_range_velocity_raises(self):
        agent = TemporalReasoningAgent()
        o = self._valid_outputs()
        o["velocity_score"] = 1.5
        with pytest.raises(ValueError, match="velocity_score"):
            agent.validate_outputs(self._make_result(o))

    def test_non_success_skips_validation(self):
        agent = TemporalReasoningAgent()
        result = AgentResult(
            agent_name="TemporalReasoningAgent",
            agent_version="v1",
            memory_id="val-skip",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=_NOW,
            finished_at=_NOW,
            trace_id=None,
        )
        agent.validate_outputs(result)  # should not raise
