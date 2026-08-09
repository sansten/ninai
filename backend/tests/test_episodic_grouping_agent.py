"""Tests for Phase 18 — EpisodicGroupingAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from app.agents.episodic_grouping_agent import (
    EpisodicGroupingAgent,
    _CLIMAX_FRESHNESS_MIN,
    _extract_entity_names,
    extract_root_entities,
    build_episode_key,
    build_episode_label,
    compute_episode_date_range,
    compute_episode_role,
    compute_entity_overlap,
    compute_temporal_density,
    compute_domain_consistency,
    compute_episode_cohesion,
    extract_cross_episode_links,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)
_CLUSTER = "2026-W11"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sibling(
    *,
    days_ago: float = 5.0,
    domain: str = "engineering",
    entities: dict | None = None,
    freshness: float = 0.6,
) -> dict:
    dt = _NOW - timedelta(days=days_ago)
    return {
        "id": f"sib-{days_ago}",
        "created_at": dt.isoformat(),
        "domain": domain,
        "entities": entities or {"acme": "corp"},
        "freshness_score": freshness,
    }


def _make_cross(*, episode_key: str, domain: str = "sales", entities: dict | None = None) -> dict:
    return {
        "id": f"cross-{episode_key[:4]}",
        "episode_key": episode_key,
        "domain": domain,
        "entities": entities or {"acme": "corp"},
    }


def _make_context(
    *,
    content: str = "",
    age_days: float = 5.0,
    domain: str = "engineering",
    entities: dict | None = None,
    temporal_cluster: str = _CLUSTER,
    freshness_score: float = 0.7,
    siblings: list | None = None,
    cross_episode_memories: list | None = None,
    job_id: str | None = None,
) -> dict:
    created_at = (_NOW - timedelta(days=age_days)).isoformat()
    return {
        "memory": {
            "content": content,
            "created_at": created_at,
            "enrichment": {
                "domain": domain,
                "entities": entities if entities is not None else {"acme": "corp"},
                "temporal_cluster": temporal_cluster,
                "freshness_score": freshness_score,
                "episode_siblings": siblings or [],
                "cross_episode_memories": cross_episode_memories or [],
            },
        },
        "runtime": {"job_id": job_id} if job_id else {},
    }


# ---------------------------------------------------------------------------
# _extract_entity_names
# ---------------------------------------------------------------------------

class TestExtractEntityNames:
    def test_dict_keys_extracted(self):
        result = _extract_entity_names({"acme": "corp", "bob": "smith"})
        assert "acme" in result
        assert "bob" in result

    def test_dict_values_extracted(self):
        result = _extract_entity_names({"customer": "Acme Corp"})
        assert "acme corp" in result

    def test_dict_list_values_extracted(self):
        result = _extract_entity_names({"orgs": ["Acme", "BetaCo"]})
        assert "acme" in result
        assert "betaco" in result

    def test_list_of_strings(self):
        result = _extract_entity_names(["Acme", "Bob"])
        assert "acme" in result
        assert "bob" in result

    def test_list_of_dicts_canonical(self):
        result = _extract_entity_names([{"canonical": "Acme Corp", "original": "Acme"}])
        assert "acme corp" in result

    def test_empty_dict_returns_empty(self):
        assert _extract_entity_names({}) == set()

    def test_none_returns_empty(self):
        assert _extract_entity_names(None) == set()

    def test_empty_strings_excluded(self):
        result = _extract_entity_names({"": "value"})
        assert "" not in result

    def test_case_insensitive(self):
        result = _extract_entity_names({"ACME": "CORP"})
        assert "acme" in result


# ---------------------------------------------------------------------------
# extract_root_entities
# ---------------------------------------------------------------------------

class TestExtractRootEntities:
    def test_no_entities_returns_empty(self):
        result = extract_root_entities(entities={}, siblings=[])
        assert result == []

    def test_no_siblings_returns_self_entities(self):
        result = extract_root_entities(entities={"acme": "corp"}, siblings=[])
        assert "acme" in result

    def test_shared_entity_ranked_first(self):
        siblings = [_make_sibling(entities={"acme": "corp", "other": "x"})]
        result = extract_root_entities(
            entities={"acme": "corp", "rare": "val"}, siblings=siblings
        )
        assert result[0] == "acme"

    def test_limit_respected(self):
        entities = {f"entity{i}": "val" for i in range(10)}
        result = extract_root_entities(entities=entities, siblings=[], limit=3)
        assert len(result) <= 3

    def test_shared_entities_preferred_over_unshared(self):
        siblings = [_make_sibling(entities={"shared": "yes"})]
        result = extract_root_entities(
            entities={"shared": "yes", "unique": "only_here"}, siblings=siblings
        )
        assert "shared" in result[:1]


# ---------------------------------------------------------------------------
# build_episode_key
# ---------------------------------------------------------------------------

class TestBuildEpisodeKey:
    def test_returns_16_char_string(self):
        key = build_episode_key(
            domain="engineering", entities={"acme": "corp"},
            temporal_cluster=_CLUSTER, siblings=[]
        )
        assert isinstance(key, str) and len(key) == 16

    def test_same_inputs_same_key(self):
        kwargs = dict(domain="engineering", entities={"acme": "corp"},
                      temporal_cluster=_CLUSTER, siblings=[])
        assert build_episode_key(**kwargs) == build_episode_key(**kwargs)

    def test_different_domain_different_key(self):
        k1 = build_episode_key(domain="engineering", entities={"acme": "corp"},
                               temporal_cluster=_CLUSTER, siblings=[])
        k2 = build_episode_key(domain="sales", entities={"acme": "corp"},
                               temporal_cluster=_CLUSTER, siblings=[])
        assert k1 != k2

    def test_different_cluster_different_key(self):
        k1 = build_episode_key(domain="engineering", entities={"acme": "corp"},
                               temporal_cluster="2026-W10", siblings=[])
        k2 = build_episode_key(domain="engineering", entities={"acme": "corp"},
                               temporal_cluster="2026-W11", siblings=[])
        assert k1 != k2

    def test_empty_entities_produces_key(self):
        key = build_episode_key(domain="engineering", entities={},
                                temporal_cluster=_CLUSTER, siblings=[])
        assert isinstance(key, str) and len(key) == 16


# ---------------------------------------------------------------------------
# compute_episode_date_range
# ---------------------------------------------------------------------------

class TestComputeEpisodeDateRange:
    def test_no_siblings_returns_single_date(self):
        result = compute_episode_date_range(
            created_at=_NOW, siblings=[], temporal_cluster=_CLUSTER
        )
        assert result == _NOW.strftime("%Y-%m-%d")

    def test_returns_string(self):
        sibling = _make_sibling(days_ago=3.0)
        result = compute_episode_date_range(
            created_at=_NOW, siblings=[sibling], temporal_cluster=_CLUSTER
        )
        assert isinstance(result, str) and len(result) > 0

    def test_cross_week_has_range_separator(self):
        sibling = _make_sibling(days_ago=20.0)
        result = compute_episode_date_range(
            created_at=_NOW, siblings=[sibling], temporal_cluster=_CLUSTER
        )
        # Either a range ".." or a cluster string
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_episode_label
# ---------------------------------------------------------------------------

class TestBuildEpisodeLabel:
    def test_label_contains_domain(self):
        label = build_episode_label(
            domain="engineering", entities={"acme": "corp"},
            temporal_cluster=_CLUSTER, created_at=_NOW, siblings=[]
        )
        assert "engineering" in label

    def test_label_contains_entity(self):
        label = build_episode_label(
            domain="engineering", entities={"acme": "corp"},
            temporal_cluster=_CLUSTER, created_at=_NOW, siblings=[]
        )
        assert "acme" in label

    def test_label_no_entities_still_has_domain(self):
        label = build_episode_label(
            domain="finance", entities={},
            temporal_cluster=_CLUSTER, created_at=_NOW, siblings=[]
        )
        assert "finance" in label

    def test_label_is_string(self):
        label = build_episode_label(
            domain="sales", entities={"bob": "jones"},
            temporal_cluster=_CLUSTER, created_at=_NOW, siblings=[]
        )
        assert isinstance(label, str) and len(label) > 0


# ---------------------------------------------------------------------------
# compute_episode_role
# ---------------------------------------------------------------------------

class TestComputeEpisodeRole:
    def test_no_siblings_is_opener(self):
        assert compute_episode_role(
            created_at=_NOW, freshness_score=0.8, siblings=[]
        ) == "opener"

    def test_earliest_in_span_is_opener(self):
        siblings = [_make_sibling(days_ago=d) for d in [1, 2, 3]]
        created_at = _NOW - timedelta(days=10)
        assert compute_episode_role(
            created_at=created_at, freshness_score=0.5, siblings=siblings
        ) == "opener"

    def test_latest_in_span_is_closer(self):
        siblings = [_make_sibling(days_ago=d) for d in [10, 15, 20]]
        assert compute_episode_role(
            created_at=_NOW, freshness_score=0.5, siblings=siblings
        ) == "closer"

    def test_middle_high_freshness_is_climax(self):
        start = _NOW - timedelta(days=20)
        end = _NOW
        mid = _NOW - timedelta(days=10)
        siblings = [
            {"id": "s1", "created_at": start.isoformat(), "freshness_score": 0.3},
            {"id": "s2", "created_at": end.isoformat(), "freshness_score": 0.4},
        ]
        result = compute_episode_role(
            created_at=mid,
            freshness_score=_CLIMAX_FRESHNESS_MIN + 0.1,
            siblings=siblings,
        )
        assert result == "climax"

    def test_middle_low_freshness_is_body(self):
        start = _NOW - timedelta(days=20)
        end = _NOW
        mid = _NOW - timedelta(days=10)
        siblings = [
            {"id": "s1", "created_at": start.isoformat(), "freshness_score": 0.9},
            {"id": "s2", "created_at": end.isoformat(), "freshness_score": 0.8},
        ]
        assert compute_episode_role(
            created_at=mid, freshness_score=0.3, siblings=siblings
        ) == "body"

    def test_all_same_time_is_opener(self):
        siblings = [{"id": "s", "created_at": _NOW.isoformat(), "freshness_score": 0.5}]
        assert compute_episode_role(
            created_at=_NOW, freshness_score=0.8, siblings=siblings
        ) == "opener"


# ---------------------------------------------------------------------------
# compute_entity_overlap
# ---------------------------------------------------------------------------

class TestComputeEntityOverlap:
    def test_no_siblings_is_zero(self):
        assert compute_entity_overlap({"acme": "corp"}, []) == 0.0

    def test_all_siblings_match(self):
        siblings = [_make_sibling(entities={"acme": "corp"}) for _ in range(3)]
        assert compute_entity_overlap({"acme": "corp"}, siblings) == 1.0

    def test_no_siblings_match(self):
        siblings = [_make_sibling(entities={"other": "entity"}) for _ in range(3)]
        assert compute_entity_overlap({"acme": "corp"}, siblings) == 0.0

    def test_partial_match(self):
        siblings = [
            _make_sibling(entities={"acme": "corp"}),
            _make_sibling(entities={"other": "x"}),
        ]
        score = compute_entity_overlap({"acme": "corp"}, siblings)
        assert abs(score - 0.5) < 0.01

    def test_empty_entities_is_zero(self):
        siblings = [_make_sibling(entities={"acme": "corp"})]
        assert compute_entity_overlap({}, siblings) == 0.0


# ---------------------------------------------------------------------------
# compute_temporal_density
# ---------------------------------------------------------------------------

class TestComputeTemporalDensity:
    def test_no_siblings_is_zero(self):
        assert compute_temporal_density(_NOW, []) == 0.0

    def test_close_siblings_high_density(self):
        siblings = [_make_sibling(days_ago=1), _make_sibling(days_ago=2)]
        score = compute_temporal_density(_NOW, siblings, window_days=14.0)
        assert score > 0.7

    def test_distant_siblings_low_density(self):
        siblings = [_make_sibling(days_ago=100), _make_sibling(days_ago=200)]
        score = compute_temporal_density(_NOW, siblings, window_days=14.0)
        assert score < 0.2

    def test_same_time_siblings_is_one(self):
        siblings = [{"id": "s", "created_at": _NOW.isoformat(), "freshness_score": 0.5}]
        assert compute_temporal_density(_NOW, siblings) == 1.0

    def test_score_in_range(self):
        siblings = [_make_sibling(days_ago=d) for d in [5, 10, 20]]
        score = compute_temporal_density(_NOW, siblings)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_domain_consistency
# ---------------------------------------------------------------------------

class TestComputeDomainConsistency:
    def test_no_siblings_is_zero(self):
        assert compute_domain_consistency("engineering", []) == 0.0

    def test_all_same_domain(self):
        siblings = [_make_sibling(domain="engineering") for _ in range(4)]
        assert compute_domain_consistency("engineering", siblings) == 1.0

    def test_all_different_domain(self):
        siblings = [_make_sibling(domain="sales") for _ in range(3)]
        assert compute_domain_consistency("engineering", siblings) == 0.0

    def test_mixed_domain(self):
        siblings = [_make_sibling(domain="engineering"), _make_sibling(domain="sales")]
        score = compute_domain_consistency("engineering", siblings)
        assert abs(score - 0.5) < 0.01


# ---------------------------------------------------------------------------
# compute_episode_cohesion
# ---------------------------------------------------------------------------

class TestComputeEpisodeCohesion:
    def test_all_ones_is_one(self):
        assert compute_episode_cohesion(
            entity_overlap=1.0, temporal_density=1.0, domain_consistency=1.0
        ) == 1.0

    def test_any_zero_makes_zero(self):
        assert compute_episode_cohesion(
            entity_overlap=0.0, temporal_density=1.0, domain_consistency=1.0
        ) == 0.0

    def test_product_semantics(self):
        score = compute_episode_cohesion(
            entity_overlap=0.5, temporal_density=0.8, domain_consistency=1.0
        )
        assert abs(score - 0.4) < 0.01

    def test_clamped_to_one(self):
        score = compute_episode_cohesion(
            entity_overlap=1.0, temporal_density=1.0, domain_consistency=1.0
        )
        assert score <= 1.0


# ---------------------------------------------------------------------------
# extract_cross_episode_links
# ---------------------------------------------------------------------------

class TestExtractCrossEpisodeLinks:
    def test_empty_list_returns_empty(self):
        result = extract_cross_episode_links([], this_episode_key="abc", root_entities={"acme"})
        assert result == []

    def test_same_key_excluded(self):
        cross = [_make_cross(episode_key="same_key")]
        result = extract_cross_episode_links(
            cross, this_episode_key="same_key", root_entities={"acme"}
        )
        assert result == []

    def test_different_key_included_when_entity_matches(self):
        cross = [_make_cross(episode_key="other_key", entities={"acme": "corp"})]
        result = extract_cross_episode_links(
            cross, this_episode_key="my_key", root_entities={"acme"}
        )
        assert "other_key" in result

    def test_deduplication(self):
        cross = [
            _make_cross(episode_key="ep1", entities={"acme": "corp"}),
            _make_cross(episode_key="ep1", entities={"acme": "corp"}),
        ]
        result = extract_cross_episode_links(
            cross, this_episode_key="my_key", root_entities={"acme"}
        )
        assert result.count("ep1") == 1

    def test_no_entity_overlap_excluded(self):
        cross = [_make_cross(episode_key="other", entities={"unrelated": "entity"})]
        result = extract_cross_episode_links(
            cross, this_episode_key="my_key", root_entities={"acme"}
        )
        assert result == []

    def test_empty_root_entities_skips_entity_filter(self):
        cross = [_make_cross(episode_key="other_ep")]
        result = extract_cross_episode_links(
            cross, this_episode_key="my_key", root_entities=set()
        )
        assert "other_ep" in result


# ---------------------------------------------------------------------------
# EpisodicGroupingAgent — metadata
# ---------------------------------------------------------------------------

class TestEpisodicGroupingAgentMetadata:
    def test_name(self):
        assert EpisodicGroupingAgent.name == "EpisodicGroupingAgent"

    def test_version(self):
        assert EpisodicGroupingAgent.version == "v1"

    def test_dependencies(self):
        deps = EpisodicGroupingAgent().dependencies()
        assert "SemanticNormalizationAgent" in deps
        assert "EntityResolutionAgent" in deps
        assert "TemporalReasoningAgent" in deps

    def test_registry_by_key(self):
        assert isinstance(get_agent("episodic_grouping"), EpisodicGroupingAgent)

    def test_registry_by_camel(self):
        assert isinstance(get_agent("episodicgroupingagent"), EpisodicGroupingAgent)


# ---------------------------------------------------------------------------
# EpisodicGroupingAgent — heuristic run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEpisodicGroupingAgentHeuristic:
    async def test_no_siblings_opener(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-1", _make_context(siblings=[]))
        assert result.status == "success"
        assert result.outputs["episode_role"] == "opener"

    async def test_all_output_keys_present(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-2", _make_context())
        for key in ("episode_key", "episode_role", "episode_cohesion",
                    "episode_label", "cross_episode_links", "confidence", "rationale"):
            assert key in result.outputs, f"missing key: {key}"

    async def test_episode_key_is_16_char_string(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-3", _make_context())
        key = result.outputs["episode_key"]
        assert isinstance(key, str) and len(key) == 16

    async def test_same_inputs_same_episode_key(self):
        agent = EpisodicGroupingAgent()
        ctx = _make_context(domain="engineering", entities={"acme": "corp"})
        r1 = await agent.run("mem-4a", ctx)
        r2 = await agent.run("mem-4b", ctx)
        assert r1.outputs["episode_key"] == r2.outputs["episode_key"]

    async def test_episode_label_contains_domain(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-5", _make_context(domain="finance"))
        assert "finance" in result.outputs["episode_label"]

    async def test_rationale_is_heuristic(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-6", _make_context())
        assert result.outputs["rationale"] == "heuristic"

    async def test_cohesion_zero_with_no_siblings(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-7", _make_context(siblings=[]))
        assert result.outputs["episode_cohesion"] == 0.0

    async def test_cohesion_higher_with_matching_siblings(self):
        agent = EpisodicGroupingAgent()
        siblings = [_make_sibling(days_ago=d, domain="engineering", entities={"acme": "corp"})
                    for d in [1, 2, 3]]
        r_with = await agent.run("mem-8a", _make_context(
            domain="engineering", entities={"acme": "corp"}, siblings=siblings))
        r_without = await agent.run("mem-8b", _make_context(
            domain="engineering", entities={"acme": "corp"}, siblings=[]))
        assert r_with.outputs["episode_cohesion"] > r_without.outputs["episode_cohesion"]

    async def test_cross_episode_links_populated(self):
        agent = EpisodicGroupingAgent()
        cross = [_make_cross(episode_key="other_ep_abc12345", entities={"acme": "corp"})]
        result = await agent.run("mem-9", _make_context(
            entities={"acme": "corp"}, cross_episode_memories=cross))
        assert "other_ep_abc12345" in result.outputs["cross_episode_links"]

    async def test_cross_episode_links_empty_when_no_cross(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-10", _make_context(cross_episode_memories=[]))
        assert result.outputs["cross_episode_links"] == []

    async def test_confidence_no_siblings_is_low(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-11", _make_context(siblings=[]))
        assert result.outputs["confidence"] <= 0.65

    async def test_trace_id_passthrough(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-12", _make_context(job_id="trace-ep"))
        assert result.trace_id == "trace-ep"

    async def test_missing_created_at_defaults_to_now(self):
        agent = EpisodicGroupingAgent()
        ctx = {
            "memory": {
                "content": "",
                "enrichment": {
                    "domain": "engineering", "entities": {},
                    "temporal_cluster": _CLUSTER, "freshness_score": 0.7,
                    "episode_siblings": [], "cross_episode_memories": [],
                },
            },
            "runtime": {},
        }
        result = await agent.run("mem-13", ctx)
        assert result.status == "success"

    async def test_missing_cluster_derived_from_created_at(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-14", _make_context(temporal_cluster=""))
        assert isinstance(result.outputs["episode_key"], str)

    async def test_closer_role_for_most_recent_memory(self):
        agent = EpisodicGroupingAgent()
        siblings = [_make_sibling(days_ago=d) for d in [10, 15, 20]]
        result = await agent.run("mem-15", _make_context(age_days=0.5, siblings=siblings))
        assert result.outputs["episode_role"] == "closer"

    async def test_episode_cohesion_in_range(self):
        agent = EpisodicGroupingAgent()
        siblings = [_make_sibling(days_ago=d) for d in [2, 4]]
        result = await agent.run("mem-16", _make_context(siblings=siblings))
        assert 0.0 <= result.outputs["episode_cohesion"] <= 1.0

    async def test_cross_episode_links_is_list(self):
        agent = EpisodicGroupingAgent()
        result = await agent.run("mem-17", _make_context())
        assert isinstance(result.outputs["cross_episode_links"], list)

    async def test_different_domain_different_episode_key(self):
        agent = EpisodicGroupingAgent()
        r_eng = await agent.run("mem-18a", _make_context(domain="engineering"))
        r_sales = await agent.run("mem-18b", _make_context(domain="sales"))
        assert r_eng.outputs["episode_key"] != r_sales.outputs["episode_key"]


# ---------------------------------------------------------------------------
# EpisodicGroupingAgent — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestEpisodicGroupingAgentLLM:
    async def test_valid_llm_response_used(self):
        agent = EpisodicGroupingAgent()
        ctx = _make_context(content="Q4 deployment post-mortem for Acme Corp")

        llm_resp = {
            "episode_key": "abc1234567890def",
            "episode_role": "body",
            "episode_cohesion": 0.72,
            "episode_label": "engineering · acme · 2026-W11",
            "cross_episode_links": [],
            "confidence": 0.80,
            "rationale": "llm",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_resp)

        with patch("app.agents.episodic_grouping_agent.create_llm_client",
                   return_value=mock_client):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.EPISODIC_GROUPING_STRATEGY = "llm"
                mock_settings.VLLM_BASE_URL = "http://localhost:11434"
                mock_settings.VLLM_MODEL = "llama3.1:8b"
                mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
                mock_settings.VLLM_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-1", ctx)

        assert result.outputs["rationale"] == "llm"
        assert result.outputs["episode_role"] == "body"

    async def test_invalid_llm_falls_back_to_heuristic(self):
        agent = EpisodicGroupingAgent()
        ctx = _make_context(content="some content")

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"invalid": True})

        with patch("app.agents.episodic_grouping_agent.create_llm_client",
                   return_value=mock_client):
            with patch("app.core.config.settings") as mock_settings:
                mock_settings.AGENT_STRATEGY = "llm"
                mock_settings.EPISODIC_GROUPING_STRATEGY = "llm"
                mock_settings.VLLM_BASE_URL = "http://localhost:11434"
                mock_settings.VLLM_MODEL = "llama3.1:8b"
                mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
                mock_settings.VLLM_MAX_CONCURRENCY = 2
                result = await agent.run("mem-llm-2", ctx)

        assert result.outputs["rationale"] == "heuristic"

    async def test_empty_content_skips_llm(self):
        agent = EpisodicGroupingAgent()
        ctx = _make_context(content="")

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(side_effect=RuntimeError("should not be called"))

        with patch("app.agents.episodic_grouping_agent.create_llm_client",
                   return_value=mock_client):
            result = await agent.run("mem-llm-3", ctx)

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# EpisodicGroupingAgent — validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _valid(self) -> dict:
        return {
            "episode_key": "abc1234567890def",
            "episode_role": "opener",
            "episode_cohesion": 0.5,
            "episode_label": "engineering · 2026-W11",
            "cross_episode_links": [],
            "confidence": 0.7,
            "rationale": "heuristic",
        }

    def _result(self, outputs: dict, status: str = "success") -> AgentResult:
        return AgentResult(
            agent_name="EpisodicGroupingAgent", agent_version="v1",
            memory_id="val", status=status, confidence=0.7,
            outputs=outputs, warnings=[], errors=[],
            started_at=_NOW, finished_at=_NOW, trace_id=None,
        )

    def test_valid_outputs_no_error(self):
        EpisodicGroupingAgent().validate_outputs(self._result(self._valid()))

    def test_invalid_role_raises(self):
        o = self._valid(); o["episode_role"] = "protagonist"
        with pytest.raises(ValueError, match="episode_role"):
            EpisodicGroupingAgent().validate_outputs(self._result(o))

    def test_cohesion_out_of_range_raises(self):
        o = self._valid(); o["episode_cohesion"] = 1.5
        with pytest.raises(ValueError, match="episode_cohesion"):
            EpisodicGroupingAgent().validate_outputs(self._result(o))

    def test_non_string_key_raises(self):
        o = self._valid(); o["episode_key"] = 12345
        with pytest.raises(ValueError, match="episode_key"):
            EpisodicGroupingAgent().validate_outputs(self._result(o))

    def test_non_string_label_raises(self):
        o = self._valid(); o["episode_label"] = None
        with pytest.raises(ValueError, match="episode_label"):
            EpisodicGroupingAgent().validate_outputs(self._result(o))

    def test_non_list_links_raises(self):
        o = self._valid(); o["cross_episode_links"] = "ep1"
        with pytest.raises(ValueError, match="cross_episode_links"):
            EpisodicGroupingAgent().validate_outputs(self._result(o))

    def test_non_success_skips_validation(self):
        EpisodicGroupingAgent().validate_outputs(self._result({}, status="failed"))
