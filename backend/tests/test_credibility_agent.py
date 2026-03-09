"""Tests for Phase 19 — CredibilityAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.credibility_agent import (
    CredibilityAgent,
    _TIER_BASE,
    _extract_entity_names,
    classify_source_tier,
    count_citation_depth,
    count_corroboration,
    count_conflict_exposure,
    build_credibility_flags,
    compute_credibility_score,
    compute_agent_confidence,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conflict(
    *,
    entity: str = "acme",
    severity: str = "high",
    conflict_type: str = "state_mismatch",
) -> dict:
    return {
        "entity": entity,
        "entity_type": "org",
        "conflict_type": conflict_type,
        "silos": ["engineering", "sales"],
        "severity": severity,
    }


def _make_corroborating(*, count: int = 1) -> list[dict]:
    return [{"id": f"m{i}", "content": "corroborates", "entities": {}} for i in range(count)]


def _make_context(
    *,
    content: str = "deployment went live on the api",
    source_type: str = "api",
    author_role: str = "engineer",
    entities: dict | None = None,
    conflicts: list | None = None,
    corroborating_memories: list | None = None,
    job_id: str | None = None,
) -> dict:
    return {
        "memory": {
            "content": content,
            "source_type": source_type,
            "author_role": author_role,
            "enrichment": {
                "entities": entities if entities is not None else {"acme": "corp"},
                "conflicts": conflicts if conflicts is not None else [],
                "corroborating_memories": corroborating_memories if corroborating_memories is not None else [],
            },
        },
        "runtime": {"job_id": job_id} if job_id else {},
    }


# ---------------------------------------------------------------------------
# _extract_entity_names
# ---------------------------------------------------------------------------

class TestExtractEntityNames:
    def test_dict_keys_extracted(self):
        result = _extract_entity_names({"acme": "corp", "project_x": "initiative"})
        assert "acme" in result
        assert "project_x" in result

    def test_dict_with_nested_list(self):
        result = _extract_entity_names({
            "people": [{"canonical": "Alice", "original": "alice"}]
        })
        assert "alice" in result

    def test_list_of_dicts(self):
        result = _extract_entity_names([
            {"canonical": "acme", "original": "ACME Inc"},
        ])
        assert "acme" in result

    def test_list_of_strings(self):
        result = _extract_entity_names(["alpha", "beta", "gamma"])
        assert "alpha" in result and "beta" in result

    def test_scalar_string(self):
        result = _extract_entity_names("project_phoenix")
        assert "project_phoenix" in result

    def test_none_returns_empty(self):
        assert _extract_entity_names(None) == set()

    def test_empty_dict(self):
        assert _extract_entity_names({}) == set()

    def test_empty_list(self):
        assert _extract_entity_names([]) == set()

    def test_normalizes_to_lowercase(self):
        result = _extract_entity_names({"ACME": "Corp"})
        assert "acme" in result
        assert "corp" in result

    def test_entity_key_in_list_item(self):
        result = _extract_entity_names([{"entity": "phoenix", "name": "Phoenix Project"}])
        assert "phoenix" in result


# ---------------------------------------------------------------------------
# classify_source_tier
# ---------------------------------------------------------------------------

class TestClassifySourceTier:
    def test_api_source_type_is_primary(self):
        assert classify_source_tier(source_type="api", author_role="", content="") == "primary"

    def test_system_source_type_is_primary(self):
        assert classify_source_tier(source_type="system", author_role="", content="") == "primary"

    def test_internal_source_type_is_primary(self):
        assert classify_source_tier(source_type="internal", author_role="", content="") == "primary"

    def test_webhook_source_type_is_primary(self):
        assert classify_source_tier(source_type="webhook", author_role="", content="") == "primary"

    def test_document_source_type_is_secondary(self):
        assert classify_source_tier(source_type="document", author_role="", content="") == "secondary"

    def test_import_source_type_is_secondary(self):
        assert classify_source_tier(source_type="import", author_role="", content="") == "secondary"

    def test_integration_source_type_is_secondary(self):
        assert classify_source_tier(source_type="integration", author_role="", content="") == "secondary"

    def test_scrape_source_type_is_tertiary(self):
        assert classify_source_tier(source_type="scrape", author_role="", content="") == "tertiary"

    def test_aggregated_source_type_is_tertiary(self):
        assert classify_source_tier(source_type="aggregated", author_role="", content="") == "tertiary"

    def test_engineer_author_role_is_primary(self):
        assert classify_source_tier(source_type="", author_role="engineer", content="") == "primary"

    def test_analyst_author_role_is_primary(self):
        assert classify_source_tier(source_type="", author_role="analyst", content="") == "primary"

    def test_reporter_author_role_is_secondary(self):
        assert classify_source_tier(source_type="", author_role="reporter", content="") == "secondary"

    def test_anonymous_author_role_is_tertiary(self):
        assert classify_source_tier(source_type="", author_role="anonymous", content="") == "tertiary"

    def test_unknown_author_role_is_tertiary(self):
        assert classify_source_tier(source_type="", author_role="unknown", content="") == "tertiary"

    def test_no_signals_is_unknown(self):
        assert classify_source_tier(source_type="", author_role="", content="a short note") == "unknown"

    def test_hearsay_content_is_tertiary(self):
        result = classify_source_tier(
            source_type="",
            author_role="",
            content="I heard the project was cancelled",
        )
        assert result == "tertiary"

    def test_reportedly_hearsay(self):
        result = classify_source_tier(
            source_type="",
            author_role="",
            content="Reportedly the deal closed yesterday",
        )
        assert result == "tertiary"

    def test_tertiary_author_overrides_secondary_source_type(self):
        # document source_type but anonymous author → tertiary
        result = classify_source_tier(source_type="document", author_role="anonymous", content="")
        assert result == "tertiary"

    def test_source_type_case_insensitive(self):
        assert classify_source_tier(source_type="API", author_role="", content="") == "primary"

    def test_author_role_case_insensitive(self):
        assert classify_source_tier(source_type="", author_role="ENGINEER", content="") == "primary"


# ---------------------------------------------------------------------------
# count_citation_depth
# ---------------------------------------------------------------------------

class TestCountCitationDepth:
    def test_empty_content_is_zero(self):
        assert count_citation_depth("") == 0

    def test_no_citations_is_zero(self):
        assert count_citation_depth("the sprint was completed on time") == 0

    def test_url_counts(self):
        assert count_citation_depth("see https://example.com for details") >= 1

    def test_according_to_counts(self):
        assert count_citation_depth("According to the report, revenue rose.") >= 1

    def test_source_colon_counts(self):
        assert count_citation_depth("source: internal wiki page") >= 1

    def test_ref_colon_counts(self):
        assert count_citation_depth("ref: ticket-1234") >= 1

    def test_based_on_counts(self):
        assert count_citation_depth("Based on the analysis, the system is healthy.") >= 1

    def test_multiple_citations_accumulate(self):
        content = (
            "According to the report, see https://example.com. "
            "source: wiki. based on our research."
        )
        assert count_citation_depth(content) >= 3

    def test_capped_at_10(self):
        # Construct text with many citation signals
        content = " ".join(["according to x"] * 20)
        assert count_citation_depth(content) == 10

    def test_none_treated_as_empty(self):
        assert count_citation_depth(None) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# count_corroboration
# ---------------------------------------------------------------------------

class TestCountCorroboration:
    def test_empty_list_is_zero(self):
        assert count_corroboration([]) == 0

    def test_none_is_zero(self):
        assert count_corroboration(None) == 0  # type: ignore[arg-type]

    def test_single_entry(self):
        assert count_corroboration([{"id": "x"}]) == 1

    def test_multiple_entries(self):
        assert count_corroboration([{"id": "a"}, {"id": "b"}, {"id": "c"}]) == 3

    def test_non_dict_entries_excluded(self):
        assert count_corroboration([{"id": "a"}, "not_a_dict"]) == 1


# ---------------------------------------------------------------------------
# count_conflict_exposure
# ---------------------------------------------------------------------------

class TestCountConflictExposure:
    def test_no_entities_returns_zero(self):
        total, high = count_conflict_exposure(entities={}, conflicts=[_make_conflict()])
        assert total == 0 and high == 0

    def test_no_conflicts_returns_zero(self):
        total, high = count_conflict_exposure(entities={"acme": "corp"}, conflicts=[])
        assert total == 0 and high == 0

    def test_matching_entity_counted(self):
        total, high = count_conflict_exposure(
            entities={"acme": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="high")],
        )
        assert total == 1 and high == 1

    def test_non_matching_entity_not_counted(self):
        total, high = count_conflict_exposure(
            entities={"beta": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="high")],
        )
        assert total == 0 and high == 0

    def test_medium_severity_counted_in_total_not_high(self):
        total, high = count_conflict_exposure(
            entities={"acme": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="medium")],
        )
        assert total == 1 and high == 0

    def test_multiple_matching_entities(self):
        total, high = count_conflict_exposure(
            entities={"acme": "corp", "project_x": "init"},
            conflicts=[
                _make_conflict(entity="acme", severity="high"),
                _make_conflict(entity="project_x", severity="low"),
            ],
        )
        assert total == 2 and high == 1


# ---------------------------------------------------------------------------
# build_credibility_flags
# ---------------------------------------------------------------------------

class TestBuildCredibilityFlags:
    def test_no_source_info_flag(self):
        flags = build_credibility_flags(
            source_type="",
            author_role="",
            source_tier="unknown",
            citation_depth=0,
            content="a short note",
            entities={},
            conflicts=[],
        )
        assert "no_source_info" in flags

    def test_low_citation_depth_flag_for_long_content(self):
        flags = build_credibility_flags(
            source_type="api",
            author_role="engineer",
            source_tier="primary",
            citation_depth=0,
            content="x " * 120,  # > 200 chars
            entities={},
            conflicts=[],
        )
        assert "low_citation_depth" in flags

    def test_no_low_citation_flag_for_short_content(self):
        flags = build_credibility_flags(
            source_type="api",
            author_role="engineer",
            source_tier="primary",
            citation_depth=0,
            content="short",
            entities={},
            conflicts=[],
        )
        assert "low_citation_depth" not in flags

    def test_tertiary_source_flag(self):
        flags = build_credibility_flags(
            source_type="",
            author_role="",
            source_tier="tertiary",
            citation_depth=0,
            content="",
            entities={},
            conflicts=[],
        )
        assert "tertiary_source" in flags

    def test_unverified_claim_flag(self):
        flags = build_credibility_flags(
            source_type="",
            author_role="",
            source_tier="unknown",
            citation_depth=0,
            content="the deal is done",
            entities={},
            conflicts=[],
        )
        assert "unverified_claim" in flags

    def test_hearsay_content_flag(self):
        flags = build_credibility_flags(
            source_type="",
            author_role="",
            source_tier="tertiary",
            citation_depth=0,
            content="I heard the release was delayed",
            entities={},
            conflicts=[],
        )
        assert "hearsay_content" in flags

    def test_conflicted_entity_flag(self):
        flags = build_credibility_flags(
            source_type="api",
            author_role="engineer",
            source_tier="primary",
            citation_depth=2,
            content="acme is healthy",
            entities={"acme": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="high")],
        )
        assert "conflicted_entity:acme" in flags

    def test_no_conflicted_entity_for_medium_severity(self):
        flags = build_credibility_flags(
            source_type="api",
            author_role="engineer",
            source_tier="primary",
            citation_depth=2,
            content="acme is healthy",
            entities={"acme": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="medium")],
        )
        assert not any("conflicted_entity" in f for f in flags)

    def test_clean_memory_has_no_flags(self):
        flags = build_credibility_flags(
            source_type="api",
            author_role="engineer",
            source_tier="primary",
            citation_depth=3,
            content="x " * 20,  # < 200 chars
            entities={},
            conflicts=[],
        )
        assert flags == []


# ---------------------------------------------------------------------------
# compute_credibility_score
# ---------------------------------------------------------------------------

class TestComputeCredibilityScore:
    def test_primary_no_bonus_no_penalty(self):
        score = compute_credibility_score(
            source_tier="primary",
            citation_depth=0,
            corroboration_count=0,
            high_severity_conflict_count=0,
        )
        assert score == pytest.approx(0.80, abs=0.01)

    def test_secondary_base(self):
        score = compute_credibility_score(
            source_tier="secondary",
            citation_depth=0,
            corroboration_count=0,
            high_severity_conflict_count=0,
        )
        assert score == pytest.approx(0.60, abs=0.01)

    def test_tertiary_base(self):
        score = compute_credibility_score(
            source_tier="tertiary",
            citation_depth=0,
            corroboration_count=0,
            high_severity_conflict_count=0,
        )
        assert score == pytest.approx(0.35, abs=0.01)

    def test_unknown_base(self):
        score = compute_credibility_score(
            source_tier="unknown",
            citation_depth=0,
            corroboration_count=0,
            high_severity_conflict_count=0,
        )
        assert score == pytest.approx(0.45, abs=0.01)

    def test_citation_bonus_adds(self):
        score_0 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        score_5 = compute_credibility_score(
            source_tier="primary", citation_depth=5,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        assert score_5 > score_0

    def test_citation_bonus_capped_at_0_10(self):
        score_10 = compute_credibility_score(
            source_tier="primary", citation_depth=10,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        score_100 = compute_credibility_score(
            source_tier="primary", citation_depth=100,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        assert score_10 == score_100

    def test_corroboration_bonus_adds(self):
        score_0 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        score_4 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=4, high_severity_conflict_count=0,
        )
        assert score_4 > score_0

    def test_conflict_penalty_subtracts(self):
        score_0 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        score_3 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=3,
        )
        assert score_3 < score_0

    def test_conflict_penalty_capped_at_3(self):
        score_3 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=3,
        )
        score_99 = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=99,
        )
        assert score_3 == score_99

    def test_score_clamped_above_0_05(self):
        # Tertiary base (0.35) minus max penalty (0.15) = 0.20, well above 0.05
        # Push it lower with repeated calls — check floor
        score = compute_credibility_score(
            source_tier="tertiary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=3,
        )
        assert score >= 0.05

    def test_score_clamped_below_0_95(self):
        score = compute_credibility_score(
            source_tier="primary", citation_depth=10,
            corroboration_count=10, high_severity_conflict_count=0,
        )
        assert score <= 0.95

    def test_returns_float(self):
        score = compute_credibility_score(
            source_tier="primary", citation_depth=0,
            corroboration_count=0, high_severity_conflict_count=0,
        )
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# compute_agent_confidence
# ---------------------------------------------------------------------------

class TestComputeAgentConfidence:
    def test_source_and_corroboration_highest(self):
        c = compute_agent_confidence(source_type="api", author_role="engineer", corroboration_count=3)
        assert c == pytest.approx(0.82, abs=0.01)

    def test_source_only(self):
        c = compute_agent_confidence(source_type="api", author_role="", corroboration_count=0)
        assert c == pytest.approx(0.75, abs=0.01)

    def test_author_role_only(self):
        c = compute_agent_confidence(source_type="", author_role="engineer", corroboration_count=0)
        assert c == pytest.approx(0.75, abs=0.01)

    def test_corroboration_only(self):
        c = compute_agent_confidence(source_type="", author_role="", corroboration_count=1)
        assert c == pytest.approx(0.65, abs=0.01)

    def test_no_signals_lowest(self):
        c = compute_agent_confidence(source_type="", author_role="", corroboration_count=0)
        assert c == pytest.approx(0.55, abs=0.01)


# ---------------------------------------------------------------------------
# CredibilityAgent.validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict) -> AgentResult:
        return AgentResult(
            agent_name="CredibilityAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.75,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=_NOW,
            finished_at=_NOW,
        )

    def test_valid_outputs_passes(self):
        agent = CredibilityAgent()
        result = self._make_result({
            "credibility_score": 0.80,
            "source_tier": "primary",
            "corroboration_count": 2,
            "credibility_flags": [],
        })
        agent.validate_outputs(result)  # should not raise

    def test_invalid_score_type_raises(self):
        agent = CredibilityAgent()
        result = self._make_result({
            "credibility_score": "high",
            "source_tier": "primary",
            "corroboration_count": 0,
            "credibility_flags": [],
        })
        with pytest.raises(ValueError, match="credibility_score"):
            agent.validate_outputs(result)

    def test_score_out_of_range_raises(self):
        agent = CredibilityAgent()
        result = self._make_result({
            "credibility_score": 1.5,
            "source_tier": "primary",
            "corroboration_count": 0,
            "credibility_flags": [],
        })
        with pytest.raises(ValueError, match="credibility_score"):
            agent.validate_outputs(result)

    def test_invalid_source_tier_raises(self):
        agent = CredibilityAgent()
        result = self._make_result({
            "credibility_score": 0.75,
            "source_tier": "gold",
            "corroboration_count": 0,
            "credibility_flags": [],
        })
        with pytest.raises(ValueError, match="source_tier"):
            agent.validate_outputs(result)

    def test_non_int_corroboration_count_raises(self):
        agent = CredibilityAgent()
        result = self._make_result({
            "credibility_score": 0.75,
            "source_tier": "primary",
            "corroboration_count": 2.5,
            "credibility_flags": [],
        })
        with pytest.raises(ValueError, match="corroboration_count"):
            agent.validate_outputs(result)

    def test_non_list_flags_raises(self):
        agent = CredibilityAgent()
        result = self._make_result({
            "credibility_score": 0.75,
            "source_tier": "primary",
            "corroboration_count": 0,
            "credibility_flags": "none",
        })
        with pytest.raises(ValueError, match="credibility_flags"):
            agent.validate_outputs(result)

    def test_skipped_for_non_success(self):
        agent = CredibilityAgent()
        result = AgentResult(
            agent_name="CredibilityAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=_NOW,
            finished_at=_NOW,
        )
        agent.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# CredibilityAgent heuristic path (run())
# ---------------------------------------------------------------------------

class TestCredibilityAgentHeuristic:
    @pytest.mark.asyncio
    async def test_primary_source_type(self):
        agent = CredibilityAgent()
        ctx = _make_context(source_type="api", author_role="engineer")
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.status == "success"
        assert result.outputs["source_tier"] == "primary"
        assert result.outputs["credibility_score"] >= 0.75

    @pytest.mark.asyncio
    async def test_secondary_source_type(self):
        agent = CredibilityAgent()
        ctx = _make_context(source_type="document", author_role="reporter")
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.outputs["source_tier"] == "secondary"

    @pytest.mark.asyncio
    async def test_tertiary_hearsay_content(self):
        agent = CredibilityAgent()
        ctx = _make_context(
            content="I heard the project was cancelled last week",
            source_type="",
            author_role="",
        )
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.outputs["source_tier"] == "tertiary"
        assert "hearsay_content" in result.outputs["credibility_flags"]

    @pytest.mark.asyncio
    async def test_unknown_source_no_info(self):
        agent = CredibilityAgent()
        ctx = _make_context(source_type="", author_role="", content="project update")
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.outputs["source_tier"] == "unknown"
        assert "no_source_info" in result.outputs["credibility_flags"]

    @pytest.mark.asyncio
    async def test_corroboration_increases_score(self):
        agent = CredibilityAgent()
        ctx_low = _make_context(corroborating_memories=[])
        ctx_high = _make_context(corroborating_memories=_make_corroborating(count=4))
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            r_low = await agent.run("m1", ctx_low)
            r_high = await agent.run("m2", ctx_high)
        assert r_high.outputs["credibility_score"] > r_low.outputs["credibility_score"]

    @pytest.mark.asyncio
    async def test_high_severity_conflict_reduces_score(self):
        agent = CredibilityAgent()
        ctx_clean = _make_context(conflicts=[])
        ctx_conflict = _make_context(
            entities={"acme": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="high")],
        )
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            r_clean = await agent.run("m1", ctx_clean)
            r_conflict = await agent.run("m2", ctx_conflict)
        assert r_conflict.outputs["credibility_score"] < r_clean.outputs["credibility_score"]

    @pytest.mark.asyncio
    async def test_conflicted_entity_flag_set(self):
        agent = CredibilityAgent()
        ctx = _make_context(
            entities={"acme": "corp"},
            conflicts=[_make_conflict(entity="acme", severity="high")],
        )
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert any("conflicted_entity:acme" in f for f in result.outputs["credibility_flags"])

    @pytest.mark.asyncio
    async def test_citation_in_content_raises_score(self):
        agent = CredibilityAgent()
        ctx_bare = _make_context(
            content="x " * 20,  # < 200 chars, no citations
            source_type="secondary",
            author_role="reporter",
        )
        ctx_cited = _make_context(
            content="According to the report https://example.com, the sprint is done.",
            source_type="secondary",
            author_role="reporter",
        )
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            r_bare = await agent.run("m1", ctx_bare)
            r_cited = await agent.run("m2", ctx_cited)
        assert r_cited.outputs["credibility_score"] > r_bare.outputs["credibility_score"]

    @pytest.mark.asyncio
    async def test_empty_content_uses_heuristic(self):
        agent = CredibilityAgent()
        ctx = _make_context(content="", source_type="api", author_role="engineer")
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "llm"
            result = await agent.run("m1", ctx)
        # Empty content must fall back to heuristic even in llm mode
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_rationale_is_heuristic(self):
        agent = CredibilityAgent()
        ctx = _make_context()
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_corroboration_count_in_outputs(self):
        agent = CredibilityAgent()
        ctx = _make_context(corroborating_memories=_make_corroborating(count=3))
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.outputs["corroboration_count"] == 3

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = CredibilityAgent()
        ctx = _make_context(job_id="job-xyz")
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.trace_id == "job-xyz"

    @pytest.mark.asyncio
    async def test_no_trace_id_when_absent(self):
        agent = CredibilityAgent()
        ctx = _make_context(job_id=None)
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_memory_id_propagated(self):
        agent = CredibilityAgent()
        ctx = _make_context()
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("mem-999", ctx)
        assert result.memory_id == "mem-999"

    @pytest.mark.asyncio
    async def test_agent_name_and_version(self):
        agent = CredibilityAgent()
        ctx = _make_context()
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert result.agent_name == "CredibilityAgent"
        assert result.agent_version == "v1"

    @pytest.mark.asyncio
    async def test_resolved_entities_field_used(self):
        agent = CredibilityAgent()
        ctx = {
            "memory": {
                "content": "test",
                "source_type": "api",
                "author_role": "engineer",
                "enrichment": {
                    "resolved_entities": {"acme": "corp"},
                    "conflicts": [_make_conflict(entity="acme", severity="high")],
                    "corroborating_memories": [],
                },
            },
        }
        with patch("app.agents.credibility_agent.settings") as mock_settings:
            mock_settings.CREDIBILITY_STRATEGY = "heuristic"
            result = await agent.run("m1", ctx)
        assert any("conflicted_entity:acme" in f for f in result.outputs["credibility_flags"])


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

class TestCredibilityAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_path_uses_response(self):
        agent = CredibilityAgent()
        ctx = _make_context(content="deployment complete. see https://dash.example.com")

        llm_response = {
            "credibility_score": 0.88,
            "source_tier": "primary",
            "corroboration_count": 2,
            "credibility_flags": [],
            "confidence": 0.82,
            "rationale": "llm: clear primary source with citation",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.credibility_agent.settings") as mock_settings, \
             patch("app.agents.credibility_agent.create_ollama_client", return_value=mock_client):
            mock_settings.CREDIBILITY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("m1", ctx)

        assert result.outputs["source_tier"] == "primary"
        assert result.outputs["credibility_score"] == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_llm_invalid_tier_falls_back_to_heuristic(self):
        agent = CredibilityAgent()
        ctx = _make_context(content="some text here", source_type="api", author_role="engineer")

        llm_response = {
            "credibility_score": 0.88,
            "source_tier": "gold",  # invalid
            "corroboration_count": 0,
            "credibility_flags": [],
            "confidence": 0.80,
            "rationale": "bad tier",
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.credibility_agent.settings") as mock_settings, \
             patch("app.agents.credibility_agent.create_ollama_client", return_value=mock_client):
            mock_settings.CREDIBILITY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("m1", ctx)

        assert result.outputs["rationale"] == "heuristic"
        assert result.outputs["source_tier"] in {"primary", "secondary", "tertiary", "unknown"}

    @pytest.mark.asyncio
    async def test_llm_missing_flags_field_falls_back(self):
        agent = CredibilityAgent()
        ctx = _make_context(content="some content here", source_type="api", author_role="engineer")

        llm_response = {
            "credibility_score": 0.75,
            "source_tier": "primary",
            "corroboration_count": 1,
            # missing credibility_flags
            "confidence": 0.80,
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.credibility_agent.settings") as mock_settings, \
             patch("app.agents.credibility_agent.create_ollama_client", return_value=mock_client):
            mock_settings.CREDIBILITY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("m1", ctx)

        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_non_int_corroboration_count_falls_back(self):
        agent = CredibilityAgent()
        ctx = _make_context(content="some text here", source_type="api", author_role="engineer")

        llm_response = {
            "credibility_score": 0.75,
            "source_tier": "primary",
            "corroboration_count": "two",  # invalid
            "credibility_flags": [],
            "confidence": 0.80,
        }

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)

        with patch("app.agents.credibility_agent.settings") as mock_settings, \
             patch("app.agents.credibility_agent.create_ollama_client", return_value=mock_client):
            mock_settings.CREDIBILITY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("m1", ctx)

        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_none_response_falls_back(self):
        agent = CredibilityAgent()
        ctx = _make_context(content="some content here", source_type="api", author_role="engineer")

        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=None)

        with patch("app.agents.credibility_agent.settings") as mock_settings, \
             patch("app.agents.credibility_agent.create_ollama_client", return_value=mock_client):
            mock_settings.CREDIBILITY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("m1", ctx)

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_by_name(self):
        agent = get_agent("credibility_agent")
        assert isinstance(agent, CredibilityAgent)

    def test_get_by_short_name(self):
        agent = get_agent("credibility")
        assert isinstance(agent, CredibilityAgent)

    def test_get_by_class_name(self):
        agent = get_agent("credibilityagent")
        assert isinstance(agent, CredibilityAgent)

    def test_dependencies(self):
        agent = CredibilityAgent()
        deps = agent.dependencies()
        assert "EntityResolutionAgent" in deps
        assert "ConflictDetectionAgent" in deps
