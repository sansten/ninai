"""Tests for Phase 20 — PlaybookAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.playbook_agent import (
    PlaybookAgent,
    _MATCH_THRESHOLD,
    _extract_entity_names,
    extract_action_tokens,
    score_step_match,
    find_best_step,
    match_against_playbooks,
    detect_new_playbook_candidate,
    compute_agent_confidence,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


_NOW = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_playbook(
    *,
    id: str = "pb-001",
    name: str = "Deploy Patch",
    domain: str = "engineering",
    steps: list[str] | None = None,
    success_rate: float = 0.8,
    occurrence_count: int = 5,
) -> dict:
    if steps is None:
        steps = ["analyze bug report", "create fix branch", "deploy patch", "verify deployment"]
    return {
        "id": id,
        "name": name,
        "domain": domain,
        "steps": steps,
        "success_rate": success_rate,
        "occurrence_count": occurrence_count,
    }


def _make_context(
    *,
    content: str = "deploy patch and verify deployment",
    domain: str = "engineering",
    entities: dict | None = None,
    episode_role: str = "body",
    episode_key: str = "abc123",
    playbook_candidates: list | None = None,
    job_id: str | None = None,
) -> dict:
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "domain": domain,
                "entities": entities if entities is not None else {"acme": "corp"},
                "episode_role": episode_role,
                "episode_key": episode_key,
                "playbook_candidates": playbook_candidates if playbook_candidates is not None else [],
            },
        },
        "runtime": {"job_id": job_id} if job_id else {},
    }


# ---------------------------------------------------------------------------
# extract_action_tokens
# ---------------------------------------------------------------------------

class TestExtractActionTokens:
    def test_empty_string(self):
        assert extract_action_tokens("") == set()

    def test_none_equivalent(self):
        assert extract_action_tokens("") == set()

    def test_short_words_excluded(self):
        tokens = extract_action_tokens("do it go")
        assert "do" not in tokens
        assert "it" not in tokens
        assert "go" not in tokens

    def test_stop_words_excluded(self):
        tokens = extract_action_tokens("the and for are was this that with from")
        assert tokens == set()

    def test_normal_content(self):
        tokens = extract_action_tokens("deploy patch and verify deployment")
        assert "deploy" in tokens
        assert "patch" in tokens
        assert "verify" in tokens
        assert "deployment" in tokens
        assert "and" not in tokens

    def test_case_insensitive(self):
        tokens = extract_action_tokens("Deploy PATCH Verify")
        assert "deploy" in tokens
        assert "patch" in tokens
        assert "verify" in tokens

    def test_three_char_words_included(self):
        tokens = extract_action_tokens("run fix bug")
        assert "run" in tokens
        assert "fix" in tokens
        assert "bug" in tokens

    def test_two_char_words_excluded(self):
        tokens = extract_action_tokens("no go on")
        # "on" is 2 chars — excluded; "no" and "go" are 2 chars — excluded
        assert tokens == set()

    def test_mixed_content(self):
        tokens = extract_action_tokens("analyze the bug report and create fix")
        assert "analyze" in tokens
        assert "report" in tokens
        assert "create" in tokens
        assert "the" not in tokens
        assert "and" not in tokens

    def test_numbers_not_extracted(self):
        tokens = extract_action_tokens("version 2 release")
        assert "version" in tokens
        assert "release" in tokens
        # numbers won't match [a-z]{3,}


# ---------------------------------------------------------------------------
# score_step_match
# ---------------------------------------------------------------------------

class TestScoreStepMatch:
    def test_empty_step(self):
        assert score_step_match("", {"deploy", "patch"}) == 0.0

    def test_empty_tokens(self):
        assert score_step_match("deploy patch", set()) == 0.0

    def test_full_match(self):
        # step = "deploy patch" → tokens {"deploy","patch"}; content has both
        score = score_step_match("deploy patch", {"deploy", "patch", "verify"})
        assert score == 1.0

    def test_partial_match(self):
        # step has 2 tokens, content has 1 → 0.5
        score = score_step_match("deploy patch", {"deploy", "other"})
        assert score == 0.5

    def test_no_overlap(self):
        score = score_step_match("analyze report", {"deploy", "patch"})
        assert score == 0.0

    def test_stop_words_in_step_ignored(self):
        # "the" is a stop word, only "deployment" counts
        score = score_step_match("the deployment", {"deployment", "patch"})
        assert score == 1.0

    def test_step_all_stop_words(self):
        score = score_step_match("the and for", {"the", "and", "for"})
        assert score == 0.0

    def test_score_is_rounded(self):
        # step has 3 tokens (analyze, bug, report), content has 2
        score = score_step_match("analyze bug report", {"analyze", "bug", "deploy"})
        assert score == round(2 / 3, 4)


# ---------------------------------------------------------------------------
# find_best_step
# ---------------------------------------------------------------------------

class TestFindBestStep:
    def test_empty_steps(self):
        pos, score = find_best_step([], {"deploy", "patch"})
        assert pos == 0
        assert score == 0.0

    def test_single_step_match(self):
        pos, score = find_best_step(["deploy patch"], {"deploy", "patch"})
        assert pos == 1
        assert score == 1.0

    def test_multiple_steps_best_returned(self):
        steps = ["analyze bug", "deploy patch verify", "write report"]
        tokens = {"deploy", "patch", "verify"}
        pos, score = find_best_step(steps, tokens)
        assert pos == 2  # "deploy patch verify" matches all 3
        assert score == 1.0

    def test_no_matching_step(self):
        steps = ["analyze report", "write document"]
        tokens = {"deploy", "patch"}
        pos, score = find_best_step(steps, tokens)
        assert pos == 0
        assert score == 0.0

    def test_first_best_wins_on_tie(self):
        # Both steps have same single token match
        steps = ["deploy fix", "deploy release"]
        tokens = {"deploy"}
        pos, score = find_best_step(steps, tokens)
        assert pos == 1  # first one wins


# ---------------------------------------------------------------------------
# _extract_entity_names
# ---------------------------------------------------------------------------

class TestExtractEntityNames:
    def test_empty_dict(self):
        assert _extract_entity_names({}) == set()

    def test_empty_list(self):
        assert _extract_entity_names([]) == set()

    def test_none(self):
        assert _extract_entity_names(None) == set()

    def test_scalar_string(self):
        result = _extract_entity_names("acme")
        assert "acme" in result

    def test_dict_keys(self):
        result = _extract_entity_names({"Acme": "corp", "BobCo": "partner"})
        assert "acme" in result
        assert "bobco" in result

    def test_dict_with_list_of_dicts(self):
        result = _extract_entity_names({"org": [{"canonical": "AcmeCorp"}]})
        assert "acmecorp" in result

    def test_list_of_dicts_canonical(self):
        result = _extract_entity_names([{"canonical": "Acme", "original": "acme inc"}])
        assert "acme" in result
        assert "acme inc" in result

    def test_list_of_strings(self):
        result = _extract_entity_names(["acme", "bobco"])
        assert "acme" in result
        assert "bobco" in result

    def test_empty_strings_filtered(self):
        result = _extract_entity_names(["", "  "])
        assert result == set()


# ---------------------------------------------------------------------------
# match_against_playbooks
# ---------------------------------------------------------------------------

class TestMatchAgainstPlaybooks:
    def test_no_candidates_returns_none(self):
        mid, conf, step = match_against_playbooks(
            content="deploy patch",
            domain="engineering",
            entities={},
            playbook_candidates=[],
        )
        assert mid is None
        assert conf == 0.0
        assert step is None

    def test_candidate_with_no_steps_skipped(self):
        mid, conf, step = match_against_playbooks(
            content="deploy patch",
            domain="engineering",
            entities={},
            playbook_candidates=[{"id": "pb-1", "domain": "engineering", "steps": []}],
        )
        assert mid is None

    def test_non_dict_candidate_skipped(self):
        mid, conf, step = match_against_playbooks(
            content="deploy patch",
            domain="engineering",
            entities={},
            playbook_candidates=["not a dict"],
        )
        assert mid is None

    def test_good_match_same_domain(self):
        pb = _make_playbook(domain="engineering", steps=["deploy patch", "verify deployment"])
        mid, conf, step = match_against_playbooks(
            content="deploy patch and verify deployment",
            domain="engineering",
            entities={},
            playbook_candidates=[pb],
        )
        assert mid == "pb-001"
        assert conf >= _MATCH_THRESHOLD
        assert step is not None

    def test_good_match_different_domain_lower_score(self):
        pb_same = _make_playbook(id="pb-same", domain="engineering")
        pb_diff = _make_playbook(id="pb-diff", domain="sales")
        mid, conf, step = match_against_playbooks(
            content="deploy patch and verify deployment",
            domain="engineering",
            entities={},
            playbook_candidates=[pb_diff, pb_same],
        )
        # Should prefer same domain
        assert mid == "pb-same"

    def test_below_threshold_returns_none(self):
        pb = _make_playbook(steps=["completely unrelated action step here"])
        mid, conf, step = match_against_playbooks(
            content="deploy patch",
            domain="engineering",
            entities={},
            playbook_candidates=[pb],
        )
        # Very low overlap → below threshold
        assert mid is None
        assert conf == 0.0

    def test_multiple_candidates_best_returned(self):
        pb_weak = _make_playbook(id="pb-weak", steps=["analyze something"])
        pb_strong = _make_playbook(id="pb-strong", steps=["deploy patch verify deployment"])
        mid, conf, step = match_against_playbooks(
            content="deploy patch verify deployment",
            domain="engineering",
            entities={},
            playbook_candidates=[pb_weak, pb_strong],
        )
        assert mid == "pb-strong"

    def test_step_position_correct(self):
        pb = _make_playbook(steps=["analyze bug", "deploy patch", "verify deployment"])
        _, _, step = match_against_playbooks(
            content="deploy patch fix",
            domain="engineering",
            entities={},
            playbook_candidates=[pb],
        )
        # "deploy patch" is step 2
        assert step == 2

    def test_entity_tokens_used_in_matching(self):
        pb = _make_playbook(steps=["acme contract review"])
        mid, conf, step = match_against_playbooks(
            content="routine update",
            domain="sales",
            entities={"acme": "org", "contract": "doc"},
            playbook_candidates=[pb],
        )
        # Entities provide "acme" and "contract" → overlap with step tokens
        assert mid == "pb-001"

    def test_high_occurrence_boosts_reliability(self):
        pb_low = _make_playbook(id="pb-low", steps=["deploy patch"], occurrence_count=1, success_rate=0.5)
        pb_high = _make_playbook(id="pb-high", steps=["deploy patch"], occurrence_count=10, success_rate=1.0)
        _, conf_low, _ = match_against_playbooks(
            content="deploy patch",
            domain="engineering",
            entities={},
            playbook_candidates=[pb_low],
        )
        _, conf_high, _ = match_against_playbooks(
            content="deploy patch",
            domain="engineering",
            entities={},
            playbook_candidates=[pb_high],
        )
        assert conf_high >= conf_low


# ---------------------------------------------------------------------------
# detect_new_playbook_candidate
# ---------------------------------------------------------------------------

class TestDetectNewPlaybookCandidate:
    def test_opener_role_false(self):
        assert detect_new_playbook_candidate(episode_role="opener", matched_confidence=0.0) is False

    def test_body_role_false(self):
        assert detect_new_playbook_candidate(episode_role="body", matched_confidence=0.0) is False

    def test_closer_no_match_true(self):
        assert detect_new_playbook_candidate(episode_role="closer", matched_confidence=0.0) is True

    def test_closer_with_match_false(self):
        assert detect_new_playbook_candidate(episode_role="closer", matched_confidence=0.60) is False

    def test_climax_no_match_true(self):
        assert detect_new_playbook_candidate(episode_role="climax", matched_confidence=0.0) is True

    def test_climax_with_match_false(self):
        assert detect_new_playbook_candidate(episode_role="climax", matched_confidence=0.80) is False

    def test_closer_at_threshold_boundary(self):
        # Exactly at threshold → below threshold (< not <=) → True
        assert detect_new_playbook_candidate(
            episode_role="closer",
            matched_confidence=_MATCH_THRESHOLD - 0.01,
        ) is True

    def test_closer_above_threshold_false(self):
        assert detect_new_playbook_candidate(
            episode_role="closer",
            matched_confidence=_MATCH_THRESHOLD + 0.01,
        ) is False

    def test_unknown_role_false(self):
        assert detect_new_playbook_candidate(episode_role="unknown_role", matched_confidence=0.0) is False


# ---------------------------------------------------------------------------
# compute_agent_confidence
# ---------------------------------------------------------------------------

class TestComputeAgentConfidence:
    def test_high_match_confidence(self):
        c = compute_agent_confidence(
            matched_confidence=0.80,
            is_new_candidate=False,
            playbook_candidates_count=3,
        )
        assert c == 0.82

    def test_medium_match_confidence(self):
        c = compute_agent_confidence(
            matched_confidence=0.55,
            is_new_candidate=False,
            playbook_candidates_count=3,
        )
        assert c == 0.75

    def test_low_match_at_threshold(self):
        c = compute_agent_confidence(
            matched_confidence=_MATCH_THRESHOLD,
            is_new_candidate=False,
            playbook_candidates_count=3,
        )
        assert c == 0.68

    def test_no_match_new_candidate(self):
        c = compute_agent_confidence(
            matched_confidence=0.0,
            is_new_candidate=True,
            playbook_candidates_count=2,
        )
        assert c == 0.62

    def test_no_match_has_candidates(self):
        c = compute_agent_confidence(
            matched_confidence=0.0,
            is_new_candidate=False,
            playbook_candidates_count=2,
        )
        assert c == 0.58

    def test_no_match_no_candidates(self):
        c = compute_agent_confidence(
            matched_confidence=0.0,
            is_new_candidate=False,
            playbook_candidates_count=0,
        )
        assert c == 0.50

    def test_boundary_70(self):
        c = compute_agent_confidence(
            matched_confidence=0.70,
            is_new_candidate=False,
            playbook_candidates_count=1,
        )
        assert c == 0.82

    def test_boundary_50(self):
        c = compute_agent_confidence(
            matched_confidence=0.50,
            is_new_candidate=False,
            playbook_candidates_count=1,
        )
        assert c == 0.75


# ---------------------------------------------------------------------------
# PlaybookAgent.validate_outputs
# ---------------------------------------------------------------------------

class TestValidateOutputs:
    def _make_result(self, outputs: dict) -> AgentResult:
        return AgentResult(
            agent_name="PlaybookAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.70,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=_NOW,
            finished_at=_NOW,
            trace_id=None,
        )

    def test_valid_outputs_no_match(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": None,
            "playbook_confidence": 0.0,
            "is_new_playbook_candidate": False,
            "step_position": None,
            "confidence": 0.50,
            "rationale": "heuristic",
        })
        agent.validate_outputs(result)  # should not raise

    def test_valid_outputs_with_match(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": "pb-001",
            "playbook_confidence": 0.75,
            "is_new_playbook_candidate": False,
            "step_position": 2,
            "confidence": 0.80,
            "rationale": "heuristic",
        })
        agent.validate_outputs(result)

    def test_invalid_playbook_confidence_out_of_range(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": None,
            "playbook_confidence": 1.5,
            "is_new_playbook_candidate": False,
            "step_position": None,
            "confidence": 0.50,
            "rationale": "heuristic",
        })
        with pytest.raises(ValueError, match="playbook_confidence"):
            agent.validate_outputs(result)

    def test_invalid_playbook_confidence_not_number(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": None,
            "playbook_confidence": "high",
            "is_new_playbook_candidate": False,
            "step_position": None,
            "confidence": 0.50,
            "rationale": "heuristic",
        })
        with pytest.raises(ValueError, match="playbook_confidence"):
            agent.validate_outputs(result)

    def test_invalid_is_new_not_bool(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": None,
            "playbook_confidence": 0.0,
            "is_new_playbook_candidate": "yes",
            "step_position": None,
            "confidence": 0.50,
            "rationale": "heuristic",
        })
        with pytest.raises(ValueError, match="is_new_playbook_candidate"):
            agent.validate_outputs(result)

    def test_invalid_step_position_not_int(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": "pb-1",
            "playbook_confidence": 0.60,
            "is_new_playbook_candidate": False,
            "step_position": "2",
            "confidence": 0.70,
            "rationale": "heuristic",
        })
        with pytest.raises(ValueError, match="step_position"):
            agent.validate_outputs(result)

    def test_invalid_matched_id_not_str(self):
        agent = PlaybookAgent()
        result = self._make_result({
            "matched_playbook_id": 123,
            "playbook_confidence": 0.60,
            "is_new_playbook_candidate": False,
            "step_position": 1,
            "confidence": 0.70,
            "rationale": "heuristic",
        })
        with pytest.raises(ValueError, match="matched_playbook_id"):
            agent.validate_outputs(result)

    def test_skip_validation_on_non_success(self):
        agent = PlaybookAgent()
        result = AgentResult(
            agent_name="PlaybookAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=["fail"],
            started_at=_NOW,
            finished_at=_NOW,
            trace_id=None,
        )
        agent.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# PlaybookAgent._heuristic
# ---------------------------------------------------------------------------

class TestPlaybookAgentHeuristic:
    def test_no_candidates(self):
        agent = PlaybookAgent()
        out = agent._heuristic(
            content="deploy patch",
            domain="engineering",
            entities={},
            episode_role="body",
            playbook_candidates=[],
        )
        assert out["matched_playbook_id"] is None
        assert out["playbook_confidence"] == 0.0
        assert out["is_new_playbook_candidate"] is False
        assert out["step_position"] is None
        assert out["rationale"] == "heuristic"

    def test_good_match(self):
        agent = PlaybookAgent()
        pb = _make_playbook(steps=["deploy patch", "verify deployment"])
        out = agent._heuristic(
            content="deploy patch and verify deployment",
            domain="engineering",
            entities={},
            episode_role="body",
            playbook_candidates=[pb],
        )
        assert out["matched_playbook_id"] == "pb-001"
        assert out["playbook_confidence"] >= _MATCH_THRESHOLD
        assert out["is_new_playbook_candidate"] is False
        assert out["step_position"] is not None

    def test_closer_no_match_flags_new_candidate(self):
        agent = PlaybookAgent()
        out = agent._heuristic(
            content="completely unique action sequence never seen before",
            domain="engineering",
            entities={},
            episode_role="closer",
            playbook_candidates=[],
        )
        assert out["is_new_playbook_candidate"] is True
        assert out["matched_playbook_id"] is None

    def test_opener_no_match_not_new_candidate(self):
        agent = PlaybookAgent()
        out = agent._heuristic(
            content="new thing",
            domain="engineering",
            entities={},
            episode_role="opener",
            playbook_candidates=[],
        )
        assert out["is_new_playbook_candidate"] is False

    def test_confidence_present(self):
        agent = PlaybookAgent()
        out = agent._heuristic(
            content="deploy patch",
            domain="engineering",
            entities={},
            episode_role="body",
            playbook_candidates=[],
        )
        assert isinstance(out["confidence"], float)
        assert 0.0 <= out["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# PlaybookAgent.run() — heuristic path
# ---------------------------------------------------------------------------

class TestPlaybookAgentRunHeuristic:
    @pytest.mark.asyncio
    async def test_basic_run_no_candidates(self, monkeypatch):
        import app.agents.playbook_agent as pa_mod
        monkeypatch.setattr(pa_mod.settings, "AGENT_STRATEGY", "heuristic")
        agent = PlaybookAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.agent_name == "PlaybookAgent"
        assert result.outputs["matched_playbook_id"] is None
        assert result.outputs["playbook_confidence"] == 0.0
        assert isinstance(result.outputs["is_new_playbook_candidate"], bool)

    @pytest.mark.asyncio
    async def test_empty_content_uses_heuristic(self):
        agent = PlaybookAgent()
        ctx = _make_context(content="")
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_episode_role_defaults_to_body(self):
        agent = PlaybookAgent()
        ctx = {
            "memory": {
                "content": "some content here",
                "enrichment": {
                    "domain": "engineering",
                    "entities": {},
                    "playbook_candidates": [],
                    # episode_role absent
                },
            }
        }
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_trace_id_populated(self):
        agent = PlaybookAgent()
        ctx = _make_context(job_id="job-xyz")
        result = await agent.run("mem-1", ctx)
        assert result.trace_id == "job-xyz"

    @pytest.mark.asyncio
    async def test_no_trace_id(self):
        agent = PlaybookAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        assert result.trace_id is None

    @pytest.mark.asyncio
    async def test_with_matching_playbook(self):
        agent = PlaybookAgent()
        pb = _make_playbook(steps=["deploy patch", "verify deployment"])
        ctx = _make_context(
            content="deploy patch and verify deployment",
            domain="engineering",
            episode_role="body",
            playbook_candidates=[pb],
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["matched_playbook_id"] == "pb-001"
        assert result.outputs["playbook_confidence"] >= _MATCH_THRESHOLD

    @pytest.mark.asyncio
    async def test_closer_new_candidate(self):
        agent = PlaybookAgent()
        ctx = _make_context(
            content="unprecedented novel procedure",
            episode_role="closer",
            playbook_candidates=[],
        )
        result = await agent.run("mem-1", ctx)
        assert result.outputs["is_new_playbook_candidate"] is True

    @pytest.mark.asyncio
    async def test_resolved_entities_fallback(self):
        agent = PlaybookAgent()
        ctx = {
            "memory": {
                "content": "deploy patch",
                "enrichment": {
                    "domain": "engineering",
                    "resolved_entities": {"acme": "corp"},
                    "episode_role": "body",
                    "playbook_candidates": [],
                },
            }
        }
        result = await agent.run("mem-1", ctx)
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_validate_outputs_called(self):
        agent = PlaybookAgent()
        ctx = _make_context()
        result = await agent.run("mem-1", ctx)
        # validate_outputs would raise if something is wrong
        assert result.outputs["playbook_confidence"] is not None


# ---------------------------------------------------------------------------
# PlaybookAgent.run() — LLM path
# ---------------------------------------------------------------------------

class TestPlaybookAgentRunLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = PlaybookAgent()
        ctx = _make_context(content="deploy patch and verify")
        llm_response = {
            "matched_playbook_id": "pb-llm-001",
            "playbook_confidence": 0.85,
            "is_new_playbook_candidate": False,
            "step_position": 3,
            "confidence": 0.80,
            "rationale": "llm matched step 3 of Deploy Patch playbook",
        }
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value=llm_response)
        with patch("app.agents.playbook_agent.create_ollama_client", return_value=mock_client), \
             patch("app.agents.playbook_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.outputs["matched_playbook_id"] == "pb-llm-001"
        assert result.outputs["playbook_confidence"] == 0.85
        assert result.outputs["step_position"] == 3

    @pytest.mark.asyncio
    async def test_invalid_llm_response_falls_back(self):
        agent = PlaybookAgent()
        ctx = _make_context(content="deploy patch")
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={"invalid": "response"})
        with patch("app.agents.playbook_agent.create_ollama_client", return_value=mock_client), \
             patch("app.agents.playbook_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_missing_is_new_falls_back(self):
        agent = PlaybookAgent()
        ctx = _make_context(content="deploy patch")
        # Valid confidence but missing is_new_playbook_candidate
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(return_value={
            "matched_playbook_id": "pb-1",
            "playbook_confidence": 0.75,
            # is_new_playbook_candidate missing
            "step_position": 1,
            "confidence": 0.70,
        })
        with patch("app.agents.playbook_agent.create_ollama_client", return_value=mock_client), \
             patch("app.agents.playbook_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back(self):
        agent = PlaybookAgent()
        ctx = _make_context(content="deploy patch")
        mock_client = AsyncMock()
        mock_client.complete_json = AsyncMock(side_effect=Exception("LLM timeout"))
        with patch("app.agents.playbook_agent.create_ollama_client", return_value=mock_client), \
             patch("app.agents.playbook_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_STRATEGY = "llm"
            mock_settings.AGENT_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            with pytest.raises(Exception):
                await agent.run("mem-1", ctx)

    @pytest.mark.asyncio
    async def test_heuristic_strategy_bypasses_llm(self):
        agent = PlaybookAgent()
        ctx = _make_context(content="deploy patch")
        with patch("app.agents.playbook_agent.settings") as mock_settings:
            mock_settings.PLAYBOOK_STRATEGY = "heuristic"
            mock_settings.AGENT_STRATEGY = "heuristic"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_MODEL = "llama3.1:8b"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            result = await agent.run("mem-1", ctx)

        assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_get_agent_playbook(self):
        agent = get_agent("playbook")
        assert isinstance(agent, PlaybookAgent)

    def test_get_agent_playbook_agent(self):
        agent = get_agent("playbookagent")
        assert isinstance(agent, PlaybookAgent)

    def test_get_agent_playbook_underscore(self):
        agent = get_agent("playbook_agent")
        assert isinstance(agent, PlaybookAgent)

    def test_get_agent_case_insensitive(self):
        agent = get_agent("PlaybookAgent")
        assert isinstance(agent, PlaybookAgent)

    def test_get_agent_unknown_still_none(self):
        agent = get_agent("unknown_xyz_phase20")
        assert agent is None


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestPlaybookAgentMeta:
    def test_name(self):
        assert PlaybookAgent().name == "PlaybookAgent"

    def test_version(self):
        assert PlaybookAgent().version == "v1"

    def test_dependencies(self):
        deps = PlaybookAgent().dependencies()
        assert "EpisodicGroupingAgent" in deps
        assert "EntityResolutionAgent" in deps
