from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.active_knowledge_seeker_agent import (
    ActiveKnowledgeSeekerAgent,
    _memory_text,
    _pick_top_question,
    _tokenize,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.models.knowledge_gap import KnowledgeGap


def _ctx(
    *,
    goal: str,
    available_memories: list[dict],
    required_entities: list[str],
    confidence_threshold: float = 0.4,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "goal": goal,
                "available_memories": available_memories,
                "required_entities": required_entities,
                "confidence_threshold": confidence_threshold,
            }
        },
        "runtime": {"job_id": "trace-63"},
    }


class TestHelpers:
    def test_tokenize_lowercases(self):
        assert "customer" in _tokenize("Customer")

    def test_memory_text_uses_content_title_tags(self):
        text = _memory_text({"content": "alpha", "title": "beta", "tags": ["gamma"]})
        assert "alpha" in text
        assert "beta" in text
        assert "gamma" in text

    def test_pick_top_question_prefers_critical(self):
        q = _pick_top_question(
            [
                {"priority": "high", "question_to_ask": "Q1"},
                {"priority": "critical", "question_to_ask": "Q2"},
            ]
        )
        assert q == "Q2"

    def test_pick_top_question_falls_back_high(self):
        q = _pick_top_question(
            [
                {"priority": "medium", "question_to_ask": "Q1"},
                {"priority": "high", "question_to_ask": "Q2"},
            ]
        )
        assert q == "Q2"

    def test_pick_top_question_none_for_empty(self):
        assert _pick_top_question([]) is None


class TestHeuristic:
    def test_all_entities_covered(self):
        out = run_heuristic(
            goal="improve retention for acme",
            available_memories=[{"content": "acme retention baseline and churn"}],
            required_entities=["acme", "retention"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"] == []
        assert out["is_sufficient"] is True

    def test_missing_entity_creates_gap(self):
        out = run_heuristic(
            goal="investigate acme incident",
            available_memories=[{"content": "acme dashboard healthy"}],
            required_entities=["acme", "root cause"],
            confidence_threshold=0.4,
        )
        assert len(out["knowledge_gaps"]) == 1
        assert "root cause" in out["knowledge_gaps"][0]["gap_description"].lower()

    def test_missing_entity_question_shape(self):
        out = run_heuristic(
            goal="investigate acme incident",
            available_memories=[{"content": "acme dashboard healthy"}],
            required_entities=["root cause"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"][0]["question_to_ask"] == "What is the current status of root cause?"

    def test_entity_in_goal_is_critical(self):
        out = run_heuristic(
            goal="determine root cause for outage",
            available_memories=[{"content": "outage symptoms only"}],
            required_entities=["root cause"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"][0]["priority"] == "critical"

    def test_entity_not_in_goal_is_high(self):
        out = run_heuristic(
            goal="determine outage impact",
            available_memories=[{"content": "impact list"}],
            required_entities=["vendor timeline"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"][0]["priority"] == "high"

    def test_coverage_score_two_of_four(self):
        out = run_heuristic(
            goal="status report",
            available_memories=[{"content": "alpha beta"}],
            required_entities=["alpha", "beta", "gamma", "delta"],
            confidence_threshold=0.4,
        )
        assert out["coverage_score"] == 0.5

    def test_is_sufficient_false_when_below_threshold(self):
        out = run_heuristic(
            goal="status report",
            available_memories=[{"content": "alpha"}],
            required_entities=["alpha", "beta", "gamma"],
            confidence_threshold=0.8,
        )
        assert out["is_sufficient"] is False

    def test_is_sufficient_true_when_equal_threshold(self):
        out = run_heuristic(
            goal="status report",
            available_memories=[{"content": "alpha beta"}],
            required_entities=["alpha", "beta", "gamma", "delta"],
            confidence_threshold=0.5,
        )
        assert out["is_sufficient"] is True

    def test_top_question_prefers_critical(self):
        out = run_heuristic(
            goal="need root cause and budget owner",
            available_memories=[{"content": "nothing useful"}],
            required_entities=["root cause", "budget owner"],
            confidence_threshold=0.9,
        )
        assert "root cause" in (out["top_question"] or "").lower()

    def test_empty_required_entities_full_coverage(self):
        out = run_heuristic(
            goal="anything",
            available_memories=[],
            required_entities=[],
            confidence_threshold=0.4,
        )
        assert out["coverage_score"] == 1.0
        assert out["is_sufficient"] is True

    def test_case_insensitive_match(self):
        out = run_heuristic(
            goal="check ACME",
            available_memories=[{"content": "acme updated"}],
            required_entities=["ACME"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"] == []

    def test_multi_word_entity_match(self):
        out = run_heuristic(
            goal="investigate",
            available_memories=[{"content": "root cause identified in logs"}],
            required_entities=["root cause"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"] == []

    def test_multi_word_entity_missing(self):
        out = run_heuristic(
            goal="investigate",
            available_memories=[{"content": "only symptoms"}],
            required_entities=["root cause"],
            confidence_threshold=0.4,
        )
        assert len(out["knowledge_gaps"]) == 1

    def test_default_threshold_used_when_none(self):
        out = run_heuristic(
            goal="status",
            available_memories=[{"content": "alpha"}],
            required_entities=["alpha", "beta"],
            confidence_threshold=None,
        )
        assert out["is_sufficient"] is True

    def test_required_for_uses_goal(self):
        goal = "ship reliability report"
        out = run_heuristic(
            goal=goal,
            available_memories=[],
            required_entities=["error budget"],
            confidence_threshold=0.4,
        )
        assert out["knowledge_gaps"][0]["required_for"] == goal

    def test_confidence_constant(self):
        out = run_heuristic(
            goal="x",
            available_memories=[],
            required_entities=["y"],
            confidence_threshold=0.4,
        )
        assert out["confidence"] == 0.85


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = ActiveKnowledgeSeekerAgent()
        with patch("app.agents.active_knowledge_seeker_agent.settings") as mock_settings:
            mock_settings.ACTIVE_KNOWLEDGE_SEEKER_STRATEGY = "heuristic"
            result = await agent.run(
                "m1",
                _ctx(goal="status", available_memories=[], required_entities=["alpha"]),
            )
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = ActiveKnowledgeSeekerAgent()
        with patch("app.agents.active_knowledge_seeker_agent.settings") as mock_settings:
            mock_settings.ACTIVE_KNOWLEDGE_SEEKER_STRATEGY = "heuristic"
            result = await agent.run(
                "m1",
                _ctx(goal="status", available_memories=[], required_entities=["alpha"]),
            )
        assert result.trace_id == "trace-63"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = ActiveKnowledgeSeekerAgent()
        with patch("app.agents.active_knowledge_seeker_agent.settings") as mock_settings:
            mock_settings.ACTIVE_KNOWLEDGE_SEEKER_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run(
                "m1",
                _ctx(goal="status", available_memories=[], required_entities=["alpha"]),
            )
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = ActiveKnowledgeSeekerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "knowledge_gaps": [],
                "coverage_score": 0.8,
                "is_sufficient": True,
                "top_question": None,
                "confidence": 0.7,
                "rationale": "llm",
            }
        )
        with patch("app.agents.active_knowledge_seeker_agent.settings") as mock_settings, patch(
            "app.agents.active_knowledge_seeker_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.ACTIVE_KNOWLEDGE_SEEKER_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run(
                "m1",
                _ctx(goal="status", available_memories=[], required_entities=["alpha"]),
            )
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = ActiveKnowledgeSeekerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.active_knowledge_seeker_agent.settings") as mock_settings, patch(
            "app.agents.active_knowledge_seeker_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.ACTIVE_KNOWLEDGE_SEEKER_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run(
                "m1",
                _ctx(goal="status", available_memories=[], required_entities=["alpha"]),
            )
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="ActiveKnowledgeSeekerAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.7,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "knowledge_gaps": [],
            "coverage_score": 1.0,
            "is_sufficient": True,
            "top_question": None,
            "confidence": 0.85,
        }

    def test_validate_outputs_passes(self):
        ActiveKnowledgeSeekerAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_knowledge_gaps_type_raises(self):
        with pytest.raises(ValueError, match="knowledge_gaps"):
            ActiveKnowledgeSeekerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), knowledge_gaps="x"))
            )

    def test_coverage_score_type_raises(self):
        with pytest.raises(ValueError, match="coverage_score"):
            ActiveKnowledgeSeekerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), coverage_score="x"))
            )

    def test_coverage_score_range_raises(self):
        with pytest.raises(ValueError, match="coverage_score"):
            ActiveKnowledgeSeekerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), coverage_score=1.2))
            )

    def test_is_sufficient_type_raises(self):
        with pytest.raises(ValueError, match="is_sufficient"):
            ActiveKnowledgeSeekerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), is_sufficient="x"))
            )

    def test_top_question_type_raises(self):
        with pytest.raises(ValueError, match="top_question"):
            ActiveKnowledgeSeekerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), top_question=123))
            )

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="ActiveKnowledgeSeekerAgent",
            agent_version="v1",
            memory_id="m1",
            status="failed",
            confidence=0.0,
            outputs={},
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )
        ActiveKnowledgeSeekerAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("active_knowledge_seeker"), ActiveKnowledgeSeekerAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("activeknowledgeseeker"), ActiveKnowledgeSeekerAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("ActiveKnowledgeSeekerAgent"), ActiveKnowledgeSeekerAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("knowledge_seeker"), ActiveKnowledgeSeekerAgent)


class TestModelShape:
    def test_model_has_required_phase_63_fields(self):
        fields = KnowledgeGap.__table__.columns.keys()
        for name in (
            "org_id",
            "goal_id",
            "gap_description",
            "question_to_ask",
            "required_for",
            "priority",
            "status",
            "created_at",
            "resolved_at",
        ):
            assert name in fields
