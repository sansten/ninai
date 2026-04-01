from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.registry import get_agent
from app.agents.semantic_change_detection_agent import (
    SemanticChangeDetectionAgent,
    _has_negation_shift,
    _tokenize,
    jaccard_similarity,
    run_heuristic,
)
from app.agents.types import AgentResult


def _ctx(
    *,
    new_content: str,
    existing_memories: list[dict],
    change_threshold: float = 0.4,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "new_content": new_content,
                "existing_memories": existing_memories,
                "change_threshold": change_threshold,
            }
        },
        "runtime": {"job_id": "trace-57"},
    }


class TestHelpers:
    def test_tokenize_lowercases(self):
        assert "database" in _tokenize("Database")

    def test_jaccard_identical_is_one(self):
        s = {"a", "b"}
        assert jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint_is_zero(self):
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_negation_shift_true(self):
        assert _has_negation_shift({"service", "not", "available"}, {"service", "available"})

    def test_negation_shift_false(self):
        assert not _has_negation_shift({"service", "available"}, {"service", "available"})


class TestHeuristic:
    def test_identical_content_no_change_detected(self):
        out = run_heuristic(
            new_content="database is healthy",
            existing_memories=[{"id": "m1", "content": "database is healthy"}],
            change_threshold=0.4,
        )
        assert out["change_detected"] is False
        assert out["semantic_drift_score"] == 0.0

    def test_negation_contradiction(self):
        out = run_heuristic(
            new_content="service is not available",
            existing_memories=[{"id": "m1", "content": "service is available"}],
            change_threshold=0.6,
        )
        assert out["change_type"] == "contradiction"
        assert out["recommended_action"] == "supersede"

    def test_low_similarity_without_negation_is_update(self):
        out = run_heuristic(
            new_content="cache eviction policy changed",
            existing_memories=[{"id": "m1", "content": "database backup completed"}],
            change_threshold=0.4,
        )
        assert out["change_type"] == "update"
        assert out["recommended_action"] == "flag_review"

    def test_high_similarity_is_extension(self):
        out = run_heuristic(
            new_content="database migration completed",
            existing_memories=[{"id": "m1", "content": "database migration completed"}],
            change_threshold=0.4,
        )
        assert out["change_type"] == "extension"
        assert out["recommended_action"] == "append"

    def test_changed_memories_contains_low_similarity_items(self):
        out = run_heuristic(
            new_content="x y z",
            existing_memories=[
                {"id": "m1", "content": "a b c"},
                {"id": "m2", "content": "x y z"},
            ],
            change_threshold=0.4,
        )
        ids = {row["memory_id"] for row in out["changed_memories"]}
        assert "m1" in ids
        assert "m2" not in ids

    def test_drift_approaches_one_for_different_content(self):
        out = run_heuristic(
            new_content="alpha beta gamma",
            existing_memories=[{"id": "m1", "content": "delta epsilon zeta"}],
            change_threshold=0.4,
        )
        assert out["semantic_drift_score"] == pytest.approx(1.0, abs=1e-4)

    def test_empty_existing_memories_graceful(self):
        out = run_heuristic(new_content="anything", existing_memories=[], change_threshold=0.4)
        assert out["change_detected"] is False
        assert out["change_type"] == "unrelated"
        assert out["recommended_action"] == "ignore"

    def test_confidence_low_when_change_detected(self):
        out = run_heuristic(
            new_content="one",
            existing_memories=[{"id": "m1", "content": "two"}],
            change_threshold=0.4,
        )
        assert out["confidence"] == 0.75

    def test_confidence_high_when_no_change(self):
        out = run_heuristic(
            new_content="same text",
            existing_memories=[{"id": "m1", "content": "same text"}],
            change_threshold=0.4,
        )
        assert out["confidence"] == 0.9

    def test_change_detected_any_below_threshold(self):
        out = run_heuristic(
            new_content="database healthy",
            existing_memories=[
                {"id": "m1", "content": "database healthy"},
                {"id": "m2", "content": "network outage"},
            ],
            change_threshold=0.4,
        )
        assert out["change_detected"] is True

    def test_changed_memory_shape(self):
        out = run_heuristic(
            new_content="foo",
            existing_memories=[{"id": "m1", "content": "bar"}],
            change_threshold=0.4,
        )
        row = out["changed_memories"][0]
        for key in ("memory_id", "old_content_snippet", "similarity", "change_type"):
            assert key in row


class TestAgentRunHeuristic:
    @pytest.mark.asyncio
    async def test_run_heuristic_success(self):
        agent = SemanticChangeDetectionAgent()
        with patch("app.agents.semantic_change_detection_agent.settings") as mock_settings:
            mock_settings.SEMANTIC_CHANGE_DETECTION_STRATEGY = "heuristic"
            result = await agent.run(
                "mem-1",
                _ctx(new_content="database healthy", existing_memories=[{"id": "m1", "content": "database healthy"}]),
            )
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = SemanticChangeDetectionAgent()
        with patch("app.agents.semantic_change_detection_agent.settings") as mock_settings:
            mock_settings.SEMANTIC_CHANGE_DETECTION_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(new_content="x", existing_memories=[]))
        assert result.trace_id == "trace-57"

    @pytest.mark.asyncio
    async def test_strategy_fallback_to_agent_strategy(self):
        agent = SemanticChangeDetectionAgent()
        with patch("app.agents.semantic_change_detection_agent.settings") as mock_settings:
            mock_settings.SEMANTIC_CHANGE_DETECTION_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(new_content="x", existing_memories=[]))
        assert result.outputs["rationale"] == "heuristic"


class TestAgentRunLLM:
    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = SemanticChangeDetectionAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "change_detected": True,
                "changed_memories": [],
                "change_type": "update",
                "semantic_drift_score": 0.6,
                "recommended_action": "flag_review",
                "confidence": 0.8,
                "rationale": "llm",
            }
        )
        with patch("app.agents.semantic_change_detection_agent.settings") as mock_settings, patch(
            "app.agents.semantic_change_detection_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.SEMANTIC_CHANGE_DETECTION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("mem-1", _ctx(new_content="x", existing_memories=[]))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back_to_heuristic(self):
        agent = SemanticChangeDetectionAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.semantic_change_detection_agent.settings") as mock_settings, patch(
            "app.agents.semantic_change_detection_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.SEMANTIC_CHANGE_DETECTION_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("mem-1", _ctx(new_content="x", existing_memories=[]))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="SemanticChangeDetectionAgent",
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
            "change_detected": False,
            "changed_memories": [],
            "change_type": "unrelated",
            "semantic_drift_score": 0.0,
            "recommended_action": "ignore",
            "confidence": 0.9,
        }

    def test_validate_outputs_passes(self):
        SemanticChangeDetectionAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_invalid_change_detected_type_raises(self):
        with pytest.raises(ValueError, match="change_detected"):
            SemanticChangeDetectionAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), change_detected="yes"))
            )

    def test_invalid_changed_memories_type_raises(self):
        with pytest.raises(ValueError, match="changed_memories"):
            SemanticChangeDetectionAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), changed_memories="oops"))
            )

    def test_invalid_change_type_raises(self):
        with pytest.raises(ValueError, match="change_type"):
            SemanticChangeDetectionAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), change_type="weird"))
            )

    def test_invalid_drift_raises(self):
        with pytest.raises(ValueError, match="semantic_drift_score"):
            SemanticChangeDetectionAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), semantic_drift_score=1.5))
            )

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="recommended_action"):
            SemanticChangeDetectionAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), recommended_action="x"))
            )

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="SemanticChangeDetectionAgent",
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
        SemanticChangeDetectionAgent().validate_outputs(result)


class TestRegistry:
    def test_semantic_change_detection_alias(self):
        assert isinstance(get_agent("semantic_change_detection"), SemanticChangeDetectionAgent)

    def test_semanticchangedetection_alias(self):
        assert isinstance(get_agent("semanticchangedetection"), SemanticChangeDetectionAgent)

    def test_semanticchangedetectionagent_alias(self):
        assert isinstance(get_agent("SemanticChangeDetectionAgent"), SemanticChangeDetectionAgent)

    def test_change_detection_alias(self):
        assert isinstance(get_agent("change_detection"), SemanticChangeDetectionAgent)

    def test_registry_unknown(self):
        assert get_agent("not-a-real-agent") is None
