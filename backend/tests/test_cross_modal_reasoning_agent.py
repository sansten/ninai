from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.cross_modal_reasoning_agent import (
    CrossModalReasoningAgent,
    _as_tag_set,
    _modality_label,
    _parse_dt,
    _query_overlap,
    _tokenize,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


def _dt(minutes: int) -> str:
    base = datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
    return (base + timedelta(minutes=minutes)).isoformat()


def _ctx(
    *,
    text_memories: list[dict],
    visual_memories: list[dict],
    audio_memories: list[dict],
    query: str,
    time_window_minutes: int = 60,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "text_memories": text_memories,
                "visual_memories": visual_memories,
                "audio_memories": audio_memories,
                "query": query,
                "time_window_minutes": time_window_minutes,
            }
        },
        "runtime": {"job_id": "trace-76"},
    }


class TestHelpers:
    def test_tokenize_lower(self):
        assert "alert" in _tokenize("ALERT fired")

    def test_as_tag_set_from_list(self):
        tags = _as_tag_set(["db-primary", "error"])
        assert "db" in tags
        assert "primary" in tags

    def test_as_tag_set_from_string(self):
        tags = _as_tag_set("cache-miss")
        assert "cache" in tags

    def test_parse_dt_z_suffix(self):
        dt = _parse_dt("2026-04-02T12:00:00Z")
        assert dt.tzinfo is not None

    def test_query_overlap_threshold_example(self):
        overlap = _query_overlap({"db", "alert"}, {"db", "alert", "prod", "incident", "sev1"})
        assert overlap == pytest.approx(0.4)

    def test_modality_label_visual_alias(self):
        assert _modality_label({"modality": "image"}, "visual") == "visual"


class TestHeuristic:
    def test_text_and_visual_shared_tag_within_window_creates_link(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db", "alert"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(1), "modality": "image"}],
            audio_memories=[],
            query="db alert",
            time_window_minutes=60,
        )
        assert len(out["cross_modal_links"]) == 1
        assert out["cross_modal_links"][0]["text_id"] == "t1"
        assert out["cross_modal_links"][0]["visual_id"] == "v1"

    def test_gap_greater_than_window_no_link(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(200), "modality": "image"}],
            audio_memories=[],
            query="db",
            time_window_minutes=30,
        )
        assert out["cross_modal_links"] == []

    def test_correlation_positive_for_overlap_and_small_gap(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db", "incident"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(1)}],
            audio_memories=[],
            query="db incident",
            time_window_minutes=60,
        )
        assert out["cross_modal_links"][0]["correlation_score"] > 0

    def test_link_type_temporal_when_gap_under_300_seconds(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(4)}],
            audio_memories=[],
            query="db",
            time_window_minutes=60,
        )
        assert out["cross_modal_links"][0]["link_type"] == "temporal_co_occurrence"

    def test_modalities_used_contains_text_when_text_matches(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db", "alert"], "created_at": _dt(0)}],
            visual_memories=[],
            audio_memories=[],
            query="db alert",
            time_window_minutes=60,
        )
        assert out["modalities_used"] == ["text"]

    def test_evidence_strength_saturates_at_one_with_five_links(self):
        text = [{"id": "t1", "tags": ["db", "alert"], "created_at": _dt(0)}]
        visuals = [
            {"id": f"v{i}", "searchable_tags": ["db"], "created_at": _dt(i)}
            for i in range(1, 8)
        ]
        out = run_heuristic(
            text_memories=text,
            visual_memories=visuals,
            audio_memories=[],
            query="db alert",
            time_window_minutes=60,
        )
        assert len(out["cross_modal_links"]) >= 5
        assert out["evidence_strength"] == 1.0

    def test_unified_conclusion_contains_query_text(self):
        out = run_heuristic(
            text_memories=[],
            visual_memories=[],
            audio_memories=[],
            query="database latency incident",
            time_window_minutes=60,
        )
        assert "database latency incident" in out["unified_conclusion"]

    def test_empty_visual_memories_modalities_text_only(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["api"], "created_at": _dt(0)}],
            visual_memories=[],
            audio_memories=[],
            query="api",
            time_window_minutes=60,
        )
        assert out["cross_modal_links"] == []
        assert out["modalities_used"] == ["text"]

    def test_audio_link_uses_audio_id_field(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["voice", "alert"], "created_at": _dt(0)}],
            visual_memories=[],
            audio_memories=[{"id": "a1", "searchable_tags": ["voice"], "created_at": _dt(1), "modality": "audio"}],
            query="voice alert",
            time_window_minutes=60,
        )
        assert "audio_id" in out["cross_modal_links"][0]

    def test_confidence_clamped_at_point_nine(self):
        text = [{"id": "t1", "tags": ["db", "alert"], "created_at": _dt(0)}]
        visuals = [
            {"id": f"v{i}", "searchable_tags": ["db"], "created_at": _dt(i)}
            for i in range(1, 10)
        ]
        out = run_heuristic(
            text_memories=text,
            visual_memories=visuals,
            audio_memories=[],
            query="db alert",
            time_window_minutes=60,
        )
        assert out["confidence"] == 0.9

    def test_no_query_overlap_no_links(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(1)}],
            audio_memories=[],
            query="network",
            time_window_minutes=60,
        )
        assert out["cross_modal_links"] == []
        assert out["modalities_used"] == []

    def test_time_window_defaults_to_60(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(30)}],
            audio_memories=[],
            query="db",
            time_window_minutes=None,
        )
        assert len(out["cross_modal_links"]) == 1

    def test_small_window_clamped_to_one_minute(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(5)}],
            audio_memories=[],
            query="db",
            time_window_minutes=0,
        )
        assert out["cross_modal_links"] == []

    def test_modalities_sorted(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db", "voice"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(1)}],
            audio_memories=[{"id": "a1", "searchable_tags": ["voice"], "created_at": _dt(1)}],
            query="db voice",
            time_window_minutes=60,
        )
        assert out["modalities_used"] == ["audio", "text", "visual"]

    def test_link_requires_shared_tags(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["cache"], "created_at": _dt(1)}],
            audio_memories=[],
            query="db",
            time_window_minutes=60,
        )
        assert out["cross_modal_links"] == []

    def test_link_requires_correlation_above_point_one(self):
        out = run_heuristic(
            text_memories=[{"id": "t1", "tags": ["db"], "created_at": _dt(0)}],
            visual_memories=[{"id": "v1", "searchable_tags": ["db"], "created_at": _dt(59)}],
            audio_memories=[],
            query="db token1 token2 token3 token4 token5 token6 token7 token8 token9",
            time_window_minutes=60,
        )
        assert out["cross_modal_links"] == []


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = CrossModalReasoningAgent()
        with patch("app.agents.cross_modal_reasoning_agent.settings") as mock_settings:
            mock_settings.CROSS_MODAL_REASONING_STRATEGY = "heuristic"
            result = await agent.run(
                "m1",
                _ctx(text_memories=[], visual_memories=[], audio_memories=[], query="db"),
            )
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = CrossModalReasoningAgent()
        with patch("app.agents.cross_modal_reasoning_agent.settings") as mock_settings:
            mock_settings.CROSS_MODAL_REASONING_STRATEGY = "heuristic"
            result = await agent.run(
                "m1",
                _ctx(text_memories=[], visual_memories=[], audio_memories=[], query="db"),
            )
        assert result.trace_id == "trace-76"

    @pytest.mark.asyncio
    async def test_strategy_fallback_to_agent_strategy(self):
        agent = CrossModalReasoningAgent()
        with patch("app.agents.cross_modal_reasoning_agent.settings") as mock_settings:
            mock_settings.CROSS_MODAL_REASONING_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run(
                "m1",
                _ctx(text_memories=[], visual_memories=[], audio_memories=[], query="db"),
            )
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = CrossModalReasoningAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "cross_modal_links": [],
                "unified_conclusion": "ok",
                "modalities_used": ["text"],
                "evidence_strength": 0.2,
                "confidence": 0.6,
                "rationale": "llm",
            }
        )
        with patch("app.agents.cross_modal_reasoning_agent.settings") as mock_settings, patch(
            "app.agents.cross_modal_reasoning_agent.create_ollama_client",
            return_value=mock_client,
        ):
            mock_settings.CROSS_MODAL_REASONING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run(
                "m1",
                _ctx(text_memories=[], visual_memories=[], audio_memories=[], query="db"),
            )
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = CrossModalReasoningAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.cross_modal_reasoning_agent.settings") as mock_settings, patch(
            "app.agents.cross_modal_reasoning_agent.create_ollama_client",
            return_value=mock_client,
        ):
            mock_settings.CROSS_MODAL_REASONING_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run(
                "m1",
                _ctx(text_memories=[], visual_memories=[], audio_memories=[], query="db"),
            )
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="CrossModalReasoningAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.6,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "cross_modal_links": [],
            "unified_conclusion": "ok",
            "modalities_used": ["text"],
            "evidence_strength": 0.2,
            "confidence": 0.6,
        }

    def test_validate_outputs_passes(self):
        CrossModalReasoningAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_cross_modal_links_type_raises(self):
        with pytest.raises(ValueError, match="cross_modal_links"):
            CrossModalReasoningAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), cross_modal_links="x"))
            )

    def test_unified_conclusion_type_raises(self):
        with pytest.raises(ValueError, match="unified_conclusion"):
            CrossModalReasoningAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), unified_conclusion=[]))
            )

    def test_modalities_used_type_raises(self):
        with pytest.raises(ValueError, match="modalities_used"):
            CrossModalReasoningAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), modalities_used="x"))
            )

    def test_evidence_strength_type_raises(self):
        with pytest.raises(ValueError, match="evidence_strength"):
            CrossModalReasoningAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), evidence_strength="x"))
            )

    def test_evidence_strength_range_raises(self):
        with pytest.raises(ValueError, match="evidence_strength"):
            CrossModalReasoningAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), evidence_strength=1.2))
            )

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="CrossModalReasoningAgent",
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
        CrossModalReasoningAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("cross_modal_reasoning"), CrossModalReasoningAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("crossmodalreasoning"), CrossModalReasoningAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("CrossModalReasoningAgent"), CrossModalReasoningAgent)
