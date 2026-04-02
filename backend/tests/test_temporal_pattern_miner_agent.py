from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.registry import get_agent
from app.agents.temporal_pattern_miner_agent import (
    TemporalPatternMinerAgent,
    _avg_severity,
    _parse_dt,
    _top_tags,
    run_heuristic,
)
from app.agents.types import AgentResult
from app.models.temporal_pattern import TemporalPattern


_NOW = datetime.now(timezone.utc)


def _mem(*, mem_id: str, days_ago: int, hour: int = 9, tags: list[str] | None = None, severity: float = 0.5) -> dict:
    created = (_NOW - timedelta(days=days_ago)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return {
        "id": mem_id,
        "created_at": created.isoformat(),
        "tags": tags or [],
        "severity": severity,
    }


def _ctx(
    *,
    memories: list[dict],
    analysis_window_days: int = 90,
    min_occurrences: int = 3,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "memories": memories,
                "analysis_window_days": analysis_window_days,
                "min_occurrences": min_occurrences,
            }
        },
        "runtime": {"job_id": "trace-62"},
    }


class TestHelpers:
    def test_parse_dt_datetime_naive_gets_utc(self):
        dt = datetime(2026, 1, 1, 10, 0, 0)
        out = _parse_dt(dt)
        assert out is not None
        assert out.tzinfo is not None

    def test_parse_dt_iso_z(self):
        out = _parse_dt("2026-01-01T10:00:00Z")
        assert out is not None
        assert out.tzinfo is not None

    def test_parse_dt_invalid_returns_none(self):
        assert _parse_dt("bad-date") is None

    def test_top_tags_counts_and_limits(self):
        memories = [
            {"tags": ["alpha", "beta"]},
            {"tags": ["alpha", "gamma"]},
            {"tags": ["alpha"]},
        ]
        out = _top_tags(memories, [0, 1, 2], limit=2)
        assert out[0] == "alpha"
        assert len(out) == 2

    def test_avg_severity_uses_default_when_missing(self):
        memories = [{"severity": 0.9}, {}, {"severity": 0.3}]
        out = _avg_severity(memories, [0, 1, 2])
        assert 0.0 <= out <= 1.0


class TestHeuristic:
    def test_empty_memories(self):
        out = run_heuristic(memories=[])
        assert out["patterns"] == []
        assert out["total_events_analysed"] == 0
        assert out["rationale"] == "heuristic"

    def test_invalid_dates_filtered_out(self):
        out = run_heuristic(memories=[{"id": "m1", "created_at": "bad"}])
        assert out["total_events_analysed"] == 0

    def test_hour_pattern_detected(self):
        memories = [
            _mem(mem_id="m1", days_ago=1, hour=9, tags=["ops"], severity=0.8),
            _mem(mem_id="m2", days_ago=2, hour=9, tags=["ops"], severity=0.7),
            _mem(mem_id="m3", days_ago=3, hour=9, tags=["infra"], severity=0.6),
        ]
        out = run_heuristic(memories=memories, min_occurrences=3)
        keys = {p["pattern_key"] for p in out["patterns"]}
        assert "hour_9" in keys

    def test_day_of_week_pattern_detected(self):
        base = datetime(2026, 1, 26, 8, 0, 0, tzinfo=timezone.utc)  # Monday
        memories = []
        for i in [0, 7, 14]:
            dt = base - timedelta(days=i)
            memories.append({"id": f"m{i}", "created_at": dt.isoformat(), "tags": ["weekly"], "severity": 0.5})
        out = run_heuristic(memories=memories, min_occurrences=3)
        assert any(p["pattern_type"] == "day_of_week" for p in out["patterns"])

    def test_min_occurrences_filters_patterns(self):
        memories = [
            _mem(mem_id="m1", days_ago=1, hour=11),
            _mem(mem_id="m2", days_ago=2, hour=11),
        ]
        out = run_heuristic(memories=memories, min_occurrences=3)
        assert out["patterns"] == []

    def test_dominant_pattern_present_on_clear_winner(self):
        memories = [
            _mem(mem_id="m1", days_ago=1, hour=7),
            _mem(mem_id="m2", days_ago=2, hour=7),
            _mem(mem_id="m3", days_ago=3, hour=7),
            _mem(mem_id="m4", days_ago=4, hour=7),
            _mem(mem_id="m5", days_ago=5, hour=12),
            _mem(mem_id="m6", days_ago=6, hour=12),
            _mem(mem_id="m7", days_ago=7, hour=12),
        ]
        out = run_heuristic(memories=memories, min_occurrences=3)
        assert out["dominant_pattern"] is not None

    def test_confidence_in_range(self):
        memories = [_mem(mem_id=f"m{i}", days_ago=i, hour=10) for i in range(1, 5)]
        out = run_heuristic(memories=memories, min_occurrences=2)
        assert 0.0 <= out["confidence"] <= 1.0

    def test_anomalous_times_detected_for_spike(self):
        memories = []
        for i in range(20):
            memories.append(_mem(mem_id=f"s{i}", days_ago=i % 10, hour=9))
        for i in range(4):
            memories.append(_mem(mem_id=f"b{i}", days_ago=i + 1, hour=14))
        out = run_heuristic(memories=memories, min_occurrences=3)
        assert isinstance(out["anomalous_times"], list)


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = TemporalPatternMinerAgent()
        with patch("app.agents.temporal_pattern_miner_agent.settings") as mock_settings:
            mock_settings.TEMPORAL_PATTERN_MINER_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(memories=[]))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = TemporalPatternMinerAgent()
        with patch("app.agents.temporal_pattern_miner_agent.settings") as mock_settings:
            mock_settings.TEMPORAL_PATTERN_MINER_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(memories=[]))
        assert result.trace_id == "trace-62"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = TemporalPatternMinerAgent()
        with patch("app.agents.temporal_pattern_miner_agent.settings") as mock_settings:
            mock_settings.TEMPORAL_PATTERN_MINER_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(memories=[]))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = TemporalPatternMinerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "patterns": [],
                "dominant_pattern": None,
                "anomalous_times": [],
                "total_events_analysed": 3,
                "confidence": 0.71,
                "rationale": "llm",
            }
        )
        with patch("app.agents.temporal_pattern_miner_agent.settings") as mock_settings, patch(
            "app.agents.temporal_pattern_miner_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.TEMPORAL_PATTERN_MINER_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(memories=[]))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = TemporalPatternMinerAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.temporal_pattern_miner_agent.settings") as mock_settings, patch(
            "app.agents.temporal_pattern_miner_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.TEMPORAL_PATTERN_MINER_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(memories=[]))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="TemporalPatternMinerAgent",
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
            "patterns": [],
            "dominant_pattern": None,
            "anomalous_times": [],
            "total_events_analysed": 0,
            "confidence": 0.5,
        }

    def test_validate_outputs_passes(self):
        TemporalPatternMinerAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_patterns_type_raises(self):
        with pytest.raises(ValueError, match="patterns"):
            TemporalPatternMinerAgent().validate_outputs(self._result(dict(self._valid_outputs(), patterns="x")))

    def test_dominant_pattern_type_raises(self):
        with pytest.raises(ValueError, match="dominant_pattern"):
            TemporalPatternMinerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), dominant_pattern="x"))
            )

    def test_anomalous_times_type_raises(self):
        with pytest.raises(ValueError, match="anomalous_times"):
            TemporalPatternMinerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), anomalous_times="x"))
            )

    def test_total_events_type_raises(self):
        with pytest.raises(ValueError, match="total_events_analysed"):
            TemporalPatternMinerAgent().validate_outputs(
                self._result(dict(self._valid_outputs(), total_events_analysed="x"))
            )

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="TemporalPatternMinerAgent",
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
        TemporalPatternMinerAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("temporal_pattern_miner"), TemporalPatternMinerAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("temporalpatternminer"), TemporalPatternMinerAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("TemporalPatternMinerAgent"), TemporalPatternMinerAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("pattern_miner"), TemporalPatternMinerAgent)


class TestModelShape:
    def test_model_has_required_fields(self):
        fields = TemporalPattern.__table__.columns.keys()
        for name in (
            "org_id",
            "pattern_type",
            "pattern_key",
            "topic_tags",
            "occurrence_count",
            "avg_severity",
            "first_seen",
            "last_seen",
            "confidence",
        ):
            assert name in fields
