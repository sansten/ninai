from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.agents.narrative_compression_agent import (
    NarrativeCompressionAgent,
    _parse_created_at,
    dominant_tag,
    run_heuristic,
    select_key_events,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.models.memory import MemoryMetadata
from app.services.narrative_compression_service import NarrativeCompressionService


def _episode(
    *,
    episode_id: str,
    content: str,
    created_at: str,
    tags: list[str] | None = None,
) -> dict:
    return {
        "id": episode_id,
        "content": content,
        "created_at": created_at,
        "tags": tags or [],
    }


def _ctx(episodes: list[dict], topic: str = "database incidents Q1", max_sentences: int = 3) -> dict:
    return {
        "memory": {
            "enrichment": {
                "episodes": episodes,
                "topic": topic,
                "max_sentences": max_sentences,
            }
        },
        "runtime": {"job_id": "trace-56"},
    }


class TestNarrativeCompressionHelpers:
    def test_parse_created_at_iso(self):
        dt = _parse_created_at("2026-01-01T10:00:00+00:00")
        assert dt.year == 2026

    def test_parse_created_at_invalid_returns_min(self):
        dt = _parse_created_at("not-a-date")
        assert dt.year == 1

    def test_dominant_tag_most_frequent(self):
        episodes = [
            _episode(episode_id="1", content="a", created_at="2026-01-01T01:00:00+00:00", tags=["db", "incident"]),
            _episode(episode_id="2", content="b", created_at="2026-01-01T02:00:00+00:00", tags=["db"]),
        ]
        assert dominant_tag(episodes) == "db"

    def test_dominant_tag_empty_general(self):
        assert dominant_tag([]) == "general"

    def test_select_key_events_prefers_overlap(self):
        episodes = [
            _episode(episode_id="1", content="cache issue", created_at="2026-01-01T01:00:00+00:00", tags=["cache"]),
            _episode(episode_id="2", content="database timeout", created_at="2026-01-01T02:00:00+00:00", tags=["database", "timeout"]),
        ]
        selected = select_key_events(episodes, "database incidents")
        assert selected[0]["id"] == "2"


class TestNarrativeCompressionHeuristic:
    def test_empty_episodes_graceful(self):
        out = run_heuristic(episodes=[], topic="x", max_sentences=3)
        assert out["compressed_narrative"] == ""
        assert out["key_events"] == []
        assert out["archived_ids"] == []

    def test_single_episode_ratio_point_33(self):
        episodes = [_episode(episode_id="1", content="db latency spike", created_at="2026-01-01T01:00:00+00:00", tags=["database"])]
        out = run_heuristic(episodes=episodes, topic="database", max_sentences=3)
        assert out["compression_ratio"] == pytest.approx(0.3333, abs=1e-4)
        assert len(out["key_events"]) == 1

    def test_ten_episodes_archived_five_non_key(self):
        episodes = [
            _episode(
                episode_id=str(i),
                content=f"event {i}",
                created_at=f"2026-01-01T{(i % 10):02d}:00:00+00:00",
                tags=["database"],
            )
            for i in range(10)
        ]
        out = run_heuristic(episodes=episodes, topic="database", max_sentences=3)
        assert len(out["archived_ids"]) == 5

    def test_time_span_ordered_from_to(self):
        episodes = [
            _episode(episode_id="1", content="late", created_at="2026-01-02T00:00:00+00:00", tags=["db"]),
            _episode(episode_id="2", content="early", created_at="2026-01-01T00:00:00+00:00", tags=["db"]),
        ]
        out = run_heuristic(episodes=episodes, topic="db", max_sentences=3)
        assert out["time_span"]["from"] < out["time_span"]["to"]

    def test_compressed_narrative_contains_topic(self):
        episodes = [_episode(episode_id="1", content="db event", created_at="2026-01-01T01:00:00+00:00", tags=["db"])]
        out = run_heuristic(episodes=episodes, topic="database incidents Q1", max_sentences=3)
        assert "database incidents q1" in out["compressed_narrative"].lower()

    def test_confidence_clamped_for_large_sets(self):
        episodes = [
            _episode(episode_id=str(i), content="event", created_at=f"2026-01-01T{(i % 10):02d}:00:00+00:00", tags=["db"])
            for i in range(20)
        ]
        out = run_heuristic(episodes=episodes, topic="db", max_sentences=3)
        assert out["confidence"] == 0.9

    def test_dominant_tag_reflected_in_narrative(self):
        episodes = [
            _episode(episode_id="1", content="x", created_at="2026-01-01T01:00:00+00:00", tags=["database"]),
            _episode(episode_id="2", content="y", created_at="2026-01-01T02:00:00+00:00", tags=["database"]),
            _episode(episode_id="3", content="z", created_at="2026-01-01T03:00:00+00:00", tags=["cache"]),
        ]
        out = run_heuristic(episodes=episodes, topic="database", max_sentences=4)
        assert "overall pattern: database" in out["compressed_narrative"].lower()

    def test_key_events_max_five(self):
        episodes = [
            _episode(episode_id=str(i), content=f"e{i}", created_at=f"2026-01-01T{(i % 10):02d}:00:00+00:00", tags=["database"])
            for i in range(12)
        ]
        out = run_heuristic(episodes=episodes, topic="database", max_sentences=3)
        assert len(out["key_events"]) <= 5

    def test_archived_ids_excludes_selected(self):
        episodes = [
            _episode(episode_id="a", content="database outage", created_at="2026-01-01T01:00:00+00:00", tags=["database"]),
            _episode(episode_id="b", content="network issue", created_at="2026-01-01T02:00:00+00:00", tags=["network"]),
        ]
        out = run_heuristic(episodes=episodes, topic="database", max_sentences=3)
        assert "a" not in out["archived_ids"]

    def test_respects_max_sentences_bound(self):
        episodes = [
            _episode(episode_id="1", content="one", created_at="2026-01-01T01:00:00+00:00", tags=["db"]),
            _episode(episode_id="2", content="two", created_at="2026-01-01T02:00:00+00:00", tags=["db"]),
        ]
        out = run_heuristic(episodes=episodes, topic="db", max_sentences=2)
        # At most two sentence terminators expected for max_sentences=2 template truncation.
        assert out["compressed_narrative"].count(".") <= 2


class TestNarrativeCompressionAgentRun:
    @pytest.mark.asyncio
    async def test_runs_heuristic_strategy(self):
        agent = NarrativeCompressionAgent()
        episodes = [_episode(episode_id="1", content="db", created_at="2026-01-01T00:00:00+00:00", tags=["database"])]
        with patch("app.agents.narrative_compression_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_COMPRESSION_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(episodes))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = NarrativeCompressionAgent()
        with patch("app.agents.narrative_compression_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_COMPRESSION_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([]))
        assert result.trace_id == "trace-56"

    @pytest.mark.asyncio
    async def test_agent_strategy_fallback(self):
        agent = NarrativeCompressionAgent()
        with patch("app.agents.narrative_compression_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_COMPRESSION_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([]))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_llm_valid_response_used(self):
        agent = NarrativeCompressionAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "compressed_narrative": "Summary.",
                "compression_ratio": 2.0,
                "key_events": ["A"],
                "time_span": {"from": "2026-01-01T00:00:00+00:00", "to": "2026-01-01T01:00:00+00:00"},
                "archived_ids": ["x"],
                "confidence": 0.8,
                "rationale": "llm",
            }
        )
        with patch("app.agents.narrative_compression_agent.settings") as mock_settings, patch(
            "app.agents.narrative_compression_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.NARRATIVE_COMPRESSION_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx([]))
        assert result.outputs["compressed_narrative"] == "Summary."

    @pytest.mark.asyncio
    async def test_llm_invalid_falls_back(self):
        agent = NarrativeCompressionAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.narrative_compression_agent.settings") as mock_settings, patch(
            "app.agents.narrative_compression_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.NARRATIVE_COMPRESSION_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx([]))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_validate_outputs_passes(self):
        agent = NarrativeCompressionAgent()
        with patch("app.agents.narrative_compression_agent.settings") as mock_settings:
            mock_settings.NARRATIVE_COMPRESSION_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([]))
        agent.validate_outputs(result)


class TestNarrativeCompressionValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="NarrativeCompressionAgent",
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
            "compressed_narrative": "A summary.",
            "compression_ratio": 0.5,
            "key_events": ["evt"],
            "time_span": {"from": "2026-01-01T00:00:00+00:00", "to": "2026-01-01T01:00:00+00:00"},
            "archived_ids": ["a"],
            "confidence": 0.7,
            "rationale": "heuristic",
        }

    def test_valid_outputs_pass(self):
        agent = NarrativeCompressionAgent()
        agent.validate_outputs(self._result(self._valid_outputs()))

    def test_missing_narrative_raises(self):
        agent = NarrativeCompressionAgent()
        outputs = dict(self._valid_outputs())
        del outputs["compressed_narrative"]
        with pytest.raises(ValueError, match="compressed_narrative"):
            agent.validate_outputs(self._result(outputs))

    def test_bad_ratio_type_raises(self):
        agent = NarrativeCompressionAgent()
        with pytest.raises(ValueError, match="compression_ratio"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), compression_ratio="x")))

    def test_bad_key_events_type_raises(self):
        agent = NarrativeCompressionAgent()
        with pytest.raises(ValueError, match="key_events"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), key_events="x")))

    def test_bad_archived_ids_type_raises(self):
        agent = NarrativeCompressionAgent()
        with pytest.raises(ValueError, match="archived_ids"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), archived_ids="x")))

    def test_bad_time_span_type_raises(self):
        agent = NarrativeCompressionAgent()
        with pytest.raises(ValueError, match="time_span"):
            agent.validate_outputs(self._result(dict(self._valid_outputs(), time_span="x")))

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name="NarrativeCompressionAgent",
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
        NarrativeCompressionAgent().validate_outputs(result)


class TestNarrativeCompressionRegistry:
    def test_registry_alias_primary(self):
        assert isinstance(get_agent("narrative_compression"), NarrativeCompressionAgent)

    def test_registry_alias_camel(self):
        assert isinstance(get_agent("NarrativeCompressionAgent"), NarrativeCompressionAgent)

    def test_registry_alias_short(self):
        assert isinstance(get_agent("compression"), NarrativeCompressionAgent)

    def test_registry_unknown(self):
        assert get_agent("narrative_compress_unknown") is None


class TestNarrativeCompressionService:
    @pytest.mark.asyncio
    async def test_compress_and_archive_creates_memory_and_archives(
        self,
        db_session,
        test_org_id: str,
        test_user_id: str,
    ):
        svc = NarrativeCompressionService()
        base = datetime.now(timezone.utc)

        mem1 = MemoryMetadata(
            id=str(uuid4()),
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="internal",
            content_preview="incident one",
            content_hash="h1" * 32,
            tags=["database"],
            entities={},
            extra_metadata={},
            source_type="manual",
            source_id=None,
            vector_id=f"v-{uuid4()}",
            embedding_model="test",
        )
        mem2 = MemoryMetadata(
            id=str(uuid4()),
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="internal",
            content_preview="incident two",
            content_hash="h2" * 32,
            tags=["network"],
            entities={},
            extra_metadata={},
            source_type="manual",
            source_id=None,
            vector_id=f"v-{uuid4()}",
            embedding_model="test",
        )
        db_session.add_all([mem1, mem2])
        await db_session.commit()

        episodes = [
            {"id": mem1.id, "content": "incident one", "created_at": (base - timedelta(hours=1)).isoformat(), "tags": ["database"]},
            {"id": mem2.id, "content": "incident two", "created_at": base.isoformat(), "tags": ["network"]},
        ]

        result = await svc.compress_and_archive(
            db=db_session,
            org_id=test_org_id,
            user_id=test_user_id,
            episodes=episodes,
            topic="database incidents",
            max_sentences=3,
        )

        assert result["new_memory_id"] is not None
        assert isinstance(result["narrative"], str)

    @pytest.mark.asyncio
    async def test_compress_and_archive_archived_count_matches_existing_ids(
        self,
        db_session,
        test_org_id: str,
        test_user_id: str,
    ):
        svc = NarrativeCompressionService()

        mem = MemoryMetadata(
            id=str(uuid4()),
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="internal",
            content_preview="incident",
            content_hash="h3" * 32,
            tags=["database"],
            entities={},
            extra_metadata={},
            source_type="manual",
            source_id=None,
            vector_id=f"v-{uuid4()}",
            embedding_model="test",
        )
        db_session.add(mem)
        await db_session.commit()

        episodes = [
            {"id": mem.id, "content": "incident", "created_at": "2026-01-01T00:00:00+00:00", "tags": ["x"]},
            {"id": str(uuid4()), "content": "missing", "created_at": "2026-01-01T01:00:00+00:00", "tags": ["x"]},
            {"id": str(uuid4()), "content": "missing2", "created_at": "2026-01-01T02:00:00+00:00", "tags": ["x"]},
            {"id": str(uuid4()), "content": "missing3", "created_at": "2026-01-01T03:00:00+00:00", "tags": ["x"]},
            {"id": str(uuid4()), "content": "missing4", "created_at": "2026-01-01T04:00:00+00:00", "tags": ["x"]},
            {"id": str(uuid4()), "content": "missing5", "created_at": "2026-01-01T05:00:00+00:00", "tags": ["x"]},
        ]

        result = await svc.compress_and_archive(
            db=db_session,
            org_id=test_org_id,
            user_id=test_user_id,
            episodes=episodes,
            topic="x",
        )
        assert result["archived_count"] == 1

    @pytest.mark.asyncio
    async def test_compress_and_archive_empty_episodes_graceful(self, db_session, test_org_id: str, test_user_id: str):
        result = await NarrativeCompressionService().compress_and_archive(
            db=db_session,
            org_id=test_org_id,
            user_id=test_user_id,
            episodes=[],
            topic="none",
        )
        assert result["new_memory_id"] is None
        assert result["archived_count"] == 0

    @pytest.mark.asyncio
    async def test_compress_and_archive_sets_archive_attrs_on_rows(self, db_session, test_org_id: str, test_user_id: str):
        mem = MemoryMetadata(
            id=str(uuid4()),
            organization_id=test_org_id,
            owner_id=test_user_id,
            scope="personal",
            classification="internal",
            content_preview="incident",
            content_hash="h4" * 32,
            tags=["database"],
            entities={},
            extra_metadata={},
            source_type="manual",
            source_id=None,
            vector_id=f"v-{uuid4()}",
            embedding_model="test",
        )
        db_session.add(mem)
        await db_session.commit()

        episodes = [
            {"id": str(uuid4()), "content": "k1", "created_at": "2026-01-01T00:00:00+00:00", "tags": ["database"]},
            {"id": str(uuid4()), "content": "k2", "created_at": "2026-01-01T00:01:00+00:00", "tags": ["database"]},
            {"id": str(uuid4()), "content": "k3", "created_at": "2026-01-01T00:02:00+00:00", "tags": ["database"]},
            {"id": str(uuid4()), "content": "k4", "created_at": "2026-01-01T00:03:00+00:00", "tags": ["database"]},
            {"id": str(uuid4()), "content": "k5", "created_at": "2026-01-01T00:04:00+00:00", "tags": ["database"]},
            {"id": mem.id, "content": "archive me", "created_at": "2026-01-01T00:05:00+00:00", "tags": ["network"]},
        ]

        await NarrativeCompressionService().compress_and_archive(
            db=db_session,
            org_id=test_org_id,
            user_id=test_user_id,
            episodes=episodes,
            topic="database",
        )

        refreshed = await db_session.get(MemoryMetadata, mem.id)
        assert refreshed.extra_metadata.get("is_archived") is True
        assert refreshed.extra_metadata.get("archived_at") is not None
