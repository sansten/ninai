from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.registry import get_agent
from app.agents.social_memory_agent import SocialMemoryAgent, _clamp01, run_heuristic
from app.agents.types import AgentResult
from app.models.social_graph_edge import SocialGraphEdge


def _mem(actor: str, linked: list[str] | None = None) -> dict:
    return {
        "user_id": actor,
        "linked_user_ids": linked or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": [],
    }


def _ctx(*, memories: list[dict], org_users: list[str]) -> dict:
    return {
        "memory": {"enrichment": {"memories": memories, "org_users": org_users}},
        "runtime": {"job_id": "trace-66"},
    }


class TestHelpers:
    def test_clamp01_low(self):
        assert _clamp01(-1) == 0.0

    def test_clamp01_high(self):
        assert _clamp01(2) == 1.0

    def test_clamp01_mid(self):
        assert _clamp01(0.4) == 0.4


class TestHeuristic:
    def test_two_memories_shared_link_creates_edge(self):
        out = run_heuristic(
            memories=[_mem("u1", ["u2"]), _mem("u1", ["u2"])],
            org_users=["u1", "u2"],
        )
        assert any(e["actor"] == "u1" and e["collaborator"] == "u2" for e in out["collaboration_edges"])

    def test_user_with_no_links_is_silo(self):
        out = run_heuristic(memories=[_mem("u1", ["u2"])], org_users=["u1", "u2", "u3"])
        assert "u3" in out["knowledge_silos"]

    def test_most_connected_user(self):
        memories = [_mem("u1", ["u2", "u3"]), _mem("u2", ["u1"])]
        out = run_heuristic(memories=memories, org_users=["u1", "u2", "u3"])
        assert out["most_connected"] == "u1"

    def test_least_connected_non_zero_user(self):
        memories = [_mem("u1", ["u2", "u3"]), _mem("u2", ["u1"])]
        out = run_heuristic(memories=memories, org_users=["u1", "u2", "u3"])
        assert out["least_connected"] in {"u2", "u3"}

    def test_team_cohesion_all_connected_one(self):
        memories = [
            _mem("u1", ["u2", "u3"]),
            _mem("u2", ["u1", "u3"]),
            _mem("u3", ["u1", "u2"]),
        ]
        out = run_heuristic(memories=memories, org_users=["u1", "u2", "u3"])
        assert out["team_cohesion_score"] == 1.0

    def test_team_cohesion_no_connections_zero(self):
        memories = [_mem("u1", []), _mem("u2", [])]
        out = run_heuristic(memories=memories, org_users=["u1", "u2"])
        assert out["team_cohesion_score"] == 0.0

    def test_strength_bounded_range(self):
        out = run_heuristic(memories=[_mem("u1", ["u2", "u2", "u2"])], org_users=["u1", "u2"])
        assert all(0.0 <= e["strength"] <= 1.0 for e in out["collaboration_edges"])

    def test_org_users_empty_graceful(self):
        out = run_heuristic(memories=[], org_users=[])
        assert out["collaboration_edges"] == []
        assert out["knowledge_silos"] == []
        assert out["most_connected"] is None
        assert out["least_connected"] is None
        assert out["team_cohesion_score"] == 0.0

    def test_self_link_ignored(self):
        out = run_heuristic(memories=[_mem("u1", ["u1", "u2"])], org_users=["u1", "u2"])
        assert all(not (e["actor"] == "u1" and e["collaborator"] == "u1") for e in out["collaboration_edges"])

    def test_string_linked_user_ids_supported(self):
        mem = {"user_id": "u1", "linked_user_ids": "u2", "created_at": "2026-01-01T00:00:00Z", "tags": []}
        out = run_heuristic(memories=[mem], org_users=["u1", "u2"])
        assert len(out["collaboration_edges"]) == 1

    def test_actor_missing_skipped(self):
        out = run_heuristic(memories=[{"linked_user_ids": ["u2"]}], org_users=["u2"])
        assert out["collaboration_edges"] == []

    def test_interaction_count_aggregates(self):
        memories = [_mem("u1", ["u2"]), _mem("u1", ["u2"]), _mem("u1", ["u2"])]
        out = run_heuristic(memories=memories, org_users=["u1", "u2"])
        assert out["collaboration_edges"][0]["interaction_count"] == 3

    def test_strength_normalization_by_actor_total(self):
        memories = [_mem("u1", ["u2"]), _mem("u1", ["u2", "u3"])]
        out = run_heuristic(memories=memories, org_users=["u1", "u2", "u3"])
        edge = [e for e in out["collaboration_edges"] if e["collaborator"] == "u2"][0]
        assert edge["strength"] == round(2 / 3, 4)

    def test_silos_sorted(self):
        out = run_heuristic(memories=[_mem("u1", ["u2"])], org_users=["u3", "u4", "u1", "u2"])
        assert out["knowledge_silos"] == ["u3", "u4"]

    def test_most_connected_none_when_no_edges(self):
        out = run_heuristic(memories=[_mem("u1", [])], org_users=["u1"])
        assert out["most_connected"] is None

    def test_least_connected_none_when_no_edges(self):
        out = run_heuristic(memories=[_mem("u1", [])], org_users=["u1"])
        assert out["least_connected"] is None

    def test_team_cohesion_single_user_zero(self):
        out = run_heuristic(memories=[_mem("u1", [])], org_users=["u1"])
        assert out["team_cohesion_score"] == 0.0

    def test_undirected_edge_count_for_cohesion(self):
        memories = [_mem("u1", ["u2"]), _mem("u2", ["u1"])]
        out = run_heuristic(memories=memories, org_users=["u1", "u2"])
        assert out["team_cohesion_score"] == 1.0

    def test_confidence_constant(self):
        out = run_heuristic(memories=[], org_users=[])
        assert out["confidence"] == 0.8


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = SocialMemoryAgent()
        with patch("app.agents.social_memory_agent.settings") as mock_settings:
            mock_settings.SOCIAL_MEMORY_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(memories=[], org_users=[]))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = SocialMemoryAgent()
        with patch("app.agents.social_memory_agent.settings") as mock_settings:
            mock_settings.SOCIAL_MEMORY_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(memories=[], org_users=[]))
        assert result.trace_id == "trace-66"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = SocialMemoryAgent()
        with patch("app.agents.social_memory_agent.settings") as mock_settings:
            mock_settings.SOCIAL_MEMORY_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(memories=[], org_users=[]))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = SocialMemoryAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "collaboration_edges": [],
                "knowledge_silos": [],
                "most_connected": None,
                "least_connected": None,
                "team_cohesion_score": 0.4,
                "confidence": 0.7,
                "rationale": "llm",
            }
        )
        with patch("app.agents.social_memory_agent.settings") as mock_settings, patch(
            "app.agents.social_memory_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.SOCIAL_MEMORY_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(memories=[], org_users=[]))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = SocialMemoryAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.social_memory_agent.settings") as mock_settings, patch(
            "app.agents.social_memory_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.SOCIAL_MEMORY_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(memories=[], org_users=[]))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="SocialMemoryAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.8,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "collaboration_edges": [],
            "knowledge_silos": [],
            "most_connected": None,
            "least_connected": None,
            "team_cohesion_score": 0.0,
            "confidence": 0.8,
        }

    def test_validate_outputs_passes(self):
        SocialMemoryAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_edges_type_raises(self):
        with pytest.raises(ValueError, match="collaboration_edges"):
            SocialMemoryAgent().validate_outputs(self._result(dict(self._valid_outputs(), collaboration_edges="x")))

    def test_silos_type_raises(self):
        with pytest.raises(ValueError, match="knowledge_silos"):
            SocialMemoryAgent().validate_outputs(self._result(dict(self._valid_outputs(), knowledge_silos="x")))

    def test_most_connected_type_raises(self):
        with pytest.raises(ValueError, match="most_connected"):
            SocialMemoryAgent().validate_outputs(self._result(dict(self._valid_outputs(), most_connected=123)))

    def test_least_connected_type_raises(self):
        with pytest.raises(ValueError, match="least_connected"):
            SocialMemoryAgent().validate_outputs(self._result(dict(self._valid_outputs(), least_connected=123)))

    def test_team_cohesion_type_raises(self):
        with pytest.raises(ValueError, match="team_cohesion_score"):
            SocialMemoryAgent().validate_outputs(self._result(dict(self._valid_outputs(), team_cohesion_score="x")))

    def test_team_cohesion_range_raises(self):
        with pytest.raises(ValueError, match="team_cohesion_score"):
            SocialMemoryAgent().validate_outputs(self._result(dict(self._valid_outputs(), team_cohesion_score=1.2)))

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="SocialMemoryAgent",
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
        SocialMemoryAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("social_memory"), SocialMemoryAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("socialmemory"), SocialMemoryAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("SocialMemoryAgent"), SocialMemoryAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("team_dynamics"), SocialMemoryAgent)


class TestModelShape:
    def test_model_has_required_fields(self):
        fields = SocialGraphEdge.__table__.columns.keys()
        for name in (
            "org_id",
            "actor_user_id",
            "collaborator_user_id",
            "interaction_type",
            "interaction_count",
            "last_interaction",
            "strength",
        ):
            assert name in fields
