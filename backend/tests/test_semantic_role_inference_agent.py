from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.registry import get_agent
from app.agents.semantic_role_inference_agent import (
    SemanticRoleInferenceAgent,
    _confidence,
    _memory_text,
    _tokenize,
    run_heuristic,
)
from app.agents.types import AgentResult
from app.models.inferred_role import InferredRole


def _mem(user: str, content: str, tags: list[str] | None = None, action_type: str = "") -> dict:
    return {
        "user_id": user,
        "content": content,
        "tags": tags or [],
        "action_type": action_type,
    }


def _ctx(memories: list[dict], existing_roles: list[dict] | None = None) -> dict:
    return {
        "memory": {"enrichment": {"memories": memories, "existing_roles": existing_roles or []}},
        "runtime": {"job_id": "trace-69"},
    }


class TestHelpers:
    def test_tokenize_lower(self):
        assert "deploy" in _tokenize("Deploy")

    def test_memory_text_includes_tags(self):
        text = _memory_text(_mem("u1", "hello", ["tag1"], "act"))
        assert "tag1" in text

    def test_confidence_three_evidence(self):
        assert _confidence(3) == 0.7

    def test_confidence_clamped(self):
        assert _confidence(20) == 0.95


class TestHeuristic:
    def test_deploy_infers_deployer(self):
        out = run_heuristic(memories=[_mem("u1", "deploy to prod")], existing_roles=[])
        labels = {(r["entity_id"], r["role_label"]) for r in out["inferred_roles"]}
        assert ("u1", "deployer") in labels

    def test_three_deploy_confidence_point_seven(self):
        out = run_heuristic(
            memories=[_mem("u1", "deploy 1"), _mem("u1", "deploy 2"), _mem("u1", "deploy 3")],
            existing_roles=[],
        )
        role = [r for r in out["inferred_roles"] if r["entity_id"] == "u1" and r["role_label"] == "deployer"][0]
        assert role["confidence"] == 0.7

    def test_conflict_deployer_and_reviewer_counts(self):
        memories = [
            _mem("u1", "deploy service"),
            _mem("u1", "release patch"),
            _mem("u1", "review PR"),
            _mem("u1", "merge PR"),
        ]
        out = run_heuristic(memories=memories, existing_roles=[])
        assert any(c["entity_id"] == "u1" for c in out["conflicts"])

    def test_role_coverage_half_users(self):
        memories = [_mem("u1", "deploy service"), _mem("u2", "plain note")]
        out = run_heuristic(memories=memories, existing_roles=[])
        assert out["role_coverage"] == 0.5

    def test_empty_memories(self):
        out = run_heuristic(memories=[], existing_roles=[])
        assert out["inferred_roles"] == []
        assert out["role_coverage"] == 0.0

    def test_existing_roles_baseline_applied(self):
        out = run_heuristic(
            memories=[_mem("u1", "deploy service")],
            existing_roles=[{"entity_id": "u1", "role_label": "deployer", "evidence_count": 2}],
        )
        role = [r for r in out["inferred_roles"] if r["entity_id"] == "u1" and r["role_label"] == "deployer"][0]
        assert role["evidence_count"] == 3

    def test_confidence_clamped_to_point_nine_five(self):
        memories = [_mem("u1", "deploy now") for _ in range(12)]
        out = run_heuristic(memories=memories, existing_roles=[])
        role = [r for r in out["inferred_roles"] if r["entity_id"] == "u1" and r["role_label"] == "deployer"][0]
        assert role["confidence"] == 0.95

    def test_tags_can_trigger_role(self):
        out = run_heuristic(memories=[_mem("u1", "note", ["incident"])], existing_roles=[])
        assert any(r["role_label"] == "incident_owner" for r in out["inferred_roles"])

    def test_action_type_can_trigger_role(self):
        out = run_heuristic(memories=[_mem("u1", "note", [], "merge")], existing_roles=[])
        assert any(r["role_label"] == "reviewer" for r in out["inferred_roles"])

    def test_missing_user_skipped(self):
        out = run_heuristic(memories=[{"content": "deploy"}], existing_roles=[])
        assert out["inferred_roles"] == []

    def test_role_coverage_uses_unique_users_with_memories(self):
        memories = [_mem("u1", "deploy"), _mem("u1", "deploy again"), _mem("u2", "none")]
        out = run_heuristic(memories=memories, existing_roles=[])
        assert out["role_coverage"] == 0.5

    def test_multiple_roles_for_single_user(self):
        out = run_heuristic(memories=[_mem("u1", "deploy and review PR")], existing_roles=[])
        labels = {r["role_label"] for r in out["inferred_roles"] if r["entity_id"] == "u1"}
        assert "deployer" in labels and "reviewer" in labels

    def test_conflicts_empty_when_threshold_not_met(self):
        memories = [_mem("u1", "deploy"), _mem("u1", "review")]
        out = run_heuristic(memories=memories, existing_roles=[])
        assert out["conflicts"] == []

    def test_confidence_constant_output(self):
        out = run_heuristic(memories=[], existing_roles=[])
        assert out["confidence"] == 0.8


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = SemanticRoleInferenceAgent()
        with patch("app.agents.semantic_role_inference_agent.settings") as mock_settings:
            mock_settings.SEMANTIC_ROLE_INFERENCE_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([_mem("u1", "deploy")]))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = SemanticRoleInferenceAgent()
        with patch("app.agents.semantic_role_inference_agent.settings") as mock_settings:
            mock_settings.SEMANTIC_ROLE_INFERENCE_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([_mem("u1", "deploy")]))
        assert result.trace_id == "trace-69"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = SemanticRoleInferenceAgent()
        with patch("app.agents.semantic_role_inference_agent.settings") as mock_settings:
            mock_settings.SEMANTIC_ROLE_INFERENCE_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx([_mem("u1", "deploy")]))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = SemanticRoleInferenceAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "inferred_roles": [],
                "role_coverage": 0.6,
                "conflicts": [],
                "confidence": 0.7,
                "rationale": "llm",
            }
        )
        with patch("app.agents.semantic_role_inference_agent.settings") as mock_settings, patch(
            "app.agents.semantic_role_inference_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.SEMANTIC_ROLE_INFERENCE_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx([_mem("u1", "deploy")]))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = SemanticRoleInferenceAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.semantic_role_inference_agent.settings") as mock_settings, patch(
            "app.agents.semantic_role_inference_agent.create_llm_client", return_value=mock_client
        ):
            mock_settings.SEMANTIC_ROLE_INFERENCE_STRATEGY = "llm"
            mock_settings.VLLM_BASE_URL = "http://localhost:11434"
            mock_settings.VLLM_TIMEOUT_SECONDS = 5.0
            mock_settings.VLLM_MAX_CONCURRENCY = 2
            mock_settings.get_vllm_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx([_mem("u1", "deploy")]))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="SemanticRoleInferenceAgent",
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
            "inferred_roles": [],
            "role_coverage": 0.0,
            "conflicts": [],
            "confidence": 0.8,
        }

    def test_validate_outputs_passes(self):
        SemanticRoleInferenceAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_inferred_roles_type_raises(self):
        with pytest.raises(ValueError, match="inferred_roles"):
            SemanticRoleInferenceAgent().validate_outputs(self._result(dict(self._valid_outputs(), inferred_roles="x")))

    def test_role_coverage_type_raises(self):
        with pytest.raises(ValueError, match="role_coverage"):
            SemanticRoleInferenceAgent().validate_outputs(self._result(dict(self._valid_outputs(), role_coverage="x")))

    def test_role_coverage_range_raises(self):
        with pytest.raises(ValueError, match="role_coverage"):
            SemanticRoleInferenceAgent().validate_outputs(self._result(dict(self._valid_outputs(), role_coverage=1.2)))

    def test_conflicts_type_raises(self):
        with pytest.raises(ValueError, match="conflicts"):
            SemanticRoleInferenceAgent().validate_outputs(self._result(dict(self._valid_outputs(), conflicts="x")))

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="SemanticRoleInferenceAgent",
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
        SemanticRoleInferenceAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("semantic_role_inference"), SemanticRoleInferenceAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("semanticroleinference"), SemanticRoleInferenceAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("SemanticRoleInferenceAgent"), SemanticRoleInferenceAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("role_inference"), SemanticRoleInferenceAgent)


class TestModelShape:
    def test_model_has_required_fields(self):
        fields = InferredRole.__table__.columns.keys()
        for name in (
            "org_id",
            "entity_id",
            "entity_type",
            "role_label",
            "evidence_count",
            "confidence",
            "last_updated",
        ):
            assert name in fields
