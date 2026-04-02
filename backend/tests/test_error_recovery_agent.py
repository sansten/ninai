from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.error_recovery_agent import (
    ErrorRecoveryAgent,
    _choose_fallback_tool,
    _step_title,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


def _step(*, step_index: int = 1, title: str = "call primary_api", error_type: str = "unknown", attempts: int = 0) -> dict:
    return {
        "step_index": step_index,
        "title": title,
        "error_type": error_type,
        "error_message": f"{error_type} happened",
        "attempts": attempts,
    }


def _ctx(
    *,
    failed_step: dict,
    remaining_plan: list[dict] | None = None,
    completed_steps: list[dict] | None = None,
    available_tools: list[str] | None = None,
) -> dict:
    return {
        "memory": {
            "enrichment": {
                "failed_step": failed_step,
                "remaining_plan": remaining_plan or [],
                "completed_steps": completed_steps or [],
                "available_tools": available_tools or [],
            }
        },
        "runtime": {"job_id": "trace-68"},
    }


class TestHelpers:
    def test_step_title_returns_string(self):
        assert _step_title({"title": "hello"}) == "hello"

    def test_choose_fallback_avoids_tool_in_title(self):
        tool = _choose_fallback_tool(failed_step={"title": "run primary_api step"}, available_tools=["primary_api", "backup_api"])
        assert tool == "backup_api"

    def test_choose_fallback_first_tool_when_all_mentioned(self):
        tool = _choose_fallback_tool(failed_step={"title": "use api"}, available_tools=["api", "api"])
        assert tool == "api"

    def test_choose_fallback_none_when_no_tools(self):
        assert _choose_fallback_tool(failed_step={"title": "x"}, available_tools=[]) is None


class TestHeuristic:
    def test_transient_attempts_one_retries(self):
        failed = _step(error_type="transient", attempts=1)
        remaining = [{"title": "next"}]
        out = run_heuristic(failed_step=failed, remaining_plan=remaining, completed_steps=[], available_tools=[])
        assert out["recovery_strategy"] == "retry"
        assert out["revised_plan"][0]["title"] == failed["title"]
        assert out["revised_plan"][0]["attempts"] == 0

    def test_transient_attempts_three_not_retry(self):
        failed = _step(error_type="transient", attempts=3)
        out = run_heuristic(failed_step=failed, remaining_plan=[], completed_steps=[], available_tools=[])
        assert out["recovery_strategy"] == "escalate"

    def test_not_found_skips(self):
        remaining = [{"title": "keep going"}]
        out = run_heuristic(
            failed_step=_step(error_type="not_found", attempts=1),
            remaining_plan=remaining,
            completed_steps=[],
            available_tools=[],
        )
        assert out["recovery_strategy"] == "skip"
        assert out["revised_plan"] == remaining

    def test_permission_denied_skips(self):
        out = run_heuristic(
            failed_step=_step(error_type="permission_denied", attempts=2),
            remaining_plan=[],
            completed_steps=[],
            available_tools=[],
        )
        assert out["recovery_strategy"] == "skip"

    def test_service_unavailable_substitutes(self):
        out = run_heuristic(
            failed_step=_step(error_type="service_unavailable", attempts=1),
            remaining_plan=[],
            completed_steps=[],
            available_tools=["backup_tool"],
        )
        assert out["recovery_strategy"] == "substitute"
        assert out["substitute_step"] is not None

    def test_data_corruption_replans(self):
        out = run_heuristic(
            failed_step=_step(error_type="data_corruption", attempts=1),
            remaining_plan=[{"title": "next"}],
            completed_steps=[],
            available_tools=[],
        )
        assert out["recovery_strategy"] == "replan"
        assert out["revised_plan"] == []

    def test_unknown_attempts_five_escalates(self):
        out = run_heuristic(
            failed_step=_step(error_type="weird_failure", attempts=5),
            remaining_plan=[{"title": "next"}],
            completed_steps=[],
            available_tools=[],
        )
        assert out["recovery_strategy"] == "escalate"

    def test_confidence_retry(self):
        out = run_heuristic(failed_step=_step(error_type="transient", attempts=1), remaining_plan=[], completed_steps=[], available_tools=[])
        assert out["confidence"] == 0.8

    def test_confidence_skip(self):
        out = run_heuristic(failed_step=_step(error_type="not_found", attempts=1), remaining_plan=[], completed_steps=[], available_tools=[])
        assert out["confidence"] == 0.75

    def test_confidence_substitute(self):
        out = run_heuristic(
            failed_step=_step(error_type="service_unavailable", attempts=1),
            remaining_plan=[],
            completed_steps=[],
            available_tools=["fallback"],
        )
        assert out["confidence"] == 0.65

    def test_confidence_replan(self):
        out = run_heuristic(failed_step=_step(error_type="data_corruption", attempts=1), remaining_plan=[], completed_steps=[], available_tools=[])
        assert out["confidence"] == 0.5

    def test_confidence_escalate(self):
        out = run_heuristic(failed_step=_step(error_type="unknown", attempts=4), remaining_plan=[], completed_steps=[], available_tools=[])
        assert out["confidence"] == 0.4

    def test_substitute_uses_first_available_tool(self):
        out = run_heuristic(
            failed_step=_step(error_type="service_unavailable", title="use primary"),
            remaining_plan=[],
            completed_steps=[],
            available_tools=["fallback_a", "fallback_b"],
        )
        assert out["substitute_step"]["tool"] == "fallback_a"

    def test_skip_justification_contains_error_type(self):
        out = run_heuristic(failed_step=_step(error_type="not_found", attempts=1), remaining_plan=[], completed_steps=[], available_tools=[])
        assert "not_found" in (out["skip_justification"] or "")

    def test_escalation_reason_contains_attempts(self):
        out = run_heuristic(failed_step=_step(error_type="unknown", attempts=4), remaining_plan=[], completed_steps=[], available_tools=[])
        assert "4 attempts" in (out["escalation_reason"] or "")

    def test_retry_keeps_remaining_steps_after_failed_step(self):
        remaining = [{"title": "a"}, {"title": "b"}]
        out = run_heuristic(failed_step=_step(error_type="transient", attempts=2), remaining_plan=remaining, completed_steps=[], available_tools=[])
        assert out["revised_plan"][1:] == remaining

    def test_substitute_prepends_remaining_plan(self):
        remaining = [{"title": "next"}]
        out = run_heuristic(
            failed_step=_step(error_type="service_unavailable", attempts=1),
            remaining_plan=remaining,
            completed_steps=[],
            available_tools=["fallback"],
        )
        assert out["revised_plan"][1:] == remaining

    def test_default_unknown_escalates(self):
        out = run_heuristic(failed_step=_step(error_type="odd", attempts=1), remaining_plan=[], completed_steps=[], available_tools=[])
        assert out["recovery_strategy"] == "escalate"

    def test_completed_step_count_reported(self):
        out = run_heuristic(
            failed_step=_step(error_type="odd", attempts=1),
            remaining_plan=[],
            completed_steps=[{"title": "done1"}, {"title": "done2"}],
            available_tools=[],
        )
        assert out["completed_step_count"] == 2


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_run_heuristic(self):
        agent = ErrorRecoveryAgent()
        with patch("app.agents.error_recovery_agent.settings") as mock_settings:
            mock_settings.ERROR_RECOVERY_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(failed_step=_step(error_type="odd", attempts=1)))
        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        agent = ErrorRecoveryAgent()
        with patch("app.agents.error_recovery_agent.settings") as mock_settings:
            mock_settings.ERROR_RECOVERY_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(failed_step=_step(error_type="odd", attempts=1)))
        assert result.trace_id == "trace-68"

    @pytest.mark.asyncio
    async def test_strategy_fallback(self):
        agent = ErrorRecoveryAgent()
        with patch("app.agents.error_recovery_agent.settings") as mock_settings:
            mock_settings.ERROR_RECOVERY_STRATEGY = None
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("m1", _ctx(failed_step=_step(error_type="odd", attempts=1)))
        assert result.outputs["rationale"] == "heuristic"

    @pytest.mark.asyncio
    async def test_valid_llm_response_used(self):
        agent = ErrorRecoveryAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(
            return_value={
                "recovery_strategy": "retry",
                "revised_plan": [],
                "substitute_step": None,
                "skip_justification": None,
                "escalation_reason": None,
                "confidence": 0.8,
                "rationale": "llm",
            }
        )
        with patch("app.agents.error_recovery_agent.settings") as mock_settings, patch(
            "app.agents.error_recovery_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.ERROR_RECOVERY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(failed_step=_step(error_type="odd", attempts=1)))
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_invalid_llm_falls_back(self):
        agent = ErrorRecoveryAgent()
        mock_client = MagicMock()
        mock_client.complete_json = AsyncMock(return_value={"bad": "shape"})
        with patch("app.agents.error_recovery_agent.settings") as mock_settings, patch(
            "app.agents.error_recovery_agent.create_ollama_client", return_value=mock_client
        ):
            mock_settings.ERROR_RECOVERY_STRATEGY = "llm"
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
            mock_settings.OLLAMA_MAX_CONCURRENCY = 2
            mock_settings.get_ollama_model = lambda _x: "qwen2.5:7b"
            result = await agent.run("m1", _ctx(failed_step=_step(error_type="odd", attempts=1)))
        assert result.outputs["rationale"] == "heuristic"


class TestValidateOutputs:
    def _result(self, outputs: dict) -> AgentResult:
        now = datetime.now(timezone.utc)
        return AgentResult(
            agent_name="ErrorRecoveryAgent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.5,
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=now,
            finished_at=now,
        )

    def _valid_outputs(self) -> dict:
        return {
            "recovery_strategy": "retry",
            "revised_plan": [],
            "substitute_step": None,
            "skip_justification": None,
            "escalation_reason": None,
            "confidence": 0.8,
        }

    def test_validate_outputs_passes(self):
        ErrorRecoveryAgent().validate_outputs(self._result(self._valid_outputs()))

    def test_strategy_raises(self):
        with pytest.raises(ValueError, match="recovery_strategy"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), recovery_strategy="bad")))

    def test_revised_plan_type_raises(self):
        with pytest.raises(ValueError, match="revised_plan"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), revised_plan="x")))

    def test_substitute_step_type_raises(self):
        with pytest.raises(ValueError, match="substitute_step"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), substitute_step="x")))

    def test_skip_justification_type_raises(self):
        with pytest.raises(ValueError, match="skip_justification"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), skip_justification=123)))

    def test_escalation_reason_type_raises(self):
        with pytest.raises(ValueError, match="escalation_reason"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), escalation_reason=123)))

    def test_confidence_type_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), confidence="x")))

    def test_confidence_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            ErrorRecoveryAgent().validate_outputs(self._result(dict(self._valid_outputs(), confidence=1.2)))

    def test_failed_status_skips_validation(self):
        now = datetime.now(timezone.utc)
        res = AgentResult(
            agent_name="ErrorRecoveryAgent",
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
        ErrorRecoveryAgent().validate_outputs(res)


class TestRegistry:
    def test_alias_primary(self):
        assert isinstance(get_agent("error_recovery"), ErrorRecoveryAgent)

    def test_alias_compact(self):
        assert isinstance(get_agent("errorrecovery"), ErrorRecoveryAgent)

    def test_alias_agent(self):
        assert isinstance(get_agent("ErrorRecoveryAgent"), ErrorRecoveryAgent)

    def test_alias_short(self):
        assert isinstance(get_agent("replan"), ErrorRecoveryAgent)
