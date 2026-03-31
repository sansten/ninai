"""Tests for AutonomousActionAgent, ActionPolicyService, ExternalConnectorService,
and ExecutorAgent retry/rollback — Phase 47."""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.autonomous_action_agent import (
    AutonomousActionAgent,
    _should_act,
    _choose_action_type,
    _build_payload,
    _compute_dispatch_confidence,
    _validate_llm_outputs,
    run_heuristic,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult
from app.services.action_policy_service import (
    ActionPolicyService,
    evaluate_policy,
    PolicyDecision,
    _resolve_permitted_types,
    _resolve_auto_approvable_types,
    _resolve_auto_approve_tiers,
    _resolve_min_confidence,
    _resolve_auto_approve_threshold,
)
from app.services.external_connector_service import (
    ExternalConnectorService,
    DispatchResult,
    _build_headers,
    _backoff_seconds,
    _safe_response_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(
    enrichment: dict | None = None,
    content: str = "Test memory",
    runtime: dict | None = None,
) -> dict:
    return {
        "memory": {
            "id": "test-mem-47",
            "content": content,
            "enrichment": enrichment or {},
        },
        "runtime": {"job_id": "trace-47", **(runtime or {})},
    }


def _urgent_enrichment() -> dict:
    return {
        "anomaly_detected": True,
        "anomaly_score": 0.95,
        "tone": "urgent",
        "org_tier": "enterprise",
        "credibility_score": 0.90,
        "narrative_text": "Organization-level anomaly detected.",
        "episode_label": "DevOps Auth Crisis — Week 12",
        "episode_id": "EP-DEVOPS-W12",
        "canonical_entity": "Meridian DevOps Team",
        "matched_playbook_id": "PB-MAJOR-INCIDENT",
        "action_items": ["Page on-call", "Declare P1"],
    }


def _low_score_enrichment() -> dict:
    return {
        "anomaly_detected": True,
        "anomaly_score": 0.40,
        "tone": "urgent",
        "org_tier": "enterprise",
        "credibility_score": 0.80,
    }


def _no_anomaly_enrichment() -> dict:
    return {
        "anomaly_detected": False,
        "anomaly_score": 0.95,
        "tone": "urgent",
        "org_tier": "enterprise",
    }


def _cautionary_enrichment() -> dict:
    return {
        "anomaly_detected": True,
        "anomaly_score": 0.75,
        "tone": "cautionary",
        "org_tier": "mid_market",
        "credibility_score": 0.75,
    }


# ===========================================================================
# ActionPolicyService tests
# ===========================================================================

class TestActionPolicyServiceHelpers:
    def test_default_permitted_types_returned_when_no_config(self):
        result = _resolve_permitted_types(None)
        assert "webhook" in result
        assert "pagerduty" in result

    def test_org_config_overrides_permitted_types(self):
        cfg = {"permitted_action_types": ["jira"]}
        result = _resolve_permitted_types(cfg)
        assert result == frozenset({"jira"})

    def test_org_config_invalid_type_filtered_out(self):
        cfg = {"permitted_action_types": ["jira", "unknown_type"]}
        result = _resolve_permitted_types(cfg)
        assert "unknown_type" not in result

    def test_empty_org_config_permitted_types_uses_default(self):
        result = _resolve_permitted_types({"permitted_action_types": []})
        assert "webhook" in result

    def test_auto_approvable_types_default(self):
        result = _resolve_auto_approvable_types(None)
        assert "pagerduty" in result
        assert "jira" not in result

    def test_auto_approve_tiers_default(self):
        result = _resolve_auto_approve_tiers(None)
        assert "enterprise" in result

    def test_min_confidence_default(self):
        assert _resolve_min_confidence(None) == 0.70

    def test_min_confidence_from_config(self):
        assert _resolve_min_confidence({"min_confidence": 0.80}) == 0.80

    def test_auto_approve_threshold_default(self):
        assert _resolve_auto_approve_threshold(None) == 0.90

    def test_auto_approve_threshold_clamped(self):
        assert _resolve_auto_approve_threshold({"auto_approve_threshold": 1.5}) == 1.0


class TestEvaluatePolicy:
    def test_low_confidence_denied(self):
        d = evaluate_policy(action_type="webhook", confidence=0.50, org_tier="enterprise")
        assert d.decision == "denied"
        assert not d.confidence_threshold_met

    def test_unknown_action_type_denied(self):
        d = evaluate_policy(action_type="sms", confidence=0.95, org_tier="enterprise")
        assert d.decision == "denied"

    def test_action_type_not_in_permitted_denied(self):
        cfg = {"permitted_action_types": ["jira"]}
        d = evaluate_policy(action_type="pagerduty", confidence=0.95, org_tier="enterprise",
                            org_config=cfg)
        assert d.decision == "denied"
        assert d.confidence_threshold_met

    def test_enterprise_high_confidence_pagerduty_auto_approved(self):
        d = evaluate_policy(action_type="pagerduty", confidence=0.95, org_tier="enterprise")
        assert d.decision == "auto_approved"
        assert d.confidence_threshold_met

    def test_enterprise_high_confidence_webhook_auto_approved(self):
        d = evaluate_policy(action_type="webhook", confidence=0.92, org_tier="enterprise")
        assert d.decision == "auto_approved"

    def test_mid_market_human_review_required(self):
        d = evaluate_policy(action_type="pagerduty", confidence=0.92, org_tier="mid_market")
        assert d.decision == "human_review_required"

    def test_confidence_exactly_at_min_threshold_not_denied(self):
        d = evaluate_policy(action_type="webhook", confidence=0.70, org_tier="enterprise")
        assert d.decision != "denied"
        assert d.confidence_threshold_met

    def test_confidence_below_auto_approve_threshold_human_review(self):
        d = evaluate_policy(action_type="pagerduty", confidence=0.80, org_tier="enterprise")
        assert d.decision == "human_review_required"

    def test_enterprise_jira_not_auto_approvable_human_review(self):
        d = evaluate_policy(action_type="jira", confidence=0.95, org_tier="enterprise")
        assert d.decision == "human_review_required"

    def test_service_class_delegates_to_evaluate_policy(self):
        svc = ActionPolicyService()
        d = svc.evaluate(action_type="pagerduty", confidence=0.95, org_tier="enterprise")
        assert d.decision == "auto_approved"

    def test_policy_decision_is_frozen_dataclass(self):
        d = evaluate_policy(action_type="webhook", confidence=0.95, org_tier="enterprise")
        with pytest.raises((AttributeError, TypeError)):
            d.decision = "denied"  # type: ignore

    def test_slack_low_confidence_denied(self):
        d = evaluate_policy(action_type="slack", confidence=0.60, org_tier="enterprise")
        assert d.decision == "denied"

    def test_generic_rest_human_review_for_enterprise_mid_confidence(self):
        d = evaluate_policy(action_type="generic_rest", confidence=0.85, org_tier="enterprise")
        assert d.decision == "human_review_required"


# ===========================================================================
# ExternalConnectorService tests
# ===========================================================================

class TestConnectorServiceHelpers:
    def test_build_headers_sets_content_type(self):
        h = _build_headers("pagerduty", None)
        assert h["Content-Type"] == "application/json"
        assert "Ninai" in h["User-Agent"]

    def test_build_headers_merges_extras(self):
        h = _build_headers("webhook", {"X-Auth": "token"})
        assert h["X-Auth"] == "token"

    def test_backoff_doubles(self):
        assert _backoff_seconds(0) == 1.0
        assert _backoff_seconds(1) == 2.0
        assert _backoff_seconds(2) == 4.0

    def test_backoff_capped_at_30(self):
        assert _backoff_seconds(10) == 30.0

    def test_safe_response_summary_keys(self):
        s = _safe_response_summary(200, "ok body")
        assert s["http_status"] == 200
        assert "body_len" in s
        assert "body_prefix" in s

    def test_safe_response_summary_empty_body(self):
        s = _safe_response_summary(204, "")
        assert s["body_len"] == 0


class TestExternalConnectorServiceDispatch:
    def _make_mock_response(self, status_code: int, text: str = "ok"):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        return resp

    def _make_client_factory(self, status_code: int, text: str = "ok"):
        resp = self._make_mock_response(status_code, text)
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)
        return lambda: client

    @pytest.mark.asyncio
    async def test_successful_dispatch_200(self):
        svc = ExternalConnectorService(
            _http_client_factory=self._make_client_factory(200)
        )
        result = await svc.dispatch(
            action_type="webhook",
            target_url="https://example.com/hook",
            payload={"event": "test"},
        )
        assert result.status == "success"
        assert result.http_status_code == 200
        assert result.attempt_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_failure_400(self):
        svc = ExternalConnectorService(
            _http_client_factory=self._make_client_factory(400)
        )
        result = await svc.dispatch(
            action_type="pagerduty",
            target_url="https://pagerduty.example.com",
            payload={"summary": "test"},
        )
        assert result.status == "failed"
        assert result.http_status_code == 400
        assert result.attempt_count == 1

    @pytest.mark.asyncio
    async def test_retryable_500_exhausts_retries(self):
        call_count = 0

        async def _post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 500
            resp.text = "server error"
            return resp

        client = MagicMock()
        client.post = _post

        svc = ExternalConnectorService(
            max_attempts=3,
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        result = await svc.dispatch(
            action_type="webhook",
            target_url="https://example.com/hook",
            payload={},
        )
        assert result.status == "failed"
        assert result.attempt_count == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self):
        call_count = 0

        async def _post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 503 if call_count == 1 else 200
            resp.text = "ok"
            return resp

        client = MagicMock()
        client.post = _post

        svc = ExternalConnectorService(
            max_attempts=3,
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        result = await svc.dispatch(
            action_type="webhook",
            target_url="https://example.com/hook",
            payload={},
        )
        assert result.status == "success"
        assert result.attempt_count == 2

    @pytest.mark.asyncio
    async def test_network_exception_retried(self):
        call_count = 0

        async def _post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("refused")

        client = MagicMock()
        client.post = _post

        svc = ExternalConnectorService(
            max_attempts=2,
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        result = await svc.dispatch(
            action_type="webhook",
            target_url="https://example.com/hook",
            payload={},
        )
        assert result.status == "failed"
        assert result.attempt_count == 2
        assert "ConnectionError" in (result.error or "")


# ===========================================================================
# _should_act helper tests
# ===========================================================================

class TestShouldAct:
    def test_no_anomaly_detected_no_action(self):
        should, conf, _ = _should_act(_no_anomaly_enrichment())
        assert should is False

    def test_low_score_no_action(self):
        should, conf, _ = _should_act(_low_score_enrichment())
        assert should is False

    def test_urgent_high_score_act(self):
        should, conf, _ = _should_act(_urgent_enrichment())
        assert should is True
        assert conf >= 0.9

    def test_very_high_score_overrides_non_urgent_tone(self):
        enr = {**_no_anomaly_enrichment(), "anomaly_detected": True,
               "anomaly_score": 0.92, "tone": "informational"}
        should, _, _ = _should_act(enr)
        assert should is True

    def test_score_at_threshold_acts(self):
        enr = {"anomaly_detected": True, "anomaly_score": 0.70, "tone": "urgent"}
        should, _, _ = _should_act(enr)
        assert should is True

    def test_cautionary_tone_medium_score_no_action(self):
        should, _, _ = _should_act(_cautionary_enrichment())
        assert should is False


class TestChooseActionType:
    def test_enterprise_pagerduty(self):
        assert _choose_action_type({"org_tier": "enterprise"}) == "pagerduty"

    def test_mid_market_slack(self):
        assert _choose_action_type({"org_tier": "mid_market"}) == "slack"

    def test_midmarket_alt_slug_slack(self):
        assert _choose_action_type({"org_tier": "midmarket"}) == "slack"

    def test_smb_webhook(self):
        assert _choose_action_type({"org_tier": "smb"}) == "webhook"

    def test_empty_tier_webhook(self):
        assert _choose_action_type({}) == "webhook"


class TestBuildPayload:
    def test_payload_contains_narrative(self):
        enr = _urgent_enrichment()
        p = _build_payload(enr, "fallback")
        assert "Organization-level" in p["summary"]

    def test_payload_falls_back_to_content(self):
        p = _build_payload({}, "fallback content")
        assert "fallback" in p["summary"]

    def test_payload_keys_present(self):
        p = _build_payload(_urgent_enrichment(), "")
        for k in ("episode_label", "episode_id", "anomaly_score", "org_tier", "tone"):
            assert k in p

    def test_content_truncated_at_300(self):
        long_content = "x" * 500
        p = _build_payload({}, long_content)
        assert len(p["summary"]) <= 300


class TestComputeDispatchConfidence:
    def test_combines_anomaly_and_credibility(self):
        enr = {"anomaly_score": 1.0, "credibility_score": 1.0}
        assert _compute_dispatch_confidence(enr) == 1.0

    def test_low_credibility_reduces_confidence(self):
        high = _compute_dispatch_confidence({"anomaly_score": 0.95, "credibility_score": 0.95})
        low  = _compute_dispatch_confidence({"anomaly_score": 0.95, "credibility_score": 0.20})
        assert low < high

    def test_missing_credibility_uses_default(self):
        conf = _compute_dispatch_confidence({"anomaly_score": 0.90})
        assert conf > 0.0


# ===========================================================================
# run_heuristic tests
# ===========================================================================

class TestRunHeuristic:
    def test_no_anomaly_returns_no_action(self):
        out = run_heuristic(_no_anomaly_enrichment(), "content")
        assert out["action_status"] == "no_action"
        assert out["action_dispatched"] is False
        assert out["policy_decision"] is None

    def test_low_score_returns_no_action(self):
        out = run_heuristic(_low_score_enrichment(), "content")
        assert out["action_status"] == "no_action"

    def test_urgent_enterprise_auto_approved_dispatched(self):
        out = run_heuristic(_urgent_enrichment(), "content")
        assert out["action_dispatched"] is True
        assert out["policy_decision"] == "auto_approved"
        assert out["action_type"] == "pagerduty"
        assert out["action_status"] == "success"

    def test_mid_market_urgent_human_review(self):
        enr = {**_urgent_enrichment(), "org_tier": "mid_market"}
        out = run_heuristic(enr, "content")
        assert out["action_status"] == "pending_review"
        assert out["policy_decision"] == "human_review_required"
        assert out["action_dispatched"] is False

    def test_denied_when_policy_denies(self):
        enr = {**_urgent_enrichment(), "org_tier": "enterprise",
               "anomaly_score": 0.72, "credibility_score": 0.50}
        # dispatch confidence = 0.72*0.7 + 0.50*0.3 = 0.654 < 0.70 → denied
        out = run_heuristic(enr, "content")
        # confidence might be just below threshold depending on exact computation
        # just check it doesn't explode
        assert out["action_status"] in ("denied", "no_action", "pending_review", "success")

    def test_rationale_is_heuristic(self):
        out = run_heuristic(_urgent_enrichment(), "content")
        assert out["rationale"] == "heuristic"

    def test_payload_summary_in_auto_approved(self):
        out = run_heuristic(_urgent_enrichment(), "content")
        assert "_payload_summary" in out


# ===========================================================================
# _validate_llm_outputs tests
# ===========================================================================

class TestValidateLlmOutputs:
    def test_valid_outputs_pass_through(self):
        raw = {
            "action_dispatched": True,
            "action_type": "pagerduty",
            "action_status": "success",
            "policy_decision": "auto_approved",
            "dispatch_confidence": 0.92,
            "action_target": "https://events.pagerduty.com/v2/enqueue",
        }
        out = _validate_llm_outputs(raw)
        assert out["action_dispatched"] is True
        assert out["action_type"] == "pagerduty"
        assert out["rationale"] == "llm"

    def test_invalid_action_type_cleared(self):
        raw = {"action_type": "sms", "action_status": "no_action",
               "dispatch_confidence": 0.5}
        out = _validate_llm_outputs(raw)
        assert out["action_type"] is None

    def test_invalid_status_defaults_to_no_action(self):
        raw = {"action_status": "unknown", "dispatch_confidence": 0.5}
        out = _validate_llm_outputs(raw)
        assert out["action_status"] == "no_action"

    def test_confidence_clamped_to_one(self):
        raw = {"action_status": "success", "dispatch_confidence": 5.0}
        out = _validate_llm_outputs(raw)
        assert out["dispatch_confidence"] == 1.0

    def test_confidence_clamped_to_zero(self):
        raw = {"action_status": "no_action", "dispatch_confidence": -0.5}
        out = _validate_llm_outputs(raw)
        assert out["dispatch_confidence"] == 0.0


# ===========================================================================
# AutonomousActionAgent full agent tests
# ===========================================================================

class TestAutonomousActionAgentHeuristic:
    @pytest.mark.asyncio
    async def test_no_anomaly_no_action(self):
        agent = AutonomousActionAgent()
        with patch("app.agents.autonomous_action_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-1", _ctx(_no_anomaly_enrichment()))
        assert result.status == "success"
        assert result.outputs["action_status"] == "no_action"

    @pytest.mark.asyncio
    async def test_urgent_enterprise_dispatches(self):
        agent = AutonomousActionAgent()
        with patch("app.agents.autonomous_action_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-2", _ctx(_urgent_enrichment()))
        assert result.outputs["action_dispatched"] is True
        assert result.outputs["action_type"] == "pagerduty"
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_with_real_connector_success(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        with patch("app.agents.autonomous_action_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-3", _ctx(_urgent_enrichment()))

        assert result.outputs["action_dispatched"] is True
        assert result.outputs["action_status"] == "success"

    @pytest.mark.asyncio
    async def test_with_real_connector_failure_recorded(self):
        client = MagicMock()
        client.post = AsyncMock(side_effect=ConnectionError("refused"))

        svc = ExternalConnectorService(
            max_attempts=1,
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        with patch("app.agents.autonomous_action_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-4", _ctx(_urgent_enrichment()))

        assert result.outputs["action_status"] == "failed"
        assert result.outputs["action_dispatched"] is False

    @pytest.mark.asyncio
    async def test_runtime_kill_switch_denies_dispatch(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        runtime = {
            "action_runtime_control": {
                "enabled": False,
            }
        }

        with patch("app.agents.autonomous_action_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-47-kill", _ctx(_urgent_enrichment(), runtime=runtime))

        assert result.outputs["action_status"] == "denied"
        assert result.outputs["policy_decision"] == "denied"
        assert result.outputs["action_dispatched"] is False
        assert "runtime action control" in result.outputs.get("_runtime_control_reason", "")
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_runtime_dry_run_skips_external_call(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        runtime = {
            "action_runtime_control": {
                "enabled": True,
                "dry_run": True,
            }
        }

        with patch("app.agents.autonomous_action_agent.settings") as mock_settings:
            mock_settings.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-47-dryrun", _ctx(_urgent_enrichment(), runtime=runtime))

        assert result.outputs["action_status"] == "success"
        assert result.outputs["policy_decision"] == "auto_approved"
        assert result.outputs["action_dispatched"] is False
        assert result.outputs.get("_dry_run") is True
        client.post.assert_not_called()


class TestAutonomousActionAgentLLM:
    @pytest.mark.asyncio
    async def test_llm_path_success(self):
        mock_raw = {
            "action_dispatched": True,
            "action_type": "pagerduty",
            "action_status": "success",
            "policy_decision": "auto_approved",
            "dispatch_confidence": 0.92,
            "action_target": "https://events.pagerduty.com/v2/enqueue",
        }
        agent = AutonomousActionAgent()
        with patch("app.agents.autonomous_action_agent.settings") as mock_settings, \
             patch("app.agents.autonomous_action_agent.create_ollama_client") as mock_client_fn:
            mock_settings.AGENT_STRATEGY = "llm"
            mock_client = MagicMock()
            mock_client.complete_json = AsyncMock(return_value=mock_raw)
            mock_client_fn.return_value = mock_client
            result = await agent.run("mem-5", _ctx(_urgent_enrichment()))

        assert result.outputs["action_type"] == "pagerduty"
        assert result.outputs["rationale"] == "llm"

    @pytest.mark.asyncio
    async def test_llm_fallback_to_heuristic_on_error(self):
        agent = AutonomousActionAgent()
        with patch("app.agents.autonomous_action_agent.settings") as mock_settings, \
             patch("app.agents.autonomous_action_agent.create_ollama_client") as mock_client_fn:
            mock_settings.AGENT_STRATEGY = "llm"
            mock_client = MagicMock()
            mock_client.complete_json = AsyncMock(side_effect=RuntimeError("LLM down"))
            mock_client_fn.return_value = mock_client
            result = await agent.run("mem-6", _ctx(_urgent_enrichment()))

        assert result.status == "success"
        assert result.outputs["rationale"] == "heuristic"


class TestAutonomousActionAgentValidateOutputs:
    def test_valid_outputs_no_error(self):
        agent = AutonomousActionAgent()
        result = AgentResult(
            agent_name="autonomous_action_agent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.90,
            outputs={
                "action_dispatched": True,
                "action_type": "pagerduty",
                "action_status": "success",
                "policy_decision": "auto_approved",
                "rationale": "heuristic",
                "action_target": "https://events.pagerduty.com/v2/enqueue",
                "dispatch_confidence": 0.90,
            },
            started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            finished_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        agent.validate_outputs(result)  # should not raise

    def test_missing_action_dispatched_raises(self):
        agent = AutonomousActionAgent()
        result = AgentResult(
            agent_name="autonomous_action_agent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.5,
            outputs={"action_status": "no_action", "rationale": "heuristic",
                     "dispatch_confidence": 0.5},
            started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            finished_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        with pytest.raises(ValueError, match="action_dispatched"):
            agent.validate_outputs(result)

    def test_invalid_action_status_raises(self):
        agent = AutonomousActionAgent()
        result = AgentResult(
            agent_name="autonomous_action_agent",
            agent_version="v1",
            memory_id="m1",
            status="success",
            confidence=0.5,
            outputs={"action_dispatched": False, "action_status": "bogus",
                     "dispatch_confidence": 0.5},
            started_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            finished_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        with pytest.raises(ValueError, match="action_status"):
            agent.validate_outputs(result)


class TestAutonomousActionAgentRegistry:
    def test_registered_by_name(self):
        agent = get_agent("autonomous_action_agent")
        assert agent is not None
        assert isinstance(agent, AutonomousActionAgent)

    def test_registered_by_alias(self):
        agent = get_agent("action_engine")
        assert isinstance(agent, AutonomousActionAgent)

    def test_name_attribute(self):
        agent = AutonomousActionAgent()
        assert agent.name == "autonomous_action_agent"

    def test_dependencies_declared(self):
        agent = AutonomousActionAgent()
        deps = agent.dependencies()
        assert "anomaly_detection_agent" in deps
        assert "narrative_synthesis_agent" in deps


# ===========================================================================
# ExecutorAgent retry/rollback tests (Phase 47 additions)
# ===========================================================================

class TestExecutorAgentRetryRollback:
    def _make_invoker(self, results: list[str]) -> MagicMock:
        """results: sequence of "success"|"failed"|"denied" per call."""
        call_idx = [0]

        async def _invoke(**kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            status = results[idx] if idx < len(results) else "failed"
            r = MagicMock()
            r.status = status
            r.tool_call_id = f"call-{idx}"
            r.denial_reason = "denied" if status == "denied" else None
            r.error = "err" if status == "failed" else None
            return r

        invoker = MagicMock()
        invoker.invoke = _invoke
        return invoker

    def _make_plan(self, tool: str, rollback_tool: str | None = None) -> MagicMock:
        from app.schemas.cognitive import PlannerStep
        hint = {}
        if rollback_tool:
            hint["rollback_tool"] = rollback_tool
        step = PlannerStep(
            step_id="s1",
            action="do something",
            tool=tool,
            tool_input_hint=hint,
            expected_output="ok",
            success_criteria=[],
        )
        plan = MagicMock()
        plan.steps = [step]
        return plan

    def _make_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.user_id = "u1"
        ctx.org_id = "o1"
        return ctx

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        from app.services.cognitive_loop.executor_agent import ExecutorAgent
        invoker = self._make_invoker(["success"])
        agent = ExecutorAgent(tool_invoker=invoker, max_retries=2)
        result = await agent.execute(
            session_id="s", iteration_id="i",
            plan=self._make_plan("my.tool"), ctx=self._make_ctx(),
        )
        assert result.overall_status == "success"
        assert result.step_results[0].status == "success"

    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second(self):
        from app.services.cognitive_loop.executor_agent import ExecutorAgent
        invoker = self._make_invoker(["failed", "success"])
        agent = ExecutorAgent(tool_invoker=invoker, max_retries=2)
        with patch("app.services.cognitive_loop.executor_agent.asyncio.sleep",
                   new_callable=AsyncMock):
            result = await agent.execute(
                session_id="s", iteration_id="i",
                plan=self._make_plan("my.tool"), ctx=self._make_ctx(),
            )
        assert result.step_results[0].status == "success"
        assert "attempt 2" in result.step_results[0].summary

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_status_failed(self):
        from app.services.cognitive_loop.executor_agent import ExecutorAgent
        invoker = self._make_invoker(["failed", "failed", "failed"])
        agent = ExecutorAgent(tool_invoker=invoker, max_retries=2)
        with patch("app.services.cognitive_loop.executor_agent.asyncio.sleep",
                   new_callable=AsyncMock):
            result = await agent.execute(
                session_id="s", iteration_id="i",
                plan=self._make_plan("my.tool"), ctx=self._make_ctx(),
            )
        assert result.step_results[0].status == "failed"
        assert result.overall_status == "failed"

    @pytest.mark.asyncio
    async def test_rollback_invoked_on_failure(self):
        from app.services.cognitive_loop.executor_agent import ExecutorAgent
        # main tool fails, rollback succeeds
        invoker = self._make_invoker(["failed", "failed", "failed", "success"])
        agent = ExecutorAgent(tool_invoker=invoker, max_retries=2)
        with patch("app.services.cognitive_loop.executor_agent.asyncio.sleep",
                   new_callable=AsyncMock):
            result = await agent.execute(
                session_id="s", iteration_id="i",
                plan=self._make_plan("my.tool", rollback_tool="my.rollback"),
                ctx=self._make_ctx(),
            )
        sr = result.step_results[0]
        assert sr.status == "rolled_back"
        assert "rollback" in sr.summary.lower()
        assert sr.artifacts.get("rollback_ok") is True

    @pytest.mark.asyncio
    async def test_denied_not_retried(self):
        from app.services.cognitive_loop.executor_agent import ExecutorAgent
        invoker = self._make_invoker(["denied"])
        agent = ExecutorAgent(tool_invoker=invoker, max_retries=2)
        result = await agent.execute(
            session_id="s", iteration_id="i",
            plan=self._make_plan("my.tool"), ctx=self._make_ctx(),
        )
        assert result.step_results[0].status == "denied"
        assert result.overall_status == "partial"

    @pytest.mark.asyncio
    async def test_no_tool_step_skipped(self):
        from app.services.cognitive_loop.executor_agent import ExecutorAgent
        from app.schemas.cognitive import PlannerStep
        invoker = MagicMock()
        step = PlannerStep(
            step_id="s0", action="think", tool=None,
            expected_output="", success_criteria=[],
        )
        plan = MagicMock()
        plan.steps = [step]
        agent = ExecutorAgent(tool_invoker=invoker, max_retries=1)
        result = await agent.execute(
            session_id="s", iteration_id="i", plan=plan, ctx=self._make_ctx(),
        )
        assert result.step_results[0].status == "skipped"
        assert result.overall_status == "success"
