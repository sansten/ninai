"""Tests for ConnectorObservabilityService, ActionKillSwitchService,
and their wiring in AutonomousActionAgent — Phase 51 Slice 4."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("ninai_enterprise", reason="Enterprise-only autonomous action agent tests")

from app.agents.autonomous_action_agent import AutonomousActionAgent
from app.services.connector_observability_service import (
    ConnectorObservabilityService,
    ConnectorMetrics,
    DispatchEvent,
)
from app.services.action_kill_switch_service import (
    ActionKillSwitchService,
    KillSwitchState,
)
from app.services.external_connector_service import ExternalConnectorService, RetryClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    enrichment: dict | None = None,
    content: str = "Test memory",
    runtime: dict | None = None,
    org_id: str = "org-obs-1",
) -> dict:
    return {
        "tenant": {"org_id": org_id},
        "memory": {
            "id": "test-mem-obs",
            "content": content,
            "enrichment": enrichment or {},
        },
        "runtime": {"job_id": "trace-obs", **(runtime or {})},
    }


def _urgent_enrichment() -> dict:
    return {
        "anomaly_detected": True,
        "anomaly_score": 0.95,
        "tone": "urgent",
        "org_tier": "enterprise",
        "credibility_score": 0.90,
        "narrative_text": "Organization-level anomaly detected.",
        "episode_label": "DevOps Auth Crisis",
        "episode_id": "EP-1",
        "canonical_entity": "Meridian DevOps",
        "matched_playbook_id": "PB-MAJOR",
        "action_items": ["Page on-call"],
    }


def _make_client(status_code: int, text: str = "ok"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    return client


# ===========================================================================
# ConnectorObservabilityService unit tests
# ===========================================================================

class TestConnectorObservabilityService:

    def _event(
        self,
        org_id: str = "org-1",
        connector_type: str = "pagerduty",
        status: str = "success",
        retry_class: str | None = None,
        rollback_policy: str | None = None,
        rollback_triggered: bool = False,
        attempt_count: int = 1,
        latency_ms: float | None = 50.0,
    ) -> DispatchEvent:
        return DispatchEvent(
            org_id=org_id,
            connector_type=connector_type,
            status=status,
            retry_class=retry_class,
            rollback_policy=rollback_policy,
            rollback_triggered=rollback_triggered,
            attempt_count=attempt_count,
            latency_ms=latency_ms,
        )

    def test_record_success_increments_total_and_success(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(status="success"))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m is not None
        assert m.total == 1
        assert m.success == 1
        assert m.failed == 0

    def test_record_failed_transient_increments_failed(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(status="failed", retry_class="transient"))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m.failed == 1
        assert m.throttled == 0
        assert m.last_error is not None

    def test_record_throttled_increments_throttled_counter(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(status="failed", retry_class="throttled"))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m.throttled == 1
        assert m.failed == 1

    def test_record_permanent_increments_permanent_error_counter(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(status="failed", retry_class="permanent"))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m.permanent_error == 1

    def test_record_rollback_increments_rollback_triggered(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(status="failed", rollback_triggered=True))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m.rollback_triggered == 1

    def test_multiple_events_accumulate(self):
        svc = ConnectorObservabilityService()
        for _ in range(3):
            svc.record(self._event(status="success"))
        svc.record(self._event(status="failed", retry_class="transient"))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m.total == 4
        assert m.success == 3
        assert m.failed == 1

    def test_success_rate_computed_correctly(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(status="success"))
        svc.record(self._event(status="success"))
        svc.record(self._event(status="failed"))
        m = svc.metrics_for("org-1", "pagerduty")
        assert abs(m.success_rate - 2 / 3) < 1e-9

    def test_avg_latency_computed_correctly(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(latency_ms=100.0))
        svc.record(self._event(latency_ms=200.0))
        m = svc.metrics_for("org-1", "pagerduty")
        assert m.avg_latency_ms == 150.0

    def test_metrics_for_unknown_key_returns_none(self):
        svc = ConnectorObservabilityService()
        assert svc.metrics_for("org-x", "webhook") is None

    def test_separate_orgs_have_separate_metrics(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(org_id="org-A", status="success"))
        svc.record(self._event(org_id="org-B", status="failed"))
        assert svc.metrics_for("org-A", "pagerduty").success == 1
        assert svc.metrics_for("org-B", "pagerduty").failed == 1

    def test_org_summary_returns_all_connector_types(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(connector_type="pagerduty"))
        svc.record(self._event(connector_type="webhook"))
        summary = svc.org_summary("org-1")
        types = {m.connector_type for m in summary}
        assert types == {"pagerduty", "webhook"}

    def test_org_summary_excludes_other_orgs(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(org_id="org-1"))
        svc.record(self._event(org_id="org-2"))
        summary = svc.org_summary("org-1")
        assert all(m.org_id == "org-1" for m in summary)

    def test_reset_clears_all(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event())
        svc.reset()
        assert svc.metrics_for("org-1", "pagerduty") is None

    def test_reset_by_org_clears_only_that_org(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event(org_id="org-A"))
        svc.record(self._event(org_id="org-B"))
        svc.reset(org_id="org-A")
        assert svc.metrics_for("org-A", "pagerduty") is None
        assert svc.metrics_for("org-B", "pagerduty") is not None

    def test_to_dict_contains_expected_keys(self):
        svc = ConnectorObservabilityService()
        svc.record(self._event())
        d = svc.metrics_for("org-1", "pagerduty").to_dict()
        for key in ("total", "success", "failed", "throttled", "success_rate", "avg_latency_ms"):
            assert key in d


# ===========================================================================
# ActionKillSwitchService unit tests
# ===========================================================================

class TestActionKillSwitchService:

    def test_default_state_is_permissive(self):
        svc = ActionKillSwitchService()
        ctrl = svc.build_runtime_control("org-unknown")
        assert ctrl["enabled"] is True
        assert ctrl["dry_run"] is False
        assert ctrl["disabled_action_types"] == []

    def test_disable_all_sets_enabled_false(self):
        svc = ActionKillSwitchService()
        svc.disable_all("org-1")
        assert svc.build_runtime_control("org-1")["enabled"] is False

    def test_enable_all_restores_enabled_true(self):
        svc = ActionKillSwitchService()
        svc.disable_all("org-1")
        svc.enable_all("org-1")
        assert svc.build_runtime_control("org-1")["enabled"] is True

    def test_set_dry_run_toggle(self):
        svc = ActionKillSwitchService()
        svc.set_dry_run("org-1", dry_run=True)
        ctrl = svc.build_runtime_control("org-1")
        assert ctrl["dry_run"] is True

    def test_disable_connector_type_added_to_list(self):
        svc = ActionKillSwitchService()
        svc.disable_connector_type("org-1", "pagerduty")
        ctrl = svc.build_runtime_control("org-1")
        assert "pagerduty" in ctrl["disabled_action_types"]

    def test_enable_connector_type_removes_from_list(self):
        svc = ActionKillSwitchService()
        svc.disable_connector_type("org-1", "pagerduty")
        svc.enable_connector_type("org-1", "pagerduty")
        ctrl = svc.build_runtime_control("org-1")
        assert "pagerduty" not in ctrl["disabled_action_types"]

    def test_disable_unknown_connector_type_returns_false(self):
        svc = ActionKillSwitchService()
        assert svc.disable_connector_type("org-1", "fax") is False

    def test_enable_nonexistent_connector_type_returns_false(self):
        svc = ActionKillSwitchService()
        assert svc.enable_connector_type("org-1", "pagerduty") is False

    def test_set_org_state_partial_update(self):
        svc = ActionKillSwitchService()
        svc.disable_all("org-1")
        svc.set_org_state("org-1", dry_run=True)  # enabled should stay False
        state = svc.get_org_state("org-1")
        assert state.enabled is False
        assert state.dry_run is True

    def test_set_org_state_filters_invalid_types(self):
        svc = ActionKillSwitchService()
        svc.set_org_state("org-1", disabled_action_types={"pagerduty", "fax"})
        ctrl = svc.build_runtime_control("org-1")
        assert "fax" not in ctrl["disabled_action_types"]
        assert "pagerduty" in ctrl["disabled_action_types"]

    def test_get_org_state_none_for_unconfigured(self):
        svc = ActionKillSwitchService()
        assert svc.get_org_state("org-unknown") is None

    def test_reset_clears_all(self):
        svc = ActionKillSwitchService()
        svc.disable_all("org-1")
        svc.disable_all("org-2")
        svc.reset()
        assert svc.get_org_state("org-1") is None
        assert svc.get_org_state("org-2") is None

    def test_reset_by_org_clears_only_that_org(self):
        svc = ActionKillSwitchService()
        svc.disable_all("org-1")
        svc.disable_all("org-2")
        svc.reset(org_id="org-1")
        assert svc.get_org_state("org-1") is None
        assert svc.get_org_state("org-2") is not None

    def test_to_dict_on_state(self):
        svc = ActionKillSwitchService()
        svc.disable_connector_type("org-1", "webhook")
        state = svc.get_org_state("org-1")
        d = state.to_dict()
        assert d["org_id"] == "org-1"
        assert "webhook" in d["disabled_action_types"]


# ===========================================================================
# Integration wiring tests — agent + observability + kill-switch
# ===========================================================================

class TestAgentObservabilityWiring:
    """Validates observability.record() is called after a real dispatch."""

    @pytest.mark.asyncio
    async def test_successful_dispatch_records_observability_event(self):
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: _make_client(200),
        )
        agent = AutonomousActionAgent(connector_service=svc)
        obs = ConnectorObservabilityService()

        runtime = {"action_observability": obs}

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-obs-1", _ctx(_urgent_enrichment(), runtime=runtime))

        assert result.outputs["action_status"] == "success"
        m = obs.metrics_for("org-obs-1", "pagerduty")
        assert m is not None
        assert m.total == 1
        assert m.success == 1

    @pytest.mark.asyncio
    async def test_failed_dispatch_records_failed_event(self):
        svc = ExternalConnectorService(
            backoff_base=0.001,
            max_attempts=1,
            _http_client_factory=lambda: _make_client(500),
        )
        agent = AutonomousActionAgent(connector_service=svc)
        obs = ConnectorObservabilityService()

        runtime = {"action_observability": obs}

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            await agent.run("mem-obs-2", _ctx(_urgent_enrichment(), runtime=runtime))

        m = obs.metrics_for("org-obs-1", "pagerduty")
        assert m is not None
        assert m.failed == 1
        assert m.total == 1

    @pytest.mark.asyncio
    async def test_no_observability_service_does_not_raise(self):
        """Agent runs fine when action_observability is absent from runtime."""
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: _make_client(200),
        )
        agent = AutonomousActionAgent(connector_service=svc)

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-obs-3", _ctx(_urgent_enrichment()))

        assert result.outputs["action_status"] == "success"

    @pytest.mark.asyncio
    async def test_observability_latency_ms_is_positive(self):
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: _make_client(200),
        )
        agent = AutonomousActionAgent(connector_service=svc)
        obs = ConnectorObservabilityService()

        runtime = {"action_observability": obs}

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            await agent.run("mem-obs-4", _ctx(_urgent_enrichment(), runtime=runtime))

        m = obs.metrics_for("org-obs-1", "pagerduty")
        assert m.avg_latency_ms is not None
        assert m.avg_latency_ms >= 0.0


class TestAgentKillSwitchWiring:
    """Validates kill-switch state is resolved from ActionKillSwitchService."""

    @pytest.mark.asyncio
    async def test_kill_switch_disable_all_blocks_dispatch(self):
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: _make_client(200),
        )
        client_mock = _make_client(200)
        svc2 = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client_mock,
        )
        agent = AutonomousActionAgent(connector_service=svc2)

        ks = ActionKillSwitchService()
        ks.disable_all("org-ks-1")

        runtime = {"action_kill_switch": ks}

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-ks-1", _ctx(_urgent_enrichment(), runtime=runtime, org_id="org-ks-1"))

        assert result.outputs["action_status"] == "denied"
        assert result.outputs["action_dispatched"] is False
        client_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_disable_connector_type_blocks_that_type(self):
        client_mock = _make_client(200)
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client_mock,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        ks = ActionKillSwitchService()
        ks.disable_connector_type("org-ks-2", "pagerduty")

        runtime = {"action_kill_switch": ks}

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run(
                "mem-ks-2",
                _ctx(_urgent_enrichment(), runtime=runtime, org_id="org-ks-2"),
            )

        assert result.outputs["action_status"] == "denied"
        client_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_dry_run_skips_http(self):
        client_mock = _make_client(200)
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client_mock,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        ks = ActionKillSwitchService()
        ks.set_dry_run("org-ks-3", dry_run=True)

        runtime = {"action_kill_switch": ks}

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run(
                "mem-ks-3",
                _ctx(_urgent_enrichment(), runtime=runtime, org_id="org-ks-3"),
            )

        assert result.outputs.get("_dry_run") is True
        assert result.outputs["action_dispatched"] is False
        client_mock.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_runtime_control_wins_over_kill_switch(self):
        """Explicit action_runtime_control in runtime takes priority over kill-switch."""
        client_mock = _make_client(200)
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client_mock,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        ks = ActionKillSwitchService()
        ks.disable_all("org-ks-4")  # kill-switch says disabled

        # But explicit runtime_control says enabled
        runtime = {
            "action_kill_switch": ks,
            "action_runtime_control": {"enabled": True, "dry_run": False, "disabled_action_types": []},
        }

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run(
                "mem-ks-4",
                _ctx(_urgent_enrichment(), runtime=runtime, org_id="org-ks-4"),
            )

        # Explicit override wins — dispatch should succeed
        assert result.outputs["action_status"] == "success"
        client_mock.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_kill_switch_service_uses_permissive_defaults(self):
        """Without kill-switch service and no runtime_control, all actions allowed."""
        client_mock = _make_client(200)
        svc = ExternalConnectorService(
            backoff_base=0.001,
            _http_client_factory=lambda: client_mock,
        )
        agent = AutonomousActionAgent(connector_service=svc)

        with patch("app.agents.autonomous_action_agent.settings") as ms:
            ms.AGENT_STRATEGY = "heuristic"
            result = await agent.run("mem-ks-5", _ctx(_urgent_enrichment()))

        assert result.outputs["action_status"] == "success"
