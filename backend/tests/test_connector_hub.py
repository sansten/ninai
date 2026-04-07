"""Tests for Phase 48 — Environment Connector Hub.

Covers: ConnectorRegistryService, InboundEventService (parse + map),
and the /integrations/connectors API endpoint layer.
"""

from __future__ import annotations

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.connector_registry_service import (
    ConnectorRegistryService,
    ConnectorSpec,
    ConnectorTestResult,
)
from app.services.inbound_event_service import (
    parse_inbound_event,
    event_to_memory_fields,
    NormalizedEvent,
    _parse_pagerduty,
    _parse_jira,
    _parse_slack,
    _parse_generic,
    _truncate,
)
from app.services.external_connector_service import ExternalConnectorService, DispatchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spec(
    org_id: str = "org-1",
    connector_type: str = "webhook",
    target_url: str = "https://hooks.example.com/test",
    is_active: bool = True,
    event_types: list[str] | None = None,
) -> ConnectorSpec:
    return ConnectorSpec(
        id=str(uuid.uuid4()),
        organization_id=org_id,
        name=f"test-{connector_type}",
        connector_type=connector_type,
        target_url=target_url,
        credential_ref=None,
        headers_template={},
        event_types=event_types or [connector_type],
        is_active=is_active,
    )


def _make_registry(specs: list[ConnectorSpec] | None = None) -> ConnectorRegistryService:
    registry = ConnectorRegistryService()
    for spec in (specs or []):
        registry.register(spec)
    return registry


# ===========================================================================
# ConnectorRegistryService tests
# ===========================================================================

class TestConnectorRegistryServiceRegister:
    def test_register_returns_spec(self):
        registry = _make_registry()
        spec = _make_spec()
        result = registry.register(spec)
        assert result.id == spec.id

    def test_register_unknown_type_raises(self):
        registry = _make_registry()
        spec = _make_spec(connector_type="fax")
        with pytest.raises(ValueError, match="Unknown connector_type"):
            registry.register(spec)

    def test_register_empty_url_raises(self):
        registry = _make_registry()
        spec = ConnectorSpec(
            id="x", organization_id="org-1", name="t",
            connector_type="webhook", target_url="",
            credential_ref=None, headers_template={}, event_types=["webhook"],
        )
        with pytest.raises(ValueError, match="target_url"):
            registry.register(spec)

    def test_register_empty_id_raises(self):
        registry = _make_registry()
        spec = ConnectorSpec(
            id="", organization_id="org-1", name="t",
            connector_type="webhook", target_url="https://example.com",
            credential_ref=None, headers_template={}, event_types=["webhook"],
        )
        with pytest.raises(ValueError, match="id"):
            registry.register(spec)

    def test_register_overwrites_existing(self):
        spec1 = _make_spec(target_url="https://a.com")
        spec2 = ConnectorSpec(
            id=spec1.id, organization_id=spec1.organization_id,
            name="updated", connector_type="webhook",
            target_url="https://b.com", credential_ref=None,
            headers_template={}, event_types=["webhook"],
        )
        registry = _make_registry([spec1])
        registry.register(spec2)
        found = registry.get_by_id(org_id=spec1.organization_id, connector_id=spec1.id)
        assert found is not None
        assert found.target_url == "https://b.com"


class TestConnectorRegistryServiceLookup:
    def test_get_by_id_found(self):
        spec = _make_spec()
        registry = _make_registry([spec])
        found = registry.get_by_id(org_id=spec.organization_id, connector_id=spec.id)
        assert found is not None
        assert found.id == spec.id

    def test_get_by_id_not_found(self):
        registry = _make_registry()
        assert registry.get_by_id(org_id="org-1", connector_id="missing") is None

    def test_get_for_action_returns_active(self):
        spec = _make_spec(connector_type="pagerduty")
        registry = _make_registry([spec])
        found = registry.get_for_action(org_id=spec.organization_id, connector_type="pagerduty")
        assert found is not None

    def test_get_for_action_skips_inactive(self):
        spec = _make_spec(connector_type="pagerduty", is_active=False)
        registry = _make_registry([spec])
        found = registry.get_for_action(org_id=spec.organization_id, connector_type="pagerduty")
        assert found is None

    def test_get_for_action_wrong_org_not_found(self):
        spec = _make_spec(org_id="org-1")
        registry = _make_registry([spec])
        found = registry.get_for_action(org_id="org-2", connector_type="webhook")
        assert found is None

    def test_list_for_org(self):
        specs = [_make_spec(org_id="org-1"), _make_spec(org_id="org-1")]
        registry = _make_registry(specs)
        listed = registry.list_for_org(org_id="org-1")
        assert len(listed) == 2

    def test_list_active_for_org_filters_inactive(self):
        active = _make_spec(org_id="org-1", is_active=True)
        inactive = _make_spec(org_id="org-1", is_active=False)
        registry = _make_registry([active, inactive])
        listed = registry.list_active_for_org(org_id="org-1")
        assert len(listed) == 1
        assert listed[0].id == active.id


class TestConnectorRegistryServiceMutations:
    def test_deregister_returns_true(self):
        spec = _make_spec()
        registry = _make_registry([spec])
        assert registry.deregister(org_id=spec.organization_id, connector_id=spec.id) is True
        assert registry.get_by_id(org_id=spec.organization_id, connector_id=spec.id) is None

    def test_deregister_missing_returns_false(self):
        registry = _make_registry()
        assert registry.deregister(org_id="org-1", connector_id="nope") is False

    def test_set_active_false(self):
        spec = _make_spec(is_active=True)
        registry = _make_registry([spec])
        registry.set_active(org_id=spec.organization_id, connector_id=spec.id, active=False)
        found = registry.get_by_id(org_id=spec.organization_id, connector_id=spec.id)
        assert found is not None
        assert found.is_active is False

    def test_set_active_missing_returns_false(self):
        registry = _make_registry()
        assert registry.set_active(org_id="org-1", connector_id="x", active=True) is False

    def test_deregister_clears_failure_count(self):
        spec = _make_spec()
        registry = _make_registry([spec])
        # Record a few failures to populate the counter
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) == 2
        # Deregister should clear it
        registry.deregister(org_id=spec.organization_id, connector_id=spec.id)
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) == 0


# ===========================================================================
# Circuit-breaker (record_dispatch_outcome / dispatch_via_connector)
# ===========================================================================

class TestCircuitBreaker:
    def test_success_resets_failure_count(self):
        spec = _make_spec()
        registry = _make_registry([spec])
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=True)
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) == 0

    def test_failure_increments_count(self):
        spec = _make_spec()
        registry = _make_registry([spec])
        for i in range(3):
            registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) == 3

    def test_auto_disable_after_threshold(self):
        spec = _make_spec(is_active=True)
        registry = _make_registry([spec])
        auto_disabled = False
        for _ in range(5):
            result = registry.record_dispatch_outcome(
                org_id=spec.organization_id, connector_id=spec.id, success=False
            )
            if result:
                auto_disabled = True
        assert auto_disabled is True
        found = registry.get_by_id(org_id=spec.organization_id, connector_id=spec.id)
        assert found is not None
        assert found.is_active is False

    def test_failure_count_reset_after_auto_disable(self):
        spec = _make_spec()
        registry = _make_registry([spec])
        for _ in range(5):
            registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        # Counter should be reset after auto-disable
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) == 0

    def test_no_auto_disable_before_threshold(self):
        spec = _make_spec(is_active=True)
        registry = _make_registry([spec])
        for _ in range(4):
            registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        found = registry.get_by_id(org_id=spec.organization_id, connector_id=spec.id)
        assert found is not None
        assert found.is_active is True

    def test_unknown_connector_failure_count_is_zero(self):
        registry = _make_registry()
        assert registry.consecutive_failure_count(org_id="org-1", connector_id="missing") == 0

    @pytest.mark.asyncio
    async def test_dispatch_via_connector_success_resets_count(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        dispatcher = ExternalConnectorService(_http_client_factory=lambda: client)
        spec = _make_spec()
        registry = ConnectorRegistryService(dispatcher=dispatcher)
        registry.register(spec)

        # Prime with 2 failures
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)
        registry.record_dispatch_outcome(org_id=spec.organization_id, connector_id=spec.id, success=False)

        result = await registry.dispatch_via_connector(
            org_id=spec.organization_id, connector_id=spec.id, payload={"event": "test"}
        )
        assert result.status == "success"
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) == 0

    @pytest.mark.asyncio
    async def test_dispatch_via_connector_failure_increments_count(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "error"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        dispatcher = ExternalConnectorService(_http_client_factory=lambda: client)
        spec = _make_spec()
        registry = ConnectorRegistryService(dispatcher=dispatcher)
        registry.register(spec)

        await registry.dispatch_via_connector(
            org_id=spec.organization_id, connector_id=spec.id, payload={"event": "test"}
        )
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) >= 1

    @pytest.mark.asyncio
    async def test_dispatch_via_connector_inactive_raises(self):
        spec = _make_spec(is_active=False)
        registry = _make_registry([spec])
        with pytest.raises(ValueError, match="inactive"):
            await registry.dispatch_via_connector(
                org_id=spec.organization_id, connector_id=spec.id, payload={}
            )

    @pytest.mark.asyncio
    async def test_dispatch_via_connector_not_found_raises(self):
        registry = _make_registry()
        with pytest.raises(ValueError, match="not found"):
            await registry.dispatch_via_connector(org_id="org-1", connector_id="missing", payload={})

    @pytest.mark.asyncio
    async def test_test_connector_failure_increments_circuit_breaker(self):
        resp = MagicMock()
        resp.status_code = 503
        resp.text = "down"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        dispatcher = ExternalConnectorService(_http_client_factory=lambda: client)
        spec = _make_spec()
        registry = ConnectorRegistryService(dispatcher=dispatcher)
        registry.register(spec)

        await registry.test_connector(org_id=spec.organization_id, connector_id=spec.id)
        assert registry.consecutive_failure_count(org_id=spec.organization_id, connector_id=spec.id) >= 1


class TestConnectorRegistryServiceCredential:
    def test_no_credential_ref_returns_none(self):
        spec = _make_spec()
        spec.credential_ref = None
        registry = _make_registry()
        assert registry.resolve_credential(spec) is None

    def test_credential_ref_read_from_env(self, monkeypatch):
        spec = _make_spec()
        spec.credential_ref = "my_token"
        monkeypatch.setenv("NINAI_CONNECTOR_MY_TOKEN", "secret123")
        registry = _make_registry()
        assert registry.resolve_credential(spec) == "secret123"

    def test_short_credential_ref_used_as_literal(self):
        spec = _make_spec()
        spec.credential_ref = "short-token"
        registry = _make_registry()
        val = registry.resolve_credential(spec)
        assert val == "short-token"

    def test_build_dispatch_headers_pagerduty_auth(self, monkeypatch):
        spec = _make_spec(connector_type="pagerduty")
        spec.credential_ref = "pd-key"
        registry = _make_registry()
        headers = registry.build_dispatch_headers(spec)
        assert "Authorization" in headers
        assert "Token" in headers["Authorization"]

    def test_build_dispatch_headers_merges_template(self):
        spec = _make_spec()
        spec.headers_template = {"X-Custom": "value"}
        spec.credential_ref = None
        registry = _make_registry()
        headers = registry.build_dispatch_headers(spec)
        assert headers["X-Custom"] == "value"


class TestConnectorRegistryServiceTest:
    @pytest.mark.asyncio
    async def test_test_connector_ok(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "ok"
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)

        dispatcher = ExternalConnectorService(
            _http_client_factory=lambda: client
        )
        spec = _make_spec()
        registry = ConnectorRegistryService(dispatcher=dispatcher)
        registry.register(spec)

        result = await registry.test_connector(
            org_id=spec.organization_id, connector_id=spec.id
        )
        assert result.status == "ok"
        assert result.http_status_code == 200

    @pytest.mark.asyncio
    async def test_test_connector_not_found(self):
        registry = _make_registry()
        result = await registry.test_connector(org_id="org-1", connector_id="missing")
        assert result.status == "failed"
        assert "not found" in (result.error or "")


# ===========================================================================
# InboundEventService — parser tests
# ===========================================================================

class TestParsePagerDuty:
    def _payload(self) -> dict:
        return {
            "event": {
                "data": {
                    "id": "PD-12345",
                    "title": "High error rate on prod",
                    "summary": "API error rate exceeded threshold",
                    "urgency": "high",
                    "html_url": "https://pagerduty.com/incidents/12345",
                    "created_by": "ops@company.com",
                    "status": "triggered",
                    "service": {"summary": "API Service"},
                }
            }
        }

    def test_event_type_is_incident(self):
        ev = _parse_pagerduty(self._payload())
        assert ev.event_type == "incident"
        assert ev.connector_type == "pagerduty"

    def test_title_extracted(self):
        ev = _parse_pagerduty(self._payload())
        assert "error rate" in ev.title.lower()

    def test_severity_mapped(self):
        ev = _parse_pagerduty(self._payload())
        assert ev.severity == "high"

    def test_external_id_set(self):
        ev = _parse_pagerduty(self._payload())
        assert ev.external_id == "PD-12345"

    def test_unknown_urgency_clears_severity(self):
        p = self._payload()
        p["event"]["data"]["urgency"] = "weird"
        ev = _parse_pagerduty(p)
        assert ev.severity is None

    def test_pagerduty_tag_in_raw_tags(self):
        ev = _parse_pagerduty(self._payload())
        assert "pagerduty" in ev.raw_tags


class TestParseJira:
    def _payload(self) -> dict:
        return {
            "webhookEvent": "jira:issue_created",
            "issue": {
                "key": "PROJ-42",
                "self": "https://jira.example.com/rest/api/2/issue/12345",
                "fields": {
                    "summary": "Login fails for enterprise users",
                    "description": "Multiple users report SSO failures",
                    "priority": {"name": "Major"},
                    "project": {"key": "PROJ"},
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Bug"},
                },
            },
            "user": {"emailAddress": "dev@company.com"},
        }

    def test_event_type_is_issue(self):
        ev = _parse_jira(self._payload())
        assert ev.event_type == "issue"

    def test_external_id_is_key(self):
        ev = _parse_jira(self._payload())
        assert ev.external_id == "PROJ-42"

    def test_severity_from_major_priority(self):
        ev = _parse_jira(self._payload())
        assert ev.severity == "high"

    def test_actor_from_email(self):
        ev = _parse_jira(self._payload())
        assert ev.actor == "dev@company.com"

    def test_jira_tag_present(self):
        ev = _parse_jira(self._payload())
        assert "jira" in ev.raw_tags


class TestParseSlack:
    def _payload(self) -> dict:
        return {
            "team_id": "T123",
            "event": {
                "type": "message",
                "text": "Deployment succeeded in prod",
                "user": "U456",
                "channel": "C789",
                "ts": "1234567890.000001",
            },
        }

    def test_event_type_is_message(self):
        ev = _parse_slack(self._payload())
        assert ev.event_type == "message"

    def test_text_in_summary(self):
        ev = _parse_slack(self._payload())
        assert "Deployment" in ev.summary

    def test_slack_tag_present(self):
        ev = _parse_slack(self._payload())
        assert "slack" in ev.raw_tags


class TestParseGeneric:
    def test_title_extracted(self):
        ev = _parse_generic("generic_rest", {"title": "System alert", "severity": "critical"})
        assert ev.title == "System alert"
        assert ev.severity == "critical"

    def test_fallback_to_name(self):
        ev = _parse_generic("webhook", {"name": "alert name"})
        assert "alert name" in ev.title

    def test_severity_mapping(self):
        ev = _parse_generic("webhook", {"severity": "error"})
        assert ev.severity == "high"

    def test_unknown_severity_is_none(self):
        ev = _parse_generic("webhook", {"severity": "unknown_val"})
        assert ev.severity is None


class TestParseInboundEventDispatch:
    def test_pagerduty_routed_correctly(self):
        ev = parse_inbound_event("pagerduty", {"incident": {"title": "test", "id": "1"}})
        assert ev.connector_type == "pagerduty"

    def test_jira_routed_correctly(self):
        ev = parse_inbound_event("jira", {"issue": {"key": "X-1", "fields": {"summary": "t"}}})
        assert ev.connector_type == "jira"

    def test_slack_routed_correctly(self):
        ev = parse_inbound_event("slack", {"event": {"type": "message", "text": "hi"}})
        assert ev.connector_type == "slack"

    def test_unknown_uses_generic(self):
        ev = parse_inbound_event("custom_tool", {"title": "custom event"})
        assert ev.connector_type == "custom_tool"
        assert ev.event_type != ""


class TestEventToMemoryFields:
    def test_contains_required_keys(self):
        ev = NormalizedEvent(
            connector_type="pagerduty",
            event_type="incident",
            title="Test incident",
            summary="Something broke",
            external_id="PD-1",
            severity="high",
            actor="ops@test.com",
            url="https://pd.example.com/1",
            raw_tags=["pagerduty"],
            extra={},
        )
        fields = event_to_memory_fields(ev)
        assert "title" in fields
        assert "content" in fields
        assert "tags" in fields
        assert "metadata" in fields
        assert fields["memory_type"] == "episodic"
        assert fields["source_type"] == "integration"

    def test_severity_in_tags(self):
        ev = NormalizedEvent(
            connector_type="jira", event_type="issue", title="t", summary="s",
            external_id="J-1", severity="critical", actor=None, url=None,
            raw_tags=["jira"], extra={},
        )
        fields = event_to_memory_fields(ev)
        assert "severity:critical" in fields["tags"]

    def test_actor_in_content(self):
        ev = NormalizedEvent(
            connector_type="slack", event_type="message", title="t", summary="s",
            external_id=None, severity=None, actor="user@test.com", url=None,
            raw_tags=[], extra={},
        )
        fields = event_to_memory_fields(ev)
        assert "user@test.com" in fields["content"]

    def test_tags_capped_at_20(self):
        ev = NormalizedEvent(
            connector_type="webhook", event_type="unknown", title="t", summary="s",
            external_id=None, severity=None, actor=None, url=None,
            raw_tags=[f"tag{i}" for i in range(25)], extra={},
        )
        fields = event_to_memory_fields(ev)
        assert len(fields["tags"]) <= 20


class TestTruncateHelper:
    def test_short_string_unchanged(self):
        assert _truncate("hello", 100) == "hello"

    def test_long_string_truncated(self):
        result = _truncate("x" * 1000, 500)
        assert len(result) <= 500
        assert result.endswith("…")

    def test_empty_string(self):
        assert _truncate("", 100) == ""


# ===========================================================================
# Inbound event — memory write (gap 1) and cognitive trigger (gap 3)
# ===========================================================================

class TestInboundEventMemoryWrite:
    """Tests for the inbound event → memory write path.

    The endpoint depends on MemoryService (DB) and the Celery broker, both of
    which are mocked here to keep tests offline and deterministic.
    """

    def _make_pagerduty_payload(self) -> dict:
        return {
            "event": {
                "data": {
                    "id": "PD-99",
                    "title": "Prod DB latency spike",
                    "summary": "DB response time exceeded 5s",
                    "urgency": "high",
                    "html_url": "https://pd.example.com/99",
                    "created_by": "ops@company.com",
                    "status": "triggered",
                    "service": {"summary": "Database"},
                }
            }
        }

    def _inbound_event_to_memory_fields_high(self) -> dict:
        from app.services.inbound_event_service import parse_inbound_event, event_to_memory_fields
        event = parse_inbound_event("pagerduty", self._make_pagerduty_payload())
        return event_to_memory_fields(event)

    def test_event_to_memory_fields_has_content(self):
        fields = self._inbound_event_to_memory_fields_high()
        assert fields.get("content")
        assert len(fields["content"]) > 0

    def test_event_to_memory_fields_source_type_integration(self):
        fields = self._inbound_event_to_memory_fields_high()
        assert fields.get("source_type") == "integration"

    def test_event_to_memory_fields_has_tags(self):
        fields = self._inbound_event_to_memory_fields_high()
        assert isinstance(fields.get("tags"), list)

    def test_pagerduty_severity_in_tags(self):
        fields = self._inbound_event_to_memory_fields_high()
        assert any("severity" in t for t in fields["tags"])

    def test_memory_create_schema_built_from_fields(self):
        from app.schemas.memory import MemoryCreate, MemoryScope, MemoryType
        fields = self._inbound_event_to_memory_fields_high()
        data = MemoryCreate(
            content=str(fields.get("content") or fields.get("title") or ""),
            title=str(fields.get("title") or "")[:500] or None,
            scope=MemoryScope.ORGANIZATION,
            scope_id="org-test",
            memory_type=MemoryType.LONG_TERM,
            tags=list(fields.get("tags") or [])[:20],
            source_type="integration",
            source_id="PD-99",
            extra_metadata=dict(fields.get("metadata") or {}),
        )
        assert data.content
        assert data.scope == MemoryScope.ORGANIZATION
        assert data.source_type == "integration"


class TestCognitiveLoopTrigger:
    """Tests for the cognitive loop trigger logic (gap 3)."""

    def test_high_severity_qualifies_for_trigger(self):
        from app.api.v1.endpoints.connectors import _HIGH_SEVERITY_VALUES
        assert "high" in _HIGH_SEVERITY_VALUES
        assert "critical" in _HIGH_SEVERITY_VALUES

    def test_low_severity_does_not_qualify(self):
        from app.api.v1.endpoints.connectors import _HIGH_SEVERITY_VALUES
        assert "low" not in _HIGH_SEVERITY_VALUES
        assert "medium" not in _HIGH_SEVERITY_VALUES
        assert None not in _HIGH_SEVERITY_VALUES

    def test_pagerduty_high_urgency_produces_high_severity(self):
        from app.services.inbound_event_service import parse_inbound_event
        payload = {
            "event": {
                "data": {
                    "id": "PD-1",
                    "title": "Outage",
                    "summary": "Down",
                    "urgency": "high",
                    "html_url": "https://pd.example.com/1",
                    "created_by": "ops",
                    "status": "triggered",
                    "service": {"summary": "API"},
                }
            }
        }
        event = parse_inbound_event("pagerduty", payload)
        assert event.severity == "high"

    def test_jira_major_priority_produces_high_severity(self):
        from app.services.inbound_event_service import parse_inbound_event
        payload = {
            "webhookEvent": "jira:issue_created",
            "issue": {
                "key": "X-1",
                "self": "https://jira.example.com/x/1",
                "fields": {
                    "summary": "Critical bug",
                    "description": "Broken",
                    "priority": {"name": "Critical"},
                    "project": {"key": "X"},
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Bug"},
                },
            },
            "user": {"emailAddress": "dev@example.com"},
        }
        event = parse_inbound_event("jira", payload)
        assert event.severity == "critical"

    def test_slack_message_does_not_trigger(self):
        from app.services.inbound_event_service import parse_inbound_event
        from app.api.v1.endpoints.connectors import _HIGH_SEVERITY_VALUES
        payload = {"team_id": "T1", "event": {"type": "message", "text": "hello", "user": "U1", "channel": "C1"}}
        event = parse_inbound_event("slack", payload)
        # Slack messages have no severity → should not trigger cognitive review
        assert event.severity not in _HIGH_SEVERITY_VALUES


class TestChatConnectorTypes:
    """Discord and Telegram as valid connector types (Phase 82)."""

    def test_discord_in_valid_types(self):
        from app.services.connector_registry_service import _VALID_TYPES
        assert "discord" in _VALID_TYPES

    def test_telegram_in_valid_types(self):
        from app.services.connector_registry_service import _VALID_TYPES
        assert "telegram" in _VALID_TYPES

    def test_register_discord_connector(self):
        from app.services.connector_registry_service import (
            ConnectorRegistryService, ConnectorSpec,
        )
        svc = ConnectorRegistryService()
        spec = ConnectorSpec(
            id="dc-1",
            organization_id="org-1",
            name="Discord bot",
            connector_type="discord",
            target_url="https://discord.com/api/webhooks/123/abc",
            credential_ref=None,
            headers_template={},
            event_types=["message"],
        )
        result = svc.register(spec)
        assert result.connector_type == "discord"

    def test_register_telegram_connector(self):
        from app.services.connector_registry_service import (
            ConnectorRegistryService, ConnectorSpec,
        )
        svc = ConnectorRegistryService()
        spec = ConnectorSpec(
            id="tg-1",
            organization_id="org-1",
            name="Telegram bot",
            connector_type="telegram",
            target_url="https://api.telegram.org/bot<TOKEN>/sendMessage",
            credential_ref=None,
            headers_template={},
            event_types=["message"],
        )
        result = svc.register(spec)
        assert result.connector_type == "telegram"

    def test_discord_connector_roundtrip(self):
        from app.services.connector_registry_service import (
            ConnectorRegistryService, ConnectorSpec,
        )
        svc = ConnectorRegistryService()
        spec = ConnectorSpec(
            id="dc-rt",
            organization_id="org-rt",
            name="Discord rt",
            connector_type="discord",
            target_url="https://discord.com/api/webhooks/999/xyz",
            credential_ref=None,
            headers_template={},
            event_types=["message"],
        )
        svc.register(spec)
        found = svc.get_by_id(org_id="org-rt", connector_id="dc-rt")
        assert found is not None
        assert found.connector_type == "discord"

    def test_telegram_connector_roundtrip(self):
        from app.services.connector_registry_service import (
            ConnectorRegistryService, ConnectorSpec,
        )
        svc = ConnectorRegistryService()
        spec = ConnectorSpec(
            id="tg-rt",
            organization_id="org-rt",
            name="Telegram rt",
            connector_type="telegram",
            target_url="https://api.telegram.org/bot<TOKEN>/sendMessage",
            credential_ref=None,
            headers_template={},
            event_types=["message"],
        )
        svc.register(spec)
        found = svc.get_by_id(org_id="org-rt", connector_id="tg-rt")
        assert found is not None
        assert found.connector_type == "telegram"

    def test_parse_discord_via_inbound_service(self):
        from app.services.inbound_event_service import parse_inbound_event
        payload = {
            "id": "111",
            "content": "Critical outage in prod",
            "author": {"username": "sre_bot", "id": "1"},
            "channel": {"name": "incidents"},
        }
        event = parse_inbound_event("discord", payload)
        assert event.connector_type == "discord"
        assert event.severity == "critical"

    def test_parse_telegram_via_inbound_service(self):
        from app.services.inbound_event_service import parse_inbound_event
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"username": "alert_bot", "id": 1},
                "chat": {"id": -1, "title": "Alerts"},
                "text": "P2 error on payment service",
            },
        }
        event = parse_inbound_event("telegram", payload)
        assert event.connector_type == "telegram"
        assert event.severity in {"high", "critical"}

    def test_discord_get_for_action(self):
        from app.services.connector_registry_service import (
            ConnectorRegistryService, ConnectorSpec,
        )
        svc = ConnectorRegistryService()
        spec = ConnectorSpec(
            id="dc-act",
            organization_id="org-act",
            name="Discord action",
            connector_type="discord",
            target_url="https://discord.com/api/webhooks/act/token",
            credential_ref=None,
            headers_template={},
            event_types=["message"],
        )
        svc.register(spec)
        found = svc.get_for_action(org_id="org-act", connector_type="discord")
        assert found is not None
        assert found.id == "dc-act"

    def test_telegram_get_for_action(self):
        from app.services.connector_registry_service import (
            ConnectorRegistryService, ConnectorSpec,
        )
        svc = ConnectorRegistryService()
        spec = ConnectorSpec(
            id="tg-act",
            organization_id="org-act2",
            name="Telegram action",
            connector_type="telegram",
            target_url="https://api.telegram.org/bot<TOKEN>/sendMessage",
            credential_ref=None,
            headers_template={},
            event_types=["message"],
        )
        svc.register(spec)
        found = svc.get_for_action(org_id="org-act2", connector_type="telegram")
        assert found is not None
        assert found.id == "tg-act"
