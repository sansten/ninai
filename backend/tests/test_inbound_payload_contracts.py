"""Contract tests for inbound connector payload schemas (Phase 52 Slice 3)."""

from __future__ import annotations

import pytest

from app.services.inbound_event_service import validate_inbound_payload_contract


class TestPagerDutyContract:
    def test_accepts_event_data_shape(self):
        validate_inbound_payload_contract(
            "pagerduty",
            {"event": {"data": {"id": "PD-1", "title": "incident"}}},
        )

    def test_accepts_incident_shape(self):
        validate_inbound_payload_contract(
            "pagerduty",
            {"incident": {"id": "PD-1", "title": "incident"}, "event": {}},
        )

    def test_rejects_missing_structures(self):
        with pytest.raises(ValueError, match="event.data or incident"):
            validate_inbound_payload_contract("pagerduty", {"event": {}})


class TestJiraContract:
    def test_accepts_issue_with_fields(self):
        validate_inbound_payload_contract(
            "jira",
            {"issue": {"key": "PROJ-1", "fields": {"summary": "x"}}},
        )

    def test_rejects_missing_issue(self):
        with pytest.raises(ValueError, match="requires object field 'issue'"):
            validate_inbound_payload_contract("jira", {})

    def test_rejects_missing_fields(self):
        with pytest.raises(ValueError, match="issue.fields"):
            validate_inbound_payload_contract("jira", {"issue": {}})


class TestSlackContract:
    def test_accepts_event_wrapper(self):
        validate_inbound_payload_contract("slack", {"event": {"type": "message", "text": "hi"}})

    def test_accepts_root_event_object(self):
        validate_inbound_payload_contract("slack", {"type": "message", "text": "hi"})

    def test_rejects_missing_type(self):
        with pytest.raises(ValueError, match="event.type"):
            validate_inbound_payload_contract("slack", {"event": {"text": "hi"}})


class TestGenericContract:
    def test_accepts_identity_fields(self):
        for payload in (
            {"id": "1"},
            {"event_id": "e-1"},
            {"title": "alert"},
            {"summary": "alert"},
            {"name": "alert"},
            {"message": "alert"},
        ):
            validate_inbound_payload_contract("webhook", payload)

    def test_rejects_payload_without_identity(self):
        with pytest.raises(ValueError, match="requires one of"):
            validate_inbound_payload_contract("webhook", {"foo": "bar"})
