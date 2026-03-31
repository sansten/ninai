"""Inbound Event Service — Phase 48.

Converts raw inbound webhook payloads from external systems (Jira, PagerDuty,
Slack, GitHub, generic) into Ninai memory records via the EventPublishingService.

This is the inbound side of the connector hub — Phase 45 provided the outbound
WebSocket stream.  This service provides the inbound path: external system
→ Ninai memory → enrichment pipeline.

Design:
- parse_event(connector_type, payload) → NormalizedEvent (pure function)
- EventRouter picks which memory fields to populate based on connector_type
- Caller (endpoint) creates the memory record via the standard memory API

No DB access here — purely a transformation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Normalised event type
# ---------------------------------------------------------------------------

@dataclass
class NormalizedEvent:
    """Connector-agnostic representation of an inbound external event."""
    connector_type: str
    event_type: str           # "incident", "issue", "message", "alert", "unknown"
    title: str
    summary: str
    external_id: str | None   # ID in the originating system
    severity: str | None      # "critical" | "high" | "medium" | "low" | None
    actor: str | None         # email / username of the person who triggered it
    url: str | None           # link back to the originating system
    raw_tags: list[str]
    extra: dict               # any remaining fields preserved for enrichment
    received_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Parser helpers (pure functions — one per connector type)
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int = 500) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_pagerduty(payload: dict) -> NormalizedEvent:
    """Parse PagerDuty webhook v3 payload."""
    event_data = payload.get("event", {})
    data = event_data.get("data", {})
    incident = data if "id" in data else payload.get("incident", {})

    title = incident.get("title") or payload.get("summary") or "PagerDuty incident"
    severity = str(incident.get("urgency") or "").lower() or None
    if severity and severity not in ("critical", "high", "medium", "low"):
        severity = None

    return NormalizedEvent(
        connector_type="pagerduty",
        event_type="incident",
        title=_truncate(title, 255),
        summary=_truncate(
            incident.get("summary") or incident.get("description") or title, 500
        ),
        external_id=str(incident.get("id") or ""),
        severity=severity,
        actor=str(incident.get("created_by") or ""),
        url=incident.get("html_url") or incident.get("self") or None,
        raw_tags=["pagerduty", "incident"],
        extra={
            "service": incident.get("service", {}).get("summary", ""),
            "status": incident.get("status", ""),
        },
    )


def _parse_jira(payload: dict) -> NormalizedEvent:
    """Parse Jira webhook payload."""
    issue = payload.get("issue", {})
    fields = issue.get("fields", {})
    event_type = str(payload.get("webhookEvent") or "jira:issue_updated")

    title = fields.get("summary") or "Jira issue"
    priority = (fields.get("priority") or {}).get("name", "").lower()
    sev_map = {"blocker": "critical", "critical": "critical", "major": "high",
               "minor": "medium", "trivial": "low"}
    severity = sev_map.get(priority)

    actor_obj = payload.get("user") or payload.get("initiator") or {}
    actor = actor_obj.get("emailAddress") or actor_obj.get("displayName") or None

    return NormalizedEvent(
        connector_type="jira",
        event_type="issue",
        title=_truncate(title, 255),
        summary=_truncate(
            fields.get("description") or title, 500
        ),
        external_id=issue.get("key") or issue.get("id") or None,
        severity=severity,
        actor=actor,
        url=issue.get("self") or None,
        raw_tags=["jira", "issue", event_type.replace("jira:", "").replace("_", "-")],
        extra={
            "project": (fields.get("project") or {}).get("key", ""),
            "status": (fields.get("status") or {}).get("name", ""),
            "issue_type": (fields.get("issuetype") or {}).get("name", ""),
        },
    )


def _parse_slack(payload: dict) -> NormalizedEvent:
    """Parse Slack event API payload."""
    event = payload.get("event", payload)
    event_type = event.get("type", "message")
    text = event.get("text") or ""

    return NormalizedEvent(
        connector_type="slack",
        event_type="message",
        title=_truncate(f"Slack {event_type}", 255),
        summary=_truncate(text, 500),
        external_id=event.get("client_msg_id") or event.get("ts") or None,
        severity=None,
        actor=event.get("user") or None,
        url=None,
        raw_tags=["slack", event_type],
        extra={
            "channel": event.get("channel", ""),
            "team": payload.get("team_id", ""),
        },
    )


def _parse_generic(connector_type: str, payload: dict) -> NormalizedEvent:
    """Best-effort parser for unknown connector types."""
    title = (
        payload.get("title")
        or payload.get("summary")
        or payload.get("name")
        or payload.get("message")
        or f"{connector_type} event"
    )
    summary = (
        payload.get("body")
        or payload.get("description")
        or payload.get("text")
        or str(title)
    )
    sev_raw = str(payload.get("severity") or payload.get("priority") or "").lower()
    sev_map = {"critical": "critical", "high": "high", "medium": "medium", "low": "low",
               "error": "high", "warning": "medium", "info": "low"}
    severity = sev_map.get(sev_raw)

    return NormalizedEvent(
        connector_type=connector_type,
        event_type=str(payload.get("event_type") or payload.get("type") or "unknown"),
        title=_truncate(str(title), 255),
        summary=_truncate(str(summary), 500),
        external_id=str(payload.get("id") or payload.get("event_id") or ""),
        severity=severity,
        actor=str(payload.get("user") or payload.get("actor") or payload.get("email") or ""),
        url=payload.get("url") or payload.get("link") or None,
        raw_tags=[connector_type, "inbound"],
        extra={k: v for k, v in payload.items()
               if k not in ("title", "summary", "body", "description", "text",
                            "severity", "priority", "id", "event_id", "type")},
    )


# ---------------------------------------------------------------------------
# Public parse function
# ---------------------------------------------------------------------------

_PARSERS = {
    "pagerduty":   _parse_pagerduty,
    "jira":        _parse_jira,
    "slack":       _parse_slack,
}


def validate_inbound_payload_contract(connector_type: str, payload: dict[str, Any]) -> None:
    """Validate minimal inbound payload contracts for known connector types.

    Raises ValueError when a required shape is missing.
    """
    ctype = str(connector_type).lower().strip()

    if ctype == "pagerduty":
        event = payload.get("event")
        if not isinstance(event, dict):
            raise ValueError("pagerduty payload requires object field 'event'")
        data = event.get("data")
        incident = payload.get("incident")
        if not isinstance(data, dict) and not isinstance(incident, dict):
            raise ValueError("pagerduty payload requires event.data or incident object")
        return

    if ctype == "jira":
        issue = payload.get("issue")
        if not isinstance(issue, dict):
            raise ValueError("jira payload requires object field 'issue'")
        fields = issue.get("fields")
        if not isinstance(fields, dict):
            raise ValueError("jira payload requires issue.fields object")
        return

    if ctype == "slack":
        event = payload.get("event", payload)
        if not isinstance(event, dict):
            raise ValueError("slack payload requires object field 'event' or object root")
        if not event.get("type"):
            raise ValueError("slack payload requires event.type")
        return

    # Generic/webhook-like payloads must at least carry one identity signal.
    has_identity = any(
        payload.get(k)
        for k in ("id", "event_id", "title", "summary", "name", "message")
    )
    if not has_identity:
        raise ValueError("generic payload requires one of id/event_id/title/summary/name/message")


def parse_inbound_event(
    connector_type: str,
    payload: dict[str, Any],
) -> NormalizedEvent:
    """Convert a raw external webhook payload to a NormalizedEvent.

    Falls back to the generic parser for unregistered connector types.
    """
    parser = _PARSERS.get(str(connector_type).lower())
    if parser:
        return parser(payload)
    return _parse_generic(connector_type, payload)


def event_to_memory_fields(event: NormalizedEvent) -> dict[str, Any]:
    """Map NormalizedEvent → fields for a Ninai memory create call."""
    tags = list(event.raw_tags)
    if event.severity:
        tags.append(f"severity:{event.severity}")
    if event.event_type:
        tags.append(f"event_type:{event.event_type}")

    content_parts = [event.summary]
    if event.actor:
        content_parts.append(f"Actor: {event.actor}")
    if event.external_id:
        content_parts.append(f"External ID: {event.external_id}")
    if event.url:
        content_parts.append(f"URL: {event.url}")

    metadata: dict[str, Any] = {
        "connector_type": event.connector_type,
        "event_type": event.event_type,
        "external_id": event.external_id,
        "severity": event.severity,
        "actor": event.actor,
        "source_url": event.url,
        **event.extra,
    }

    return {
        "title": event.title,
        "content": "\n".join(content_parts),
        "memory_type": "episodic",
        "source_type": "integration",
        "tags": tags[:20],
        "metadata": metadata,
    }
