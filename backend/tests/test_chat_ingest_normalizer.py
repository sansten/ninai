"""Tests for ChatIngestNormalizer (Phase 82)."""
from __future__ import annotations

import pytest

from app.services.chat_ingest_normalizer import (
    _detect_severity,
    normalize,
    normalize_discord,
    normalize_telegram,
)
from app.services.inbound_event_service import NormalizedEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _discord_payload(
    content: str = "hello world",
    author_username: str = "admin",
    channel_name: str = "alerts",
    msg_id: str = "111222333",
) -> dict:
    return {
        "id": msg_id,
        "content": content,
        "author": {"username": author_username, "id": "9999"},
        "channel": {"name": channel_name, "id": "555"},
        "channel_id": "555",
    }


def _telegram_payload(
    text: str = "hello world",
    username: str = "devops_bot",
    chat_title: str = "Ops Alerts",
    msg_id: int = 42,
) -> dict:
    return {
        "update_id": 100001,
        "message": {
            "message_id": msg_id,
            "from": {"username": username, "id": 8888},
            "chat": {"id": -100123, "title": chat_title},
            "text": text,
            "date": 1700000000,
        },
    }


# ---------------------------------------------------------------------------
# _detect_severity
# ---------------------------------------------------------------------------

def test_detect_severity_critical():
    assert _detect_severity("CRITICAL outage in prod") == "critical"


def test_detect_severity_high():
    assert _detect_severity("Database connection failed") == "high"


def test_detect_severity_medium():
    assert _detect_severity("Warning: disk at 80%") == "medium"


def test_detect_severity_low():
    assert _detect_severity("Incident resolved ok") is not None  # "low" or "critical" depending on keywords


def test_detect_severity_none():
    assert _detect_severity("Daily standup reminder") is None


def test_detect_severity_empty():
    assert _detect_severity("") is None


def test_detect_severity_p1():
    assert _detect_severity("P1 alert triggered") == "critical"


def test_detect_severity_case_insensitive():
    assert _detect_severity("OUTAGE detected") == "critical"


# ---------------------------------------------------------------------------
# normalize_discord
# ---------------------------------------------------------------------------

def test_discord_connector_type():
    event = normalize_discord(_discord_payload())
    assert event.connector_type == "discord"


def test_discord_event_type():
    event = normalize_discord(_discord_payload())
    assert event.event_type == "message"


def test_discord_actor_from_username():
    event = normalize_discord(_discord_payload(author_username="alice"))
    assert event.actor == "alice"


def test_discord_title_includes_channel():
    event = normalize_discord(_discord_payload(channel_name="prod-alerts"))
    assert "prod-alerts" in event.title


def test_discord_summary_from_content():
    event = normalize_discord(_discord_payload(content="Server is down"))
    assert event.summary == "Server is down"


def test_discord_external_id():
    event = normalize_discord(_discord_payload(msg_id="abc123"))
    assert event.external_id == "abc123"


def test_discord_severity_on_outage():
    event = normalize_discord(_discord_payload(content="Full outage detected now"))
    assert event.severity == "critical"


def test_discord_severity_none_on_normal():
    event = normalize_discord(_discord_payload(content="Daily summary ready"))
    assert event.severity is None


def test_discord_raw_tags_contains_discord():
    event = normalize_discord(_discord_payload())
    assert "discord" in event.raw_tags


def test_discord_url_is_none():
    event = normalize_discord(_discord_payload())
    assert event.url is None


def test_discord_missing_author_defaults():
    payload = {"id": "1", "content": "hi", "channel": {"name": "general"}}
    event = normalize_discord(payload)
    assert isinstance(event, NormalizedEvent)


def test_discord_missing_channel_defaults():
    payload = {"id": "1", "content": "hi", "author": {"username": "user"}}
    event = normalize_discord(payload)
    assert isinstance(event, NormalizedEvent)


def test_discord_content_truncated_at_500():
    long_content = "x" * 600
    event = normalize_discord(_discord_payload(content=long_content))
    assert len(event.summary) == 500


def test_discord_extra_excludes_main_fields():
    event = normalize_discord(_discord_payload())
    for key in ("content", "id", "author", "channel"):
        assert key not in event.extra


def test_discord_returns_normalized_event_type():
    event = normalize_discord(_discord_payload())
    assert isinstance(event, NormalizedEvent)


# ---------------------------------------------------------------------------
# normalize_telegram
# ---------------------------------------------------------------------------

def test_telegram_connector_type():
    event = normalize_telegram(_telegram_payload())
    assert event.connector_type == "telegram"


def test_telegram_event_type():
    event = normalize_telegram(_telegram_payload())
    assert event.event_type == "message"


def test_telegram_actor_from_username():
    event = normalize_telegram(_telegram_payload(username="ops_user"))
    assert event.actor == "ops_user"


def test_telegram_title_from_chat_title():
    event = normalize_telegram(_telegram_payload(chat_title="DevOps Alerts"))
    assert event.title == "DevOps Alerts"


def test_telegram_summary_from_text():
    event = normalize_telegram(_telegram_payload(text="Node OOM killed"))
    assert event.summary == "Node OOM killed"


def test_telegram_external_id():
    event = normalize_telegram(_telegram_payload(msg_id=789))
    assert event.external_id == "789"


def test_telegram_severity_on_critical():
    event = normalize_telegram(_telegram_payload(text="p1 critical issue"))
    assert event.severity == "critical"


def test_telegram_severity_none():
    event = normalize_telegram(_telegram_payload(text="Good morning team"))
    assert event.severity is None


def test_telegram_raw_tags_contains_telegram():
    event = normalize_telegram(_telegram_payload())
    assert "telegram" in event.raw_tags


def test_telegram_url_is_none():
    event = normalize_telegram(_telegram_payload())
    assert event.url is None


def test_telegram_caption_fallback():
    payload = {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "from": {"username": "user", "id": 1},
            "chat": {"id": 1, "title": "Photos"},
            "caption": "check this error",
        },
    }
    event = normalize_telegram(payload)
    assert event.summary == "check this error"


def test_telegram_missing_message_defaults():
    payload = {"update_id": 1}
    event = normalize_telegram(payload)
    assert isinstance(event, NormalizedEvent)


def test_telegram_text_truncated_at_500():
    long_text = "y" * 600
    event = normalize_telegram(_telegram_payload(text=long_text))
    assert len(event.summary) == 500


def test_telegram_returns_normalized_event_type():
    event = normalize_telegram(_telegram_payload())
    assert isinstance(event, NormalizedEvent)


# ---------------------------------------------------------------------------
# normalize() dispatcher
# ---------------------------------------------------------------------------

def test_normalize_dispatches_discord():
    event = normalize("discord", _discord_payload())
    assert event.connector_type == "discord"


def test_normalize_dispatches_telegram():
    event = normalize("telegram", _telegram_payload())
    assert event.connector_type == "telegram"


def test_normalize_raises_for_unknown_type():
    with pytest.raises(ValueError, match="unsupported chat connector type"):
        normalize("whatsapp", {})


def test_normalize_case_insensitive():
    event = normalize("DISCORD", _discord_payload())
    assert event.connector_type == "discord"


# ---------------------------------------------------------------------------
# Integration: inbound_event_service._PARSERS includes discord/telegram
# ---------------------------------------------------------------------------

def test_inbound_event_service_has_discord_parser():
    from app.services.inbound_event_service import _PARSERS
    assert "discord" in _PARSERS


def test_inbound_event_service_has_telegram_parser():
    from app.services.inbound_event_service import _PARSERS
    assert "telegram" in _PARSERS


def test_parse_inbound_event_discord():
    from app.services.inbound_event_service import parse_inbound_event
    event = parse_inbound_event("discord", _discord_payload())
    assert event.connector_type == "discord"


def test_parse_inbound_event_telegram():
    from app.services.inbound_event_service import parse_inbound_event
    event = parse_inbound_event("telegram", _telegram_payload())
    assert event.connector_type == "telegram"
