"""Phase 82: ChatIngestNormalizer — normalize Discord and Telegram webhook payloads.

Provides parser functions that convert chat platform webhook payloads into the
shared NormalizedEvent schema used throughout the connector hub.

Usage:
    from app.services.chat_ingest_normalizer import normalize, normalize_discord, normalize_telegram
    event = normalize("discord", payload)
"""

from __future__ import annotations

from typing import Any

from app.services.inbound_event_service import NormalizedEvent, _truncate


# ---------------------------------------------------------------------------
# Severity detection
# ---------------------------------------------------------------------------

_CRITICAL_KEYWORDS = frozenset({"critical", "outage", "down", "p1", "incident"})
_HIGH_KEYWORDS = frozenset({"error", "fail", "failed", "failure", "high", "p2"})
_MEDIUM_KEYWORDS = frozenset({"warn", "warning", "caution", "p3", "degraded"})
_LOW_KEYWORDS = frozenset({"info", "low", "p4", "resolved", "ok"})


def _detect_severity(text: str) -> str | None:
    """Return a severity label inferred from message text keywords, or None."""
    lo = text.lower()
    words = set(lo.split())
    if words & _CRITICAL_KEYWORDS or any(kw in lo for kw in _CRITICAL_KEYWORDS):
        return "critical"
    if words & _HIGH_KEYWORDS or any(kw in lo for kw in _HIGH_KEYWORDS):
        return "high"
    if words & _MEDIUM_KEYWORDS or any(kw in lo for kw in _MEDIUM_KEYWORDS):
        return "medium"
    if words & _LOW_KEYWORDS or any(kw in lo for kw in _LOW_KEYWORDS):
        return "low"
    return None


# ---------------------------------------------------------------------------
# Discord parser
# ---------------------------------------------------------------------------

_DISCORD_PASSTHROUGH_EXCLUDED = frozenset({"content", "id", "author", "channel", "channel_id"})


def normalize_discord(payload: dict[str, Any]) -> NormalizedEvent:
    """Normalize a Discord webhook payload to NormalizedEvent."""
    author: dict = payload.get("author") or {}
    channel: dict = payload.get("channel") or {}

    actor = author.get("username") or (str(author["id"]) if "id" in author else None)
    channel_name = channel.get("name") or str(payload.get("channel_id", ""))
    content: str = payload.get("content") or ""
    msg_id = str(payload.get("id")) if payload.get("id") is not None else None

    title = f"#{channel_name}" if channel_name else "discord message"
    extra = {k: v for k, v in payload.items() if k not in _DISCORD_PASSTHROUGH_EXCLUDED}

    return NormalizedEvent(
        connector_type="discord",
        event_type="message",
        title=title,
        summary=_truncate(content),
        external_id=msg_id,
        severity=_detect_severity(content),
        actor=actor,
        url=None,
        raw_tags=["discord"],
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Telegram parser
# ---------------------------------------------------------------------------

_TELEGRAM_MESSAGE_EXCLUDED = frozenset({"text", "caption", "message_id", "from", "chat"})


def normalize_telegram(payload: dict[str, Any]) -> NormalizedEvent:
    """Normalize a Telegram webhook update payload to NormalizedEvent."""
    message: dict = payload.get("message") or {}
    from_user: dict = message.get("from") or {}
    chat: dict = message.get("chat") or {}

    text: str = message.get("text") or message.get("caption") or ""
    msg_id = str(message["message_id"]) if "message_id" in message else None
    actor = from_user.get("username") or (str(from_user["id"]) if "id" in from_user else None)
    chat_title = chat.get("title") or (str(chat["id"]) if "id" in chat else "telegram message")
    extra = {k: v for k, v in message.items() if k not in _TELEGRAM_MESSAGE_EXCLUDED}

    return NormalizedEvent(
        connector_type="telegram",
        event_type="message",
        title=chat_title,
        summary=_truncate(text),
        external_id=msg_id,
        severity=_detect_severity(text),
        actor=actor,
        url=None,
        raw_tags=["telegram"],
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_CHAT_PARSERS: dict[str, Any] = {
    "discord": normalize_discord,
    "telegram": normalize_telegram,
}


def normalize(connector_type: str, payload: dict[str, Any]) -> NormalizedEvent:
    """Dispatch to the appropriate chat parser.

    Raises ValueError for unsupported connector types.
    """
    parser = _CHAT_PARSERS.get(str(connector_type).lower())
    if parser is None:
        raise ValueError(f"unsupported chat connector type: {connector_type!r}")
    return parser(payload)
