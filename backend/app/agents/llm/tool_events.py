"""Lightweight tool event types for agent trajectory logging."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict


class ToolEvent(TypedDict, total=False):
    event_type: str
    payload: dict[str, Any]
    summary_text: str


ToolEventSink = Callable[[ToolEvent], Awaitable[None]]
