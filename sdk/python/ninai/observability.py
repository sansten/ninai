"""SDK-side observability primitives.

The backend has its own ToolEvent / ToolEventSink types. The SDK needs a
lightweight equivalent so notebooks and examples can capture call traces.

Design goals:
- Minimal, dependency-free
- Async-first interface
- Works for LLM calls and tool invocations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional


ToolEvent = Dict[str, Any]
ToolEventSink = Callable[[ToolEvent], Awaitable[None]]


async def emit_event(sink: ToolEventSink | None, event: ToolEvent) -> None:
    if sink is None:
        return
    await sink(event)


@dataclass
class InMemoryEventSink:
    """Collects events in memory for notebooks/tests."""

    events: list[ToolEvent] = field(default_factory=list)

    async def __call__(self, event: ToolEvent) -> None:
        self.events.append(dict(event))

    def clear(self) -> None:
        self.events.clear()
