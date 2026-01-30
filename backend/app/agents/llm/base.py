from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.agents.llm.tool_events import ToolEventSink


class LLMClient(ABC):
    @abstractmethod
    async def complete_json(
        self,
        *,
        prompt: str,
        schema_hint: dict[str, Any],
        tool_event_sink: ToolEventSink | None = None,
    ) -> dict[str, Any]: ...
