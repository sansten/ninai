from __future__ import annotations

from abc import ABC, abstractmethod

from app.agents.types import AgentContext, AgentResult


class BaseAgent(ABC):
    name: str
    version: str

    def dependencies(self) -> list[str]:
        return []

    def should_run(self, memory_id: str, context: AgentContext) -> bool:
        return True

    @abstractmethod
    async def run(self, memory_id: str, context: AgentContext) -> AgentResult: ...

    def validate_outputs(self, result: AgentResult) -> None:
        # Override in concrete agents; raise ValueError for invalid outputs.
        return
