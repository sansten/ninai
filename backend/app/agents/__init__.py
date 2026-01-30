"""Agent framework (async enrichment pipeline).

This package implements the BaseAgent contract and concrete agents
(eg ClassificationAgent) as described in AGENT_IMPLEMENTATION_GUIDE.md.
"""

from app.agents.types import AgentContext, AgentResult

__all__ = ["AgentContext", "AgentResult"]
