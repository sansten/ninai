"""Build the public self-documenting cognitive manifest."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from app.agents.registry import list_registered_agents
from app.core.config import settings
from app.schemas.cognitive_manifest import (
    CognitiveManifestAgent,
    CognitiveManifestEventStream,
    CognitiveManifestIntegrations,
    CognitiveManifestResponse,
)


_PHASE_RE = re.compile(r"\bPhase\s+(\d+)\b", re.IGNORECASE)
_WORD_BOUNDARY_RE = re.compile(r"(?<!^)(?=[A-Z])")
_CAPABILITY_OVERRIDES = {
    "PredictiveMonitorAgent": "predictive_monitoring",
    "FederatedMemoryAgent": "federated_intelligence",
    "QueryIntelligenceAgent": "semantic_search",
}
_STATIC_CAPABILITIES = {
    "semantic_search",
}


class AgentManifestService:
    def build(self) -> CognitiveManifestResponse:
        active_agents = [self._build_agent_entry(agent) for agent in list_registered_agents()]
        deployed_phases = sorted({20, *[a.phase for a in active_agents if a.phase is not None]})
        capabilities = sorted({*_STATIC_CAPABILITIES, *[a.capability for a in active_agents if a.capability]})

        return CognitiveManifestResponse(
            name=str(settings.APP_NAME or "Ninai Cognitive OS"),
            version="1.0.0",
            deployed_phases=deployed_phases,
            active_agents=sorted(active_agents, key=lambda item: item.name.lower()),
            cognitive_capabilities=capabilities,
            integrations=CognitiveManifestIntegrations(
                mcp=self._has_backend_file("mcp_server.py"),
                a2a=True,
                langchain=self._has_backend_file("integrations/langchain_adapter.py"),
                llamaindex=self._has_backend_file("integrations/llamaindex_adapter.py"),
                crewai=self._has_backend_file("integrations/crewai_adapter.py"),
                openai_tools=True,
            ),
            event_stream=CognitiveManifestEventStream(
                websocket="/ws/stream",
                sse="/sse/events",
                webhooks=True,
            ),
        )

    def _build_agent_entry(self, agent) -> CognitiveManifestAgent:
        agent_class = agent.__class__
        doc_text = inspect.getdoc(agent_class) or inspect.getdoc(inspect.getmodule(agent_class)) or ""
        phase = self._extract_phase(doc_text)

        return CognitiveManifestAgent(
            name=agent_class.__name__,
            identifier=str(getattr(agent, "name", agent_class.__name__)),
            version=str(getattr(agent, "version", "v1") or "v1"),
            status="active",
            capability=self._capability_name(agent_class.__name__),
            phase=phase,
            summary=self._extract_summary(doc_text),
            dependencies=[str(dep) for dep in (agent.dependencies() or [])],
        )

    def _extract_phase(self, doc_text: str) -> int | None:
        match = _PHASE_RE.search(doc_text or "")
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _extract_summary(self, doc_text: str) -> str | None:
        if not doc_text:
            return None
        for line in doc_text.splitlines():
            summary = line.strip().strip(".")
            if summary:
                return summary
        return None

    def _capability_name(self, class_name: str) -> str:
        if class_name in _CAPABILITY_OVERRIDES:
            return _CAPABILITY_OVERRIDES[class_name]
        stem = class_name[:-5] if class_name.endswith("Agent") else class_name
        return _WORD_BOUNDARY_RE.sub("_", stem).lower()

    def _has_backend_file(self, relative_path: str) -> bool:
        backend_root = Path(__file__).resolve().parents[2]
        return (backend_root / relative_path).exists()
