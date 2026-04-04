"""E2E test configuration.

Forces heuristic strategy for all agent calls so e2e tests exercise
deterministic heuristic paths rather than live LLM calls (which vary by
Ollama availability/version).
"""

import pytest


@pytest.fixture(autouse=True)
def force_heuristic_strategy(monkeypatch):
    """Patch all known agent modules to use heuristic strategy."""
    modules = [
        "app.agents.memory_decay_agent",
        "app.agents.credibility_agent",
        "app.agents.anomaly_detection_agent",
        "app.agents.conflict_detection_agent",
        "app.agents.narrative_synthesis_agent",
        "app.agents.feedback_integration_agent",
        "app.agents.goal_decomposition_agent",
        "app.agents.enrichment_pipeline",
    ]
    for mod_name in modules:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "settings"):
                monkeypatch.setattr(mod.settings, "AGENT_STRATEGY", "heuristic")
        except ImportError:
            pass
