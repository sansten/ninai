"""E2E test configuration.

Forces heuristic strategy for all agent calls so e2e tests exercise
deterministic heuristic paths rather than live LLM calls (which vary by
vLLM availability/version).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ninai_enterprise importable when running e2e tests from the OSS backend root.
_enterprise_src = Path(__file__).resolve().parents[4] / "ninai-enterprise" / "src"
if _enterprise_src.exists() and str(_enterprise_src) not in sys.path:
    sys.path.insert(0, str(_enterprise_src))


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
