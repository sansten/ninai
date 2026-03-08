"""Tests for Phase 5 — PredictiveMonitorAgent."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from app.agents.predictive_monitor_agent import (
    PredictiveMonitorAgent,
    build_risk_summary,
    detect_high_weight_edge_cluster,
    detect_low_confidence_cross_silo,
    detect_rapid_domain_expansion,
    detect_state_change_cascade,
    detect_unresolved_expansion,
    score_alert,
    _severity,
)
from app.agents.registry import get_agent
from app.agents.types import AgentResult


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_NODE_ACME = {
    "entity": "acme_corp",
    "entity_type": "customer",
    "domains": ["sales", "support", "finance"],
    "silo_count": 3,
    "state_version": 3,
    "node_confidence": 0.88,
}

_NODE_BUDGET = {
    "entity": "budget",
    "entity_type": "metric",
    "domains": ["engineering", "finance"],
    "silo_count": 2,
    "state_version": 2,
    "node_confidence": 0.45,
}

_NODE_CONCEPT = {
    "entity": "alignment",
    "entity_type": "concept",
    "domains": ["sales", "engineering"],
    "silo_count": 2,
    "state_version": 2,
    "node_confidence": 0.60,
}

_NODE_SINGLE = {
    "entity": "project_x",
    "entity_type": "project",
    "domains": ["engineering"],
    "silo_count": 1,
    "state_version": 1,
    "node_confidence": 0.75,
}

_EDGE_HEAVY_1 = {"from_entity": "acme_corp", "to_entity": "budget", "relationship_type": "co_occurrence", "weight": 0.88}
_EDGE_HEAVY_2 = {"from_entity": "acme_corp", "to_entity": "alignment", "relationship_type": "co_occurrence", "weight": 0.85}
_EDGE_LIGHT = {"from_entity": "budget", "to_entity": "project_x", "relationship_type": "co_occurrence", "weight": 0.52}

_CHANGE_ACME = {
    "entity": "acme_corp",
    "change_type": "domain_expansion",
    "description": "acme_corp is active across 3 silos: sales, support, finance",
}
_CHANGE_BUDGET = {
    "entity": "budget",
    "change_type": "domain_expansion",
    "description": "budget is active across 2 silos: engineering, finance",
}
_CHANGE_CONFIDENCE = {
    "entity": "project_x",
    "change_type": "high_confidence_node",
    "description": "project_x resolved with node_confidence 0.9",
}


def _make_result(outputs: dict) -> AgentResult:
    now = datetime.now(timezone.utc)
    return AgentResult(
        agent_name="PredictiveMonitorAgent",
        agent_version="v1",
        memory_id="mem-1",
        status="success",
        confidence=outputs.get("confidence", 0.5),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=now,
        finished_at=now,
    )


def _ctx(
    *,
    world_nodes: list | None = None,
    entity_edges: list | None = None,
    state_changes: list | None = None,
) -> dict:
    enrichment: dict = {
        "world_nodes": world_nodes or [],
        "entity_edges": entity_edges or [],
        "state_changes": state_changes or [],
    }
    return {
        "tenant": {"org_id": "org-1", "org_slug": "test"},
        "memory": {
            "content": "Acme Corp budget review across sales, finance, engineering",
            "enrichment": enrichment,
        },
        "runtime": {"job_id": "job-test"},
    }


# ---------------------------------------------------------------------------
# score_alert
# ---------------------------------------------------------------------------

def test_score_alert_base_pattern():
    score = score_alert(pattern="rapid_domain_expansion")
    assert 0.60 <= score <= 0.70


def test_score_alert_silo_bonus():
    low = score_alert(pattern="rapid_domain_expansion", silo_count=1)
    high = score_alert(pattern="rapid_domain_expansion", silo_count=4)
    assert high > low


def test_score_alert_confidence_penalty_reduces_score():
    base = score_alert(pattern="low_confidence_cross_silo")
    penalised = score_alert(pattern="low_confidence_cross_silo", confidence_penalty=0.05)
    assert penalised < base


def test_score_alert_capped_at_one():
    # silo spread bonus is capped at 0.15; use a no-penalty pattern with high base
    # to verify the cap logic works (score should not exceed 1.0)
    score = score_alert(pattern="rapid_domain_expansion", silo_count=100)
    assert score <= 1.0
    assert score == round(min(0.35 + 0.30 + 0.15, 1.0), 3)


def test_score_alert_unknown_pattern_uses_default():
    score = score_alert(pattern="unknown_pattern")
    assert 0.35 < score < 0.7


# ---------------------------------------------------------------------------
# _severity
# ---------------------------------------------------------------------------

def test_severity_critical():
    assert _severity(0.80) == "critical"


def test_severity_high():
    assert _severity(0.67) == "high"


def test_severity_medium():
    assert _severity(0.50) == "medium"


def test_severity_low():
    assert _severity(0.30) == "low"


# ---------------------------------------------------------------------------
# detect_rapid_domain_expansion
# ---------------------------------------------------------------------------

def test_rapid_expansion_detected_when_silo3_version3():
    alerts = detect_rapid_domain_expansion([_NODE_ACME], threshold=0.0)
    assert len(alerts) == 1
    assert alerts[0]["pattern"] == "rapid_domain_expansion"
    assert "acme_corp" in alerts[0]["affected_entities"]
    assert alerts[0]["risk_score"] >= 0.60


def test_rapid_expansion_not_triggered_single_silo():
    alerts = detect_rapid_domain_expansion([_NODE_SINGLE], threshold=0.0)
    assert alerts == []


def test_rapid_expansion_not_triggered_low_version():
    node = {**_NODE_ACME, "state_version": 2}
    alerts = detect_rapid_domain_expansion([node], threshold=0.0)
    assert alerts == []


def test_rapid_expansion_not_triggered_low_silo_count():
    node = {**_NODE_ACME, "silo_count": 2}
    alerts = detect_rapid_domain_expansion([node], threshold=0.0)
    assert alerts == []


def test_rapid_expansion_threshold_gates_alert():
    alerts = detect_rapid_domain_expansion([_NODE_ACME], threshold=0.99)
    assert alerts == []


def test_rapid_expansion_multiple_nodes():
    node2 = {
        "entity": "competitor_x",
        "entity_type": "org",
        "domains": ["sales", "support", "finance"],
        "silo_count": 3,
        "state_version": 3,
        "node_confidence": 0.80,
    }
    alerts = detect_rapid_domain_expansion([_NODE_ACME, node2], threshold=0.0)
    assert len(alerts) == 2


# ---------------------------------------------------------------------------
# detect_low_confidence_cross_silo
# ---------------------------------------------------------------------------

def test_low_confidence_cross_silo_detected():
    alerts = detect_low_confidence_cross_silo([_NODE_BUDGET], threshold=0.0)
    assert len(alerts) == 1
    assert alerts[0]["pattern"] == "low_confidence_cross_silo"
    assert "budget" in alerts[0]["affected_entities"]


def test_low_confidence_not_triggered_high_confidence():
    alerts = detect_low_confidence_cross_silo([_NODE_ACME], threshold=0.0)
    assert alerts == []


def test_low_confidence_not_triggered_single_silo():
    node = {**_NODE_BUDGET, "silo_count": 1}
    alerts = detect_low_confidence_cross_silo([node], threshold=0.0)
    assert alerts == []


def test_low_confidence_threshold_gates_alert():
    alerts = detect_low_confidence_cross_silo([_NODE_BUDGET], threshold=0.99)
    assert alerts == []


def test_low_confidence_penalty_lowers_score():
    # Very low confidence → larger penalty → lower score
    node_very_low = {**_NODE_BUDGET, "node_confidence": 0.30}
    node_moderate = {**_NODE_BUDGET, "node_confidence": 0.50}
    a1 = detect_low_confidence_cross_silo([node_very_low], threshold=0.0)
    a2 = detect_low_confidence_cross_silo([node_moderate], threshold=0.0)
    assert a1[0]["risk_score"] < a2[0]["risk_score"]


# ---------------------------------------------------------------------------
# detect_high_weight_edge_cluster
# ---------------------------------------------------------------------------

def test_high_weight_cluster_detected_two_heavy_edges():
    alerts = detect_high_weight_edge_cluster([_EDGE_HEAVY_1, _EDGE_HEAVY_2], threshold=0.0)
    assert len(alerts) == 1
    assert alerts[0]["pattern"] == "high_weight_edge_cluster"
    assert "acme_corp" in alerts[0]["affected_entities"]


def test_high_weight_cluster_not_triggered_single_heavy_edge():
    alerts = detect_high_weight_edge_cluster([_EDGE_HEAVY_1], threshold=0.0)
    assert alerts == []


def test_high_weight_cluster_not_triggered_light_edges():
    alerts = detect_high_weight_edge_cluster([_EDGE_LIGHT, _EDGE_LIGHT], threshold=0.0)
    assert alerts == []


def test_high_weight_cluster_empty_edges():
    alerts = detect_high_weight_edge_cluster([], threshold=0.0)
    assert alerts == []


def test_high_weight_cluster_threshold_gates():
    alerts = detect_high_weight_edge_cluster([_EDGE_HEAVY_1, _EDGE_HEAVY_2], threshold=0.99)
    assert alerts == []


def test_high_weight_cluster_collects_all_entity_names():
    alerts = detect_high_weight_edge_cluster([_EDGE_HEAVY_1, _EDGE_HEAVY_2], threshold=0.0)
    entities = alerts[0]["affected_entities"]
    assert "acme_corp" in entities
    assert "budget" in entities
    assert "alignment" in entities


# ---------------------------------------------------------------------------
# detect_unresolved_expansion
# ---------------------------------------------------------------------------

def test_unresolved_expansion_detected_concept_cross_silo():
    alerts = detect_unresolved_expansion([_NODE_CONCEPT], threshold=0.0)
    assert len(alerts) == 1
    assert alerts[0]["pattern"] == "unresolved_expansion"
    assert "alignment" in alerts[0]["affected_entities"]


def test_unresolved_expansion_not_triggered_non_concept():
    alerts = detect_unresolved_expansion([_NODE_ACME], threshold=0.0)
    assert alerts == []


def test_unresolved_expansion_not_triggered_single_silo():
    node = {**_NODE_CONCEPT, "silo_count": 1}
    alerts = detect_unresolved_expansion([node], threshold=0.0)
    assert alerts == []


def test_unresolved_expansion_threshold_gates():
    alerts = detect_unresolved_expansion([_NODE_CONCEPT], threshold=0.99)
    assert alerts == []


# ---------------------------------------------------------------------------
# detect_state_change_cascade
# ---------------------------------------------------------------------------

def test_cascade_detected_two_expansions():
    alerts = detect_state_change_cascade([_CHANGE_ACME, _CHANGE_BUDGET], threshold=0.0)
    assert len(alerts) == 1
    assert alerts[0]["pattern"] == "state_change_cascade"
    assert "acme_corp" in alerts[0]["affected_entities"]
    assert "budget" in alerts[0]["affected_entities"]


def test_cascade_not_triggered_single_expansion():
    alerts = detect_state_change_cascade([_CHANGE_ACME], threshold=0.0)
    assert alerts == []


def test_cascade_not_triggered_non_expansion_changes():
    alerts = detect_state_change_cascade([_CHANGE_CONFIDENCE, _CHANGE_CONFIDENCE], threshold=0.0)
    assert alerts == []


def test_cascade_not_triggered_empty_changes():
    alerts = detect_state_change_cascade([], threshold=0.0)
    assert alerts == []


def test_cascade_threshold_gates():
    alerts = detect_state_change_cascade([_CHANGE_ACME, _CHANGE_BUDGET], threshold=0.99)
    assert alerts == []


def test_cascade_mixed_change_types_only_counts_expansions():
    changes = [_CHANGE_ACME, _CHANGE_CONFIDENCE]
    alerts = detect_state_change_cascade(changes, threshold=0.0)
    assert alerts == []


# ---------------------------------------------------------------------------
# build_risk_summary
# ---------------------------------------------------------------------------

def test_risk_summary_empty_alerts():
    summary = build_risk_summary([], node_count=5)
    assert summary["total_alerts"] == 0
    assert summary["high_risk_alerts"] == 0
    assert summary["patterns_detected"] == []
    assert summary["monitored_nodes"] == 5


def test_risk_summary_counts_high_risk():
    alerts = [
        {"pattern": "rapid_domain_expansion", "risk_score": 0.80},
        {"pattern": "low_confidence_cross_silo", "risk_score": 0.55},
    ]
    summary = build_risk_summary(alerts, node_count=3)
    assert summary["total_alerts"] == 2
    assert summary["high_risk_alerts"] == 1
    assert "rapid_domain_expansion" in summary["patterns_detected"]
    assert summary["monitored_nodes"] == 3


def test_risk_summary_deduplicates_patterns():
    alerts = [
        {"pattern": "rapid_domain_expansion", "risk_score": 0.80},
        {"pattern": "rapid_domain_expansion", "risk_score": 0.75},
    ]
    summary = build_risk_summary(alerts, node_count=2)
    assert summary["patterns_detected"] == ["rapid_domain_expansion"]


# ---------------------------------------------------------------------------
# PredictiveMonitorAgent — heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_empty_world_model():
    agent = PredictiveMonitorAgent()
    result = await agent.run("mem-1", _ctx())
    assert result.status == "success"
    assert result.outputs["state_alerts"] == []
    assert result.outputs["risk_summary"]["total_alerts"] == 0
    assert result.outputs["confidence"] == 0.4
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_detects_rapid_expansion():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_ACME])
    result = await agent.run("mem-1", ctx)
    assert result.status == "success"
    patterns = [a["pattern"] for a in result.outputs["state_alerts"]]
    assert "rapid_domain_expansion" in patterns


@pytest.mark.asyncio
async def test_run_detects_low_confidence():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_BUDGET])
    result = await agent.run("mem-1", ctx)
    patterns = [a["pattern"] for a in result.outputs["state_alerts"]]
    assert "low_confidence_cross_silo" in patterns


@pytest.mark.asyncio
async def test_run_detects_edge_cluster():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(
        world_nodes=[_NODE_ACME, _NODE_BUDGET],
        entity_edges=[_EDGE_HEAVY_1, _EDGE_HEAVY_2],
    )
    result = await agent.run("mem-1", ctx)
    patterns = [a["pattern"] for a in result.outputs["state_alerts"]]
    assert "high_weight_edge_cluster" in patterns


@pytest.mark.asyncio
async def test_run_detects_unresolved_expansion():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_CONCEPT])
    result = await agent.run("mem-1", ctx)
    patterns = [a["pattern"] for a in result.outputs["state_alerts"]]
    assert "unresolved_expansion" in patterns


@pytest.mark.asyncio
async def test_run_detects_cascade():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(
        world_nodes=[_NODE_ACME],
        state_changes=[_CHANGE_ACME, _CHANGE_BUDGET],
    )
    result = await agent.run("mem-1", ctx)
    patterns = [a["pattern"] for a in result.outputs["state_alerts"]]
    assert "state_change_cascade" in patterns


@pytest.mark.asyncio
async def test_run_alerts_sorted_by_risk_desc():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(
        world_nodes=[_NODE_ACME, _NODE_BUDGET, _NODE_CONCEPT],
        entity_edges=[_EDGE_HEAVY_1, _EDGE_HEAVY_2],
        state_changes=[_CHANGE_ACME, _CHANGE_BUDGET],
    )
    result = await agent.run("mem-1", ctx)
    scores = [a["risk_score"] for a in result.outputs["state_alerts"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_run_risk_summary_populated():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_ACME])
    result = await agent.run("mem-1", ctx)
    summary = result.outputs["risk_summary"]
    assert summary["monitored_nodes"] == 1
    assert isinstance(summary["patterns_detected"], list)


@pytest.mark.asyncio
async def test_run_confidence_grows_with_high_risk():
    agent = PredictiveMonitorAgent()
    ctx_rich = _ctx(
        world_nodes=[_NODE_ACME, _NODE_BUDGET],
        state_changes=[_CHANGE_ACME, _CHANGE_BUDGET],
    )
    result_rich = await agent.run("mem-1", ctx_rich)
    result_empty = await agent.run("mem-2", _ctx())
    assert result_rich.outputs["confidence"] >= result_empty.outputs["confidence"]


@pytest.mark.asyncio
async def test_run_uses_heuristic_when_no_nodes():
    agent = PredictiveMonitorAgent()
    result = await agent.run("mem-1", _ctx())
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_heuristic_strategy_override():
    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_ACME])
    with patch("app.agents.predictive_monitor_agent.settings") as mock_settings:
        mock_settings.PREDICTIVE_MONITOR_STRATEGY = "heuristic"
        mock_settings.PREDICTIVE_MONITOR_THRESHOLD = None
        mock_settings.AGENT_STRATEGY = "heuristic"
        result = await agent.run("mem-1", ctx)
    assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# PredictiveMonitorAgent — LLM path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_llm_path_uses_response():
    llm_output = {
        "state_alerts": [
            {
                "pattern": "rapid_domain_expansion",
                "risk_score": 0.80,
                "affected_entities": ["acme_corp"],
                "predicted_outcome": "multi_team_blast_radius_if_entity_fails",
                "severity": "critical",
            }
        ],
        "risk_summary": {
            "total_alerts": 1,
            "high_risk_alerts": 1,
            "patterns_detected": ["rapid_domain_expansion"],
            "monitored_nodes": 1,
        },
        "confidence": 0.85,
        "rationale": "llm",
    }
    mock_client = AsyncMock()
    mock_client.complete_json = AsyncMock(return_value=llm_output)

    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_ACME])

    with patch("app.agents.predictive_monitor_agent.settings") as mock_settings:
        mock_settings.PREDICTIVE_MONITOR_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.PREDICTIVE_MONITOR_THRESHOLD = None
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        with patch(
            "app.agents.predictive_monitor_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run("mem-1", ctx)

    assert result.status == "success"
    assert result.outputs["rationale"] == "llm"
    assert result.outputs["state_alerts"][0]["pattern"] == "rapid_domain_expansion"


@pytest.mark.asyncio
async def test_run_llm_fallback_on_bad_response():
    mock_client = AsyncMock()
    mock_client.complete_json = AsyncMock(return_value={"bad": "response"})

    agent = PredictiveMonitorAgent()
    ctx = _ctx(world_nodes=[_NODE_ACME])

    with patch("app.agents.predictive_monitor_agent.settings") as mock_settings:
        mock_settings.PREDICTIVE_MONITOR_STRATEGY = "llm"
        mock_settings.AGENT_STRATEGY = "llm"
        mock_settings.PREDICTIVE_MONITOR_THRESHOLD = None
        mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
        mock_settings.OLLAMA_MODEL = "llama3.1:8b"
        mock_settings.OLLAMA_TIMEOUT_SECONDS = 5.0
        mock_settings.OLLAMA_MAX_CONCURRENCY = 2
        with patch(
            "app.agents.predictive_monitor_agent.create_ollama_client",
            return_value=mock_client,
        ):
            result = await agent.run("mem-1", ctx)

    assert result.status == "success"
    assert result.outputs["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# validate_outputs
# ---------------------------------------------------------------------------

def test_validate_outputs_passes_valid():
    agent = PredictiveMonitorAgent()
    result = _make_result(
        {
            "state_alerts": [
                {
                    "pattern": "rapid_domain_expansion",
                    "risk_score": 0.75,
                    "affected_entities": ["acme_corp"],
                    "predicted_outcome": "multi_team_blast_radius_if_entity_fails",
                    "severity": "critical",
                }
            ],
            "risk_summary": {
                "total_alerts": 1,
                "high_risk_alerts": 1,
                "patterns_detected": ["rapid_domain_expansion"],
                "monitored_nodes": 1,
            },
            "confidence": 0.8,
            "rationale": "heuristic",
        }
    )
    agent.validate_outputs(result)  # should not raise


def test_validate_outputs_fails_missing_pattern():
    agent = PredictiveMonitorAgent()
    result = _make_result(
        {
            "state_alerts": [{"risk_score": 0.8, "affected_entities": ["x"]}],
            "risk_summary": {},
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="pattern"):
        agent.validate_outputs(result)


def test_validate_outputs_fails_missing_risk_score():
    agent = PredictiveMonitorAgent()
    result = _make_result(
        {
            "state_alerts": [{"pattern": "rapid_domain_expansion", "affected_entities": ["x"]}],
            "risk_summary": {},
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="risk_score"):
        agent.validate_outputs(result)


def test_validate_outputs_fails_missing_affected_entities():
    agent = PredictiveMonitorAgent()
    result = _make_result(
        {
            "state_alerts": [{"pattern": "rapid_domain_expansion", "risk_score": 0.7}],
            "risk_summary": {},
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="affected_entities"):
        agent.validate_outputs(result)


def test_validate_outputs_skips_non_success():
    agent = PredictiveMonitorAgent()
    now = datetime.now(timezone.utc)
    result = AgentResult(
        agent_name="PredictiveMonitorAgent",
        agent_version="v1",
        memory_id="mem-1",
        status="failed",
        confidence=0.0,
        outputs={},
        warnings=[],
        errors=["something failed"],
        started_at=now,
        finished_at=now,
    )
    agent.validate_outputs(result)  # should not raise


def test_validate_outputs_state_alerts_not_list():
    agent = PredictiveMonitorAgent()
    result = _make_result({"state_alerts": "bad", "risk_summary": {}, "confidence": 0.5})
    with pytest.raises(ValueError, match="state_alerts must be a list"):
        agent.validate_outputs(result)


def test_validate_outputs_risk_summary_not_dict():
    agent = PredictiveMonitorAgent()
    result = _make_result({"state_alerts": [], "risk_summary": "bad", "confidence": 0.5})
    with pytest.raises(ValueError, match="risk_summary must be a dict"):
        agent.validate_outputs(result)


# ---------------------------------------------------------------------------
# Registry + dependencies
# ---------------------------------------------------------------------------

def test_registry_lookup_by_name():
    agent = get_agent("predictive_monitor")
    assert isinstance(agent, PredictiveMonitorAgent)


def test_registry_lookup_by_class_name():
    agent = get_agent("predictivemonitoragent")
    assert isinstance(agent, PredictiveMonitorAgent)


def test_dependencies_returns_world_model():
    agent = PredictiveMonitorAgent()
    assert "WorldModelAgent" in agent.dependencies()


def test_agent_name_and_version():
    agent = PredictiveMonitorAgent()
    assert agent.name == "PredictiveMonitorAgent"
    assert agent.version == "v1"
