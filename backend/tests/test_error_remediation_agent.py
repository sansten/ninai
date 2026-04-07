"""Tests for ErrorRemediationAgent (Phase 83)."""
from __future__ import annotations

import pytest

from app.agents.error_remediation_agent import (
    ErrorRemediationAgent,
    _ERROR_SOURCES,
    _HIGH_SEVERITIES,
    _best_playbook,
    _DISPATCH_CONFIDENCE_THRESHOLD,
    run_heuristic,
)
from app.agents.types import AgentContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(
    error_source: str = "sentry",
    severity: str = "critical",
    playbook_candidates: list | None = None,
    playbook_confidence: float | None = None,
) -> AgentContext:
    enrichment: dict = {
        "error_source": error_source,
        "severity": severity,
    }
    if playbook_candidates is not None:
        enrichment["playbook_candidates"] = playbook_candidates
    if playbook_confidence is not None:
        enrichment["playbook_confidence"] = playbook_confidence
    return {"memory": {"enrichment": enrichment}}


def _candidate(pb_id: str = "pb-1", confidence: float = 0.90) -> dict:
    return {"id": pb_id, "success_rate": confidence}


AGENT = ErrorRemediationAgent()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_error_sources_contains_sentry():
    assert "sentry" in _ERROR_SOURCES


def test_error_sources_contains_pagerduty():
    assert "pagerduty" in _ERROR_SOURCES


def test_error_sources_contains_opsgenie():
    assert "opsgenie" in _ERROR_SOURCES


def test_high_severities_contains_critical():
    assert "critical" in _HIGH_SEVERITIES


def test_high_severities_contains_high():
    assert "high" in _HIGH_SEVERITIES


def test_high_severities_contains_p1():
    assert "p1" in _HIGH_SEVERITIES


def test_high_severities_contains_p2():
    assert "p2" in _HIGH_SEVERITIES


def test_dispatch_confidence_threshold():
    assert _DISPATCH_CONFIDENCE_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# _best_playbook
# ---------------------------------------------------------------------------

def test_best_playbook_empty_returns_none():
    pb_id, conf = _best_playbook([])
    assert pb_id is None
    assert conf == 0.0


def test_best_playbook_single_candidate():
    pb_id, conf = _best_playbook([_candidate("pb-1", 0.85)])
    assert pb_id == "pb-1"
    assert conf == 0.85


def test_best_playbook_picks_highest_success_rate():
    candidates = [_candidate("pb-1", 0.70), _candidate("pb-2", 0.90)]
    pb_id, conf = _best_playbook(candidates)
    assert pb_id == "pb-2"
    assert conf == 0.90


def test_best_playbook_uses_confidence_key():
    candidates = [{"id": "pb-x", "confidence": 0.88}]
    pb_id, conf = _best_playbook(candidates)
    assert pb_id == "pb-x"
    assert conf == 0.88


def test_best_playbook_zero_confidence_candidate():
    pb_id, conf = _best_playbook([{"id": "pb-z", "success_rate": 0.0}])
    assert pb_id == "pb-z"
    assert conf == 0.0


# ---------------------------------------------------------------------------
# run_heuristic — ignore cases
# ---------------------------------------------------------------------------

def test_ignore_unknown_source():
    result = run_heuristic({"error_source": "datadog", "severity": "critical"})
    assert result["action"] == "ignore"
    assert result["error_source"] == "datadog"


def test_ignore_low_severity():
    result = run_heuristic({"error_source": "sentry", "severity": "low"})
    assert result["action"] == "ignore"


def test_ignore_medium_severity():
    result = run_heuristic({"error_source": "pagerduty", "severity": "medium"})
    assert result["action"] == "ignore"


def test_ignore_empty_enrichment():
    result = run_heuristic({})
    assert result["action"] == "ignore"


def test_ignore_confidence_is_high():
    result = run_heuristic({"error_source": "unknown", "severity": "low"})
    assert result["confidence"] == 0.85


def test_ignore_playbook_id_is_none():
    result = run_heuristic({"error_source": "github", "severity": "critical"})
    assert result["playbook_id"] is None


# ---------------------------------------------------------------------------
# run_heuristic — review cases
# ---------------------------------------------------------------------------

def test_review_when_no_playbook_candidates():
    result = run_heuristic({"error_source": "sentry", "severity": "critical"})
    assert result["action"] == "review"


def test_review_when_playbook_confidence_below_threshold():
    result = run_heuristic({
        "error_source": "sentry",
        "severity": "high",
        "playbook_candidates": [_candidate("pb-low", 0.50)],
    })
    assert result["action"] == "review"
    assert result["playbook_id"] == "pb-low"


def test_review_confidence_field():
    result = run_heuristic({"error_source": "opsgenie", "severity": "p1"})
    assert 0.0 < result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# run_heuristic — dispatch cases
# ---------------------------------------------------------------------------

def test_dispatch_with_high_confidence_playbook():
    result = run_heuristic({
        "error_source": "sentry",
        "severity": "critical",
        "playbook_candidates": [_candidate("pb-good", 0.90)],
    })
    assert result["action"] == "dispatch"
    assert result["playbook_id"] == "pb-good"


def test_dispatch_with_exact_threshold_confidence():
    result = run_heuristic({
        "error_source": "pagerduty",
        "severity": "high",
        "playbook_candidates": [_candidate("pb-edge", 0.70)],
    })
    assert result["action"] == "dispatch"


def test_dispatch_uses_override_confidence():
    result = run_heuristic({
        "error_source": "sentry",
        "severity": "critical",
        "playbook_candidates": [_candidate("pb-1", 0.40)],
        "playbook_confidence": 0.85,
    })
    assert result["action"] == "dispatch"
    assert result["playbook_confidence"] == 0.85


def test_dispatch_confidence_is_085():
    result = run_heuristic({
        "error_source": "opsgenie",
        "severity": "p2",
        "playbook_candidates": [_candidate("pb-x", 0.80)],
    })
    assert result["confidence"] == 0.85


def test_dispatch_all_sources():
    for src in ("sentry", "pagerduty", "opsgenie"):
        result = run_heuristic({
            "error_source": src,
            "severity": "critical",
            "playbook_candidates": [_candidate("pb-1", 0.90)],
        })
        assert result["action"] == "dispatch", f"expected dispatch for {src}"


def test_dispatch_all_high_severities():
    for sev in ("high", "critical", "p1", "p2"):
        result = run_heuristic({
            "error_source": "sentry",
            "severity": sev,
            "playbook_candidates": [_candidate("pb-1", 0.90)],
        })
        assert result["action"] == "dispatch", f"expected dispatch for severity {sev}"


def test_output_has_required_keys():
    result = run_heuristic({"error_source": "sentry", "severity": "critical"})
    for key in ("action", "error_source", "severity", "playbook_id",
                "playbook_confidence", "routing_reason", "confidence", "rationale"):
        assert key in result


def test_rationale_is_heuristic():
    result = run_heuristic({"error_source": "sentry", "severity": "critical"})
    assert result["rationale"] == "heuristic"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

def test_agent_name():
    assert AGENT.name == "ErrorRemediationAgent"


def test_agent_version():
    assert AGENT.version == "v1"


def test_agent_dependencies():
    deps = AGENT.dependencies()
    assert "PlaybookAgent" in deps
    assert "AutonomousActionAgent" in deps
    assert "HumanReviewQueueAgent" in deps


@pytest.mark.asyncio
async def test_run_ignore_unknown_source():
    ctx = _ctx(error_source="datadog", severity="critical")
    result = await AGENT.run("mem-001", ctx)
    assert result.status == "success"
    assert result.outputs["action"] == "ignore"


@pytest.mark.asyncio
async def test_run_review_no_playbook():
    ctx = _ctx(error_source="sentry", severity="critical")
    result = await AGENT.run("mem-002", ctx)
    assert result.outputs["action"] == "review"


@pytest.mark.asyncio
async def test_run_dispatch_with_playbook():
    ctx = _ctx(
        error_source="pagerduty",
        severity="high",
        playbook_candidates=[_candidate("pb-1", 0.90)],
    )
    result = await AGENT.run("mem-003", ctx)
    assert result.outputs["action"] == "dispatch"
    assert result.outputs["playbook_id"] == "pb-1"


@pytest.mark.asyncio
async def test_run_confidence_in_range():
    result = await AGENT.run("mem-004", _ctx())
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_run_agent_name_in_result():
    result = await AGENT.run("mem-005", {})
    assert result.agent_name == "ErrorRemediationAgent"


@pytest.mark.asyncio
async def test_run_timestamps_set():
    result = await AGENT.run("mem-006", {})
    assert result.started_at <= result.finished_at


@pytest.mark.asyncio
async def test_run_trace_id_from_context():
    ctx: AgentContext = {"runtime": {"job_id": "job-e83"}}
    result = await AGENT.run("mem-007", ctx)
    assert result.trace_id == "job-e83"


@pytest.mark.asyncio
async def test_run_trace_id_none_when_absent():
    result = await AGENT.run("mem-008", {})
    assert result.trace_id is None


@pytest.mark.asyncio
async def test_validate_outputs_passes():
    result = await AGENT.run("mem-009", _ctx(error_source="sentry", severity="critical"))
    AGENT.validate_outputs(result)


@pytest.mark.asyncio
async def test_run_empty_context():
    result = await AGENT.run("mem-010", {})
    assert result.status == "success"
    assert result.outputs["action"] == "ignore"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def test_registry_contains_agent():
    from app.agents.registry import AGENT_CLASSES
    names = [cls.__name__ for cls in AGENT_CLASSES]
    assert "ErrorRemediationAgent" in names


def test_registry_get_agent():
    from app.agents.registry import get_agent
    agent = get_agent("error_remediation")
    assert agent is not None
    assert isinstance(agent, ErrorRemediationAgent)
