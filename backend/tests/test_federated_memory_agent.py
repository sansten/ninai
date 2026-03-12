"""Tests for FederatedMemoryAgent — Phase 44."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.federated_memory_agent import (
    FederatedMemoryAgent,
    filter_k_anonymous_candidates,
    rank_federated_candidates,
    summarize_federated_candidates,
    percentile_rank,
    build_benchmark_insight,
    build_sharing_recommendation,
    compute_federated_confidence,
)
from app.agents.types import AgentContext
from app.core.config import settings
from app.services.differential_privacy_service import DifferentialPrivacyService


def _ctx(
    *,
    federated_candidates: list | None = None,
    org_metrics: dict | None = None,
    peer_metrics: dict | None = None,
    benchmark_metric: str = "memory_quality",
    target_percentile: float = 0.75,
    min_contributors: int = 5,
    privacy_budget_epsilon: float = 1.0,
) -> AgentContext:
    return {
        "memory": {
            "content": "federated benchmark request",
            "enrichment": {
                "federated_candidates": federated_candidates or [],
                "org_metrics": org_metrics or {},
                "peer_metrics": peer_metrics or {},
                "benchmark_metric": benchmark_metric,
                "target_percentile": target_percentile,
                "min_contributors": min_contributors,
                "privacy_budget_epsilon": privacy_budget_epsilon,
            },
        },
        "runtime": {"job_id": "trace-44"},
    }


AGENT = FederatedMemoryAgent()

CANDIDATES = [
    {
        "summary_type": "pattern",
        "domain": "incident_response",
        "quality_score": 0.82,
        "count_contributing_orgs": 9,
        "success_rate": 0.88,
        "content": {"description": "escalate within 15 minutes"},
    },
    {
        "summary_type": "playbook",
        "domain": "incident_response",
        "quality_score": 0.73,
        "count_contributing_orgs": 6,
        "success_rate": 0.81,
        "content": {"title": "high-severity incident runbook"},
    },
    {
        "summary_type": "benchmark",
        "domain": "support",
        "quality_score": 0.55,
        "count_contributing_orgs": 3,
        "success_rate": 0.62,
        "content": {"description": "median first response time below 10m"},
    },
]


def test_filter_k_anonymous_candidates():
    filtered = filter_k_anonymous_candidates(CANDIDATES, min_contributors=5)
    assert len(filtered) == 2
    assert all(c["count_contributing_orgs"] >= 5 for c in filtered)


def test_filter_k_anonymous_candidates_skips_invalid_items():
    filtered = filter_k_anonymous_candidates(CANDIDATES + ["bad"], min_contributors=1)
    assert all(isinstance(item, dict) for item in filtered)


def test_rank_federated_candidates_quality_then_orgs():
    ranked = rank_federated_candidates(CANDIDATES)
    assert ranked[0]["quality_score"] >= ranked[1]["quality_score"]


def test_summarize_federated_candidates_empty():
    summary = summarize_federated_candidates([])
    assert summary["candidate_count"] == 0
    assert summary["avg_quality"] == 0.0


def test_summarize_federated_candidates_non_empty():
    summary = summarize_federated_candidates(CANDIDATES[:2])
    assert summary["candidate_count"] == 2
    assert "incident_response" in summary["top_domains"]
    assert summary["avg_quality"] > 0.7


def test_percentile_rank_basic():
    rank = percentile_rank(80, [50, 70, 80, 90])
    assert 0.7 <= rank <= 0.8


def test_percentile_rank_empty_peers():
    assert percentile_rank(10, []) == 0.0


def test_build_benchmark_insight_no_peers():
    insight = build_benchmark_insight(
        org_metrics={"memory_quality": 0.7},
        peer_metrics={},
        benchmark_metric="memory_quality",
        target_percentile=0.75,
        dp_service=DifferentialPrivacyService(epsilon=1.0),
    )
    assert insight["status"] == "insufficient_peer_data"


def test_build_benchmark_insight_with_peers():
    insight = build_benchmark_insight(
        org_metrics={"memory_quality": 0.72},
        peer_metrics={"memory_quality": [0.4, 0.6, 0.7, 0.8, 0.9]},
        benchmark_metric="memory_quality",
        target_percentile=0.75,
        dp_service=DifferentialPrivacyService(epsilon=1.0),
    )
    assert insight["peer_count"] == 5
    assert insight["status"] in {"on_track", "below_target"}
    assert isinstance(insight["private_peer_mean"], float)


def test_build_sharing_recommendation_insufficient():
    rec = build_sharing_recommendation(privacy_risk_score=0.2, candidate_count=0, avg_quality=0.8)
    assert rec.startswith("hold")


def test_build_sharing_recommendation_restrict_high_risk():
    rec = build_sharing_recommendation(privacy_risk_score=0.8, candidate_count=4, avg_quality=0.8)
    assert "restrict" in rec


def test_build_sharing_recommendation_public_when_high_quality_low_risk():
    rec = build_sharing_recommendation(privacy_risk_score=0.2, candidate_count=4, avg_quality=0.8)
    assert "public_best_practice" in rec


def test_compute_federated_confidence_bounds():
    conf = compute_federated_confidence(
        candidate_count=10,
        avg_quality=0.85,
        peer_count=20,
        privacy_risk_score=0.2,
    )
    assert 0.0 <= conf <= 1.0


def test_compute_federated_confidence_low_signal():
    conf = compute_federated_confidence(
        candidate_count=0,
        avg_quality=0.0,
        peer_count=0,
        privacy_risk_score=0.9,
    )
    assert conf <= 0.45


def test_validate_outputs_success_shape():
    from app.agents.types import AgentResult
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    result = AgentResult(
        agent_name=AGENT.name,
        agent_version=AGENT.version,
        memory_id="m1",
        status="success",
        confidence=0.7,
        outputs={
            "federated_summary": {},
            "benchmark_insight": {},
            "sharing_recommendation": "share: controlled_peer_exchange",
            "privacy_risk_score": 0.2,
            "federated_confidence": 0.7,
            "confidence": 0.7,
            "rationale": "heuristic",
        },
        started_at=now,
        finished_at=now,
    )
    AGENT.validate_outputs(result)


def test_validate_outputs_rejects_invalid_rationale():
    from app.agents.types import AgentResult
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    result = AgentResult(
        agent_name=AGENT.name,
        agent_version=AGENT.version,
        memory_id="m1",
        status="success",
        confidence=0.7,
        outputs={
            "federated_summary": {},
            "benchmark_insight": {},
            "sharing_recommendation": "x",
            "privacy_risk_score": 0.2,
            "federated_confidence": 0.7,
            "confidence": 0.7,
            "rationale": "other",
        },
        started_at=now,
        finished_at=now,
    )
    with pytest.raises(ValueError):
        AGENT.validate_outputs(result)


@pytest.mark.asyncio
async def test_run_heuristic_outputs_all_keys(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "heuristic")

    result = await AGENT.run(
        "m44",
        _ctx(
            federated_candidates=CANDIDATES,
            org_metrics={"memory_quality": 0.7},
            peer_metrics={"memory_quality": [0.4, 0.6, 0.75, 0.8, 0.9]},
        ),
    )

    assert result.status == "success"
    required = {
        "federated_summary",
        "benchmark_insight",
        "sharing_recommendation",
        "privacy_risk_score",
        "federated_confidence",
        "confidence",
        "rationale",
    }
    assert required.issubset(result.outputs.keys())


@pytest.mark.asyncio
async def test_run_heuristic_applies_k_anonymity(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "heuristic")

    result = await AGENT.run(
        "m45",
        _ctx(
            federated_candidates=CANDIDATES,
            min_contributors=7,
            org_metrics={"memory_quality": 0.6},
            peer_metrics={"memory_quality": [0.5, 0.6, 0.7]},
        ),
    )

    assert result.outputs["federated_summary"]["candidate_count"] == 1


@pytest.mark.asyncio
async def test_run_heuristic_insufficient_data(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "heuristic")

    result = await AGENT.run("m46", _ctx())
    assert result.outputs["benchmark_insight"]["status"] == "insufficient_peer_data"
    assert result.outputs["sharing_recommendation"].startswith("hold")


@pytest.mark.asyncio
async def test_run_llm_uses_response_when_valid(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "llm")

    mock_client = AsyncMock()
    mock_client.complete_json.return_value = {
        "federated_summary": {"candidate_count": 3},
        "benchmark_insight": {"metric": "memory_quality", "peer_count": 6},
        "sharing_recommendation": "share: controlled_peer_exchange",
        "privacy_risk_score": 0.25,
        "federated_confidence": 0.78,
        "confidence": 0.78,
        "rationale": "llm",
    }

    with patch("app.agents.federated_memory_agent.create_ollama_client", return_value=mock_client):
        result = await AGENT.run(
            "m47",
            _ctx(
                federated_candidates=CANDIDATES,
                org_metrics={"memory_quality": 0.71},
                peer_metrics={"memory_quality": [0.6, 0.65, 0.8]},
            ),
        )

    assert result.outputs["rationale"] == "llm"
    assert result.outputs["sharing_recommendation"].startswith("share")


@pytest.mark.asyncio
async def test_run_llm_falls_back_on_invalid_response(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_STRATEGY", "llm")

    mock_client = AsyncMock()
    mock_client.complete_json.return_value = {"bad": "shape"}

    with patch("app.agents.federated_memory_agent.create_ollama_client", return_value=mock_client):
        result = await AGENT.run("m48", _ctx(federated_candidates=CANDIDATES))

    assert result.outputs["rationale"] == "heuristic"


def test_agent_metadata():
    assert AGENT.name == "FederatedMemoryAgent"
    assert AGENT.version == "v1"


def test_agent_dependencies():
    deps = AGENT.dependencies()
    assert "PlaybookAgent" in deps
    assert "CredibilityAgent" in deps


def test_registry_lookup():
    from app.agents.registry import get_agent

    assert isinstance(get_agent("federated_memory"), FederatedMemoryAgent)
    assert isinstance(get_agent("federatedmemoryagent"), FederatedMemoryAgent)
    assert isinstance(get_agent("collective_intelligence"), FederatedMemoryAgent)


def test_registry_miss():
    from app.agents.registry import get_agent

    assert get_agent("federated_memory_xyz") is None
