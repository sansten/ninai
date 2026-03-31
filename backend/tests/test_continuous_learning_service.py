from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.continuous_learning_service import (
    _is_success_status,
    _aggregate_strategy_metrics,
    _aggregate_playbook_metrics,
    _build_drift_alerts,
    StrategyMetric,
    StrategyEvolutionService,
)


def test_is_success_status_true_only_for_success() -> None:
    assert _is_success_status("success") is True
    assert _is_success_status("SUCCESS") is True
    assert _is_success_status("failed") is False
    assert _is_success_status(None) is False


def test_aggregate_strategy_metrics_groups_and_rates() -> None:
    outcomes = [
        {"goal_type": "analysis", "domain": "security", "success": True},
        {"goal_type": "analysis", "domain": "security", "success": False},
        {"goal_type": "planning", "domain": "devops", "success": True},
    ]

    metrics = _aggregate_strategy_metrics(outcomes)
    by_key = {(m.goal_type, m.domain): m for m in metrics}

    sec = by_key[("analysis", "security")]
    assert sec.total == 2
    assert sec.successes == 1
    assert sec.success_rate == 0.5

    dev = by_key[("planning", "devops")]
    assert dev.total == 1
    assert dev.success_rate == 1.0


def test_aggregate_playbook_metrics_ignores_missing_ids() -> None:
    outcomes = [
        {"matched_playbook_id": "PB-1", "success": True},
        {"matched_playbook_id": "PB-1", "success": False},
        {"matched_playbook_id": "", "success": True},
        {"matched_playbook_id": None, "success": True},
    ]

    metrics = _aggregate_playbook_metrics(outcomes)
    assert len(metrics) == 1
    assert metrics[0].playbook_id == "PB-1"
    assert metrics[0].total == 2
    assert metrics[0].success_rate == 0.5


def test_build_drift_alerts_when_rate_drops() -> None:
    previous = [StrategyMetric("analysis", "security", 10, 8, 0.8)]
    current = [StrategyMetric("analysis", "security", 10, 5, 0.5)]

    alerts = _build_drift_alerts(previous=previous, current=current)
    assert len(alerts) == 1
    assert alerts[0].strategy_key == "analysis:security"
    assert alerts[0].delta == -0.3
    assert alerts[0].severity == "critical"


@pytest.mark.asyncio
async def test_strategy_evolution_prunes_low_performing_entries() -> None:
    entry = SimpleNamespace(
        id="strat-1",
        goal_type="analysis",
        domain="security",
        success_rate=0.7,
        sample_count=3,
    )

    session = AsyncMock()
    result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: [entry]))
    session.execute = AsyncMock(return_value=result)
    session.flush = AsyncMock()
    session.delete = AsyncMock()

    svc = StrategyEvolutionService(session)

    svc.tracker = AsyncMock()
    svc.tracker.load_outcomes = AsyncMock(
        side_effect=[
            [
                {
                    "goal_type": "analysis",
                    "domain": "security",
                    "success": False,
                    "matched_playbook_id": "PB-1",
                },
                {
                    "goal_type": "analysis",
                    "domain": "security",
                    "success": False,
                    "matched_playbook_id": "PB-1",
                },
                {
                    "goal_type": "analysis",
                    "domain": "security",
                    "success": False,
                    "matched_playbook_id": "PB-1",
                },
                {
                    "goal_type": "analysis",
                    "domain": "security",
                    "success": False,
                    "matched_playbook_id": "PB-1",
                },
                {
                    "goal_type": "analysis",
                    "domain": "security",
                    "success": False,
                    "matched_playbook_id": "PB-1",
                },
            ],
            [],
        ]
    )

    out = await svc.run(org_id="org-1", min_samples=5)

    assert out.pruned == ["strat-1"]
    assert out.promoted == []
    session.delete.assert_awaited_once_with(entry)
    session.flush.assert_awaited_once()
