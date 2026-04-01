"""Tests for Phase 50 — Continuous Learning Pipeline.

Covers: _is_success_status, _aggregate_strategy_metrics,
_aggregate_playbook_metrics, _build_drift_alerts, OutcomeTrackerService,
StrategyEvolutionService (promote / prune / drift / no-op paths).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.continuous_learning_service import (
    _is_success_status,
    _aggregate_strategy_metrics,
    _aggregate_playbook_metrics,
    _build_drift_alerts,
    StrategyEvolutionService,
    StrategyMetric,
    PlaybookMetric,
    StrategyDriftAlert,
    EvolutionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    id: str,
    goal_type: str,
    domain: str,
    success_rate: float = 0.5,
    sample_count: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=id,
        goal_type=goal_type,
        domain=domain,
        success_rate=success_rate,
        sample_count=sample_count,
    )


def _make_svc(entries: list, outcome_batches: list[list]) -> StrategyEvolutionService:
    session = AsyncMock()
    scalars_ns = SimpleNamespace(all=lambda: entries)
    execute_result = SimpleNamespace(scalars=lambda: scalars_ns)
    session.execute = AsyncMock(return_value=execute_result)
    session.flush = AsyncMock()
    session.delete = AsyncMock()

    svc = StrategyEvolutionService(session)
    svc.tracker = AsyncMock()
    svc.tracker.load_outcomes = AsyncMock(side_effect=outcome_batches)
    return svc


def _outcome(goal_type: str, domain: str, success: bool, playbook_id: str | None = None) -> dict:
    return {
        "goal_type": goal_type,
        "domain": domain,
        "success": success,
        "matched_playbook_id": playbook_id,
    }


# ===========================================================================
# _is_success_status
# ===========================================================================

class TestIsSuccessStatus:
    def test_success_lowercase(self):
        assert _is_success_status("success") is True

    def test_success_uppercase(self):
        assert _is_success_status("SUCCESS") is True

    def test_failed_is_false(self):
        assert _is_success_status("failed") is False

    def test_denied_is_false(self):
        assert _is_success_status("denied") is False

    def test_none_is_false(self):
        assert _is_success_status(None) is False

    def test_empty_string_is_false(self):
        assert _is_success_status("") is False


# ===========================================================================
# _aggregate_strategy_metrics
# ===========================================================================

class TestAggregateStrategyMetrics:
    def test_groups_by_goal_type_and_domain(self):
        outcomes = [
            _outcome("analysis", "security", True),
            _outcome("analysis", "security", False),
            _outcome("planning", "devops", True),
        ]
        metrics = _aggregate_strategy_metrics(outcomes)
        by_key = {(m.goal_type, m.domain): m for m in metrics}
        assert by_key[("analysis", "security")].total == 2
        assert by_key[("analysis", "security")].successes == 1
        assert by_key[("analysis", "security")].success_rate == 0.5
        assert by_key[("planning", "devops")].success_rate == 1.0

    def test_empty_outcomes_returns_empty(self):
        assert _aggregate_strategy_metrics([]) == []

    def test_all_failures_zero_rate(self):
        outcomes = [_outcome("eval", "finance", False)] * 2
        metrics = _aggregate_strategy_metrics(outcomes)
        assert metrics[0].success_rate == 0.0

    def test_all_successes_full_rate(self):
        outcomes = [_outcome("gen", "general", True)] * 5
        metrics = _aggregate_strategy_metrics(outcomes)
        assert metrics[0].success_rate == 1.0
        assert metrics[0].total == 5

    def test_missing_goal_type_falls_back_to_general(self):
        outcomes = [{"goal_type": None, "domain": None, "success": True}]
        metrics = _aggregate_strategy_metrics(outcomes)
        assert metrics[0].goal_type == "general"
        assert metrics[0].domain == "general"

    def test_sorted_by_goal_type_then_domain(self):
        outcomes = [
            _outcome("z_type", "a_domain", True),
            _outcome("a_type", "z_domain", True),
        ]
        metrics = _aggregate_strategy_metrics(outcomes)
        assert metrics[0].goal_type == "a_type"

    def test_returns_strategy_metric_instances(self):
        outcomes = [_outcome("x", "y", True)]
        metrics = _aggregate_strategy_metrics(outcomes)
        assert isinstance(metrics[0], StrategyMetric)


# ===========================================================================
# _aggregate_playbook_metrics
# ===========================================================================

class TestAggregatePlaybookMetrics:
    def test_groups_by_playbook_id(self):
        outcomes = [
            _outcome("a", "b", True, "PB-1"),
            _outcome("a", "b", False, "PB-1"),
        ]
        metrics = _aggregate_playbook_metrics(outcomes)
        assert len(metrics) == 1
        assert metrics[0].total == 2
        assert metrics[0].success_rate == 0.5

    def test_ignores_empty_playbook_id(self):
        outcomes = [
            _outcome("a", "b", True, ""),
            _outcome("a", "b", True, None),
        ]
        assert _aggregate_playbook_metrics(outcomes) == []

    def test_multiple_playbooks_separated(self):
        outcomes = [
            _outcome("a", "b", True, "PB-A"),
            _outcome("a", "b", True, "PB-B"),
        ]
        metrics = _aggregate_playbook_metrics(outcomes)
        ids = {m.playbook_id for m in metrics}
        assert ids == {"PB-A", "PB-B"}

    def test_returns_playbook_metric_instances(self):
        outcomes = [_outcome("a", "b", True, "PB-1")]
        metrics = _aggregate_playbook_metrics(outcomes)
        assert isinstance(metrics[0], PlaybookMetric)

    def test_sorted_by_playbook_id(self):
        outcomes = [
            _outcome("a", "b", True, "PB-Z"),
            _outcome("a", "b", True, "PB-A"),
        ]
        metrics = _aggregate_playbook_metrics(outcomes)
        assert metrics[0].playbook_id == "PB-A"


# ===========================================================================
# _build_drift_alerts
# ===========================================================================

class TestBuildDriftAlerts:
    def test_critical_alert_on_large_drop(self):
        previous = [StrategyMetric("analysis", "security", 10, 8, 0.80)]
        current = [StrategyMetric("analysis", "security", 10, 5, 0.50)]
        alerts = _build_drift_alerts(previous=previous, current=current)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert abs(alerts[0].delta - (-0.30)) < 0.001

    def test_warning_alert_on_moderate_drop(self):
        previous = [StrategyMetric("planning", "devops", 10, 8, 0.80)]
        current = [StrategyMetric("planning", "devops", 10, 6, 0.60)]
        alerts = _build_drift_alerts(previous=previous, current=current)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"

    def test_no_alert_when_rate_improves(self):
        previous = [StrategyMetric("gen", "general", 10, 5, 0.50)]
        current = [StrategyMetric("gen", "general", 10, 9, 0.90)]
        alerts = _build_drift_alerts(previous=previous, current=current)
        assert len(alerts) == 0

    def test_no_alert_when_previous_rate_below_threshold(self):
        previous = [StrategyMetric("x", "y", 10, 3, 0.30)]
        current = [StrategyMetric("x", "y", 10, 1, 0.10)]
        alerts = _build_drift_alerts(previous=previous, current=current, min_previous_rate=0.60)
        assert len(alerts) == 0

    def test_no_alert_for_new_strategy_without_previous(self):
        current = [StrategyMetric("new", "domain", 10, 3, 0.30)]
        alerts = _build_drift_alerts(previous=[], current=current)
        assert len(alerts) == 0

    def test_alert_key_format(self):
        previous = [StrategyMetric("analysis", "finance", 10, 9, 0.90)]
        current = [StrategyMetric("analysis", "finance", 10, 6, 0.60)]
        alerts = _build_drift_alerts(previous=previous, current=current)
        assert alerts[0].strategy_key == "analysis:finance"

    def test_sorted_by_delta_ascending(self):
        previous = [
            StrategyMetric("a", "x", 10, 9, 0.90),
            StrategyMetric("b", "y", 10, 8, 0.80),
        ]
        current = [
            StrategyMetric("a", "x", 10, 5, 0.50),  # delta -0.40
            StrategyMetric("b", "y", 10, 6, 0.60),  # delta -0.20
        ]
        alerts = _build_drift_alerts(previous=previous, current=current)
        assert len(alerts) == 2
        assert alerts[0].delta < alerts[1].delta

    def test_returns_drift_alert_instances(self):
        previous = [StrategyMetric("x", "y", 10, 9, 0.90)]
        current = [StrategyMetric("x", "y", 10, 5, 0.50)]
        alerts = _build_drift_alerts(previous=previous, current=current)
        assert isinstance(alerts[0], StrategyDriftAlert)


# ===========================================================================
# StrategyEvolutionService.run() — via mocked DB
# ===========================================================================

class TestStrategyEvolutionPromotion:
    @pytest.mark.asyncio
    async def test_promotes_high_success_rate_entry(self):
        entry = _make_entry("strat-win", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", True)] * 6,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5, promote_threshold=0.80)
        assert "strat-win" in out.promoted

    @pytest.mark.asyncio
    async def test_promoted_entry_not_in_pruned(self):
        entry = _make_entry("strat-win", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", True)] * 6,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5, promote_threshold=0.80)
        assert "strat-win" not in out.pruned

    @pytest.mark.asyncio
    async def test_promotion_requires_min_samples(self):
        entry = _make_entry("strat-new", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", True)] * 3,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5, promote_threshold=0.80)
        assert out.promoted == []


class TestStrategyEvolutionPruning:
    @pytest.mark.asyncio
    async def test_prunes_low_success_rate_entry(self):
        entry = _make_entry("strat-bad", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", False, "PB-1")] * 5,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5, prune_threshold=0.20)
        assert "strat-bad" in out.pruned
        svc.session.delete.assert_awaited_once_with(entry)
        svc.session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prune_requires_min_samples(self):
        entry = _make_entry("strat-new", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", False)] * 3,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5, prune_threshold=0.20)
        assert out.pruned == []
        svc.session.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_original_prune_test(self):
        """Preserves the original acceptance test from the initial implementation."""
        entry = _make_entry("strat-1", "analysis", "security", success_rate=0.7, sample_count=3)
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", False, "PB-1")] * 5,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5)
        assert out.pruned == ["strat-1"]
        assert out.promoted == []
        svc.session.delete.assert_awaited_once_with(entry)
        svc.session.flush.assert_awaited_once()


class TestStrategyEvolutionNoOp:
    @pytest.mark.asyncio
    async def test_no_matching_entry_no_changes(self):
        entry = _make_entry("strat-orphan", "special", "niche")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", True)],
                [],
            ],
        )
        out = await svc.run(org_id="org-1")
        assert out.promoted == []
        assert out.pruned == []

    @pytest.mark.asyncio
    async def test_empty_outcomes_empty_result(self):
        svc = _make_svc([], [[], []])
        out = await svc.run(org_id="org-1")
        assert isinstance(out, EvolutionResult)
        assert out.promoted == []
        assert out.pruned == []
        assert out.strategy_metrics == []
        assert out.playbook_metrics == []
        assert out.drift_alerts == []


class TestStrategyEvolutionDrift:
    @pytest.mark.asyncio
    async def test_drift_alert_emitted_on_rate_drop(self):
        entry = _make_entry("strat-drift", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                # current: 0% success over 6
                [_outcome("analysis", "security", False)] * 6,
                # previous: 90% success
                [_outcome("analysis", "security", True)] * 9
                + [_outcome("analysis", "security", False)],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5)
        assert len(out.drift_alerts) > 0
        assert out.drift_alerts[0].strategy_key == "analysis:security"

    @pytest.mark.asyncio
    async def test_no_drift_when_stable(self):
        entry = _make_entry("strat-stable", "planning", "devops")
        svc = _make_svc(
            [entry],
            [
                [_outcome("planning", "devops", True)] * 8,
                [_outcome("planning", "devops", True)] * 8,
            ],
        )
        out = await svc.run(org_id="org-1")
        assert out.drift_alerts == []


class TestStrategyEvolutionMetrics:
    @pytest.mark.asyncio
    async def test_playbook_metrics_populated(self):
        entry = _make_entry("strat-1", "analysis", "security")
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", True, "PB-99")] * 6,
                [],
            ],
        )
        out = await svc.run(org_id="org-1", min_samples=5)
        pb_ids = [m.playbook_id for m in out.playbook_metrics]
        assert "PB-99" in pb_ids

    @pytest.mark.asyncio
    async def test_success_rate_updated_on_entry(self):
        entry = _make_entry("strat-x", "planning", "devops", success_rate=0.5)
        svc = _make_svc(
            [entry],
            [
                [_outcome("planning", "devops", True)] * 8
                + [_outcome("planning", "devops", False)] * 2,
                [],
            ],
        )
        await svc.run(org_id="org-1")
        assert abs(entry.success_rate - 0.8) < 0.01

    @pytest.mark.asyncio
    async def test_sample_count_at_least_window_total(self):
        entry = _make_entry("strat-y", "analysis", "security", sample_count=2)
        svc = _make_svc(
            [entry],
            [
                [_outcome("analysis", "security", True)] * 4,
                [],
            ],
        )
        await svc.run(org_id="org-1")
        assert entry.sample_count >= 4
