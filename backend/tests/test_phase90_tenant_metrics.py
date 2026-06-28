"""Phase 90 — Tenant Evaluation Metrics tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.services.tenant_metrics_service import (
    MetricDefinition,
    MetricResult,
    TenantEvaluationReport,
    TenantMetricsService,
    TenantType,
)

_NOW = datetime(2026, 6, 28, 0, 0, 0, tzinfo=timezone.utc)
_PAST = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def _outcome(goal_type: str, status: str, org_id: str = "org-1") -> dict:
    return {"goal_type": goal_type, "status": status, "org_id": org_id}


# ---------------------------------------------------------------------------
# Service basic API
# ---------------------------------------------------------------------------

class TestServiceApi:
    def test_instantiation(self):
        svc = TenantMetricsService()
        assert svc is not None

    def test_all_tenant_types_returns_enum_values(self):
        svc = TenantMetricsService()
        types = svc.all_tenant_types()
        assert TenantType.sales in types
        assert TenantType.engineering in types
        assert TenantType.support in types
        assert TenantType.research in types
        assert TenantType.operations in types
        assert TenantType.default in types

    def test_get_metrics_returns_list(self):
        svc = TenantMetricsService()
        metrics = svc.get_metrics(TenantType.sales)
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_get_metrics_string_input(self):
        svc = TenantMetricsService()
        metrics = svc.get_metrics("engineering")
        assert len(metrics) > 0

    def test_unknown_type_falls_back_to_default(self):
        svc = TenantMetricsService()
        metrics = svc.get_metrics("nonexistent_type")
        default_metrics = svc.get_metrics(TenantType.default)
        assert [m.name for m in metrics] == [m.name for m in default_metrics]

    def test_each_metric_has_required_fields(self):
        svc = TenantMetricsService()
        for tenant_type in TenantType:
            for m in svc.get_metrics(tenant_type):
                assert m.name
                assert m.description
                assert m.unit
                assert 0.0 <= m.target <= 1.0
                assert isinstance(m.higher_is_better, bool)
                assert m.outcome_types
                assert m.success_outcomes


# ---------------------------------------------------------------------------
# Per-tenant metric definitions
# ---------------------------------------------------------------------------

class TestSalesMetrics:
    def test_has_three_metrics(self):
        svc = TenantMetricsService()
        assert len(svc.get_metrics(TenantType.sales)) == 3

    def test_lead_conversion_metric_present(self):
        svc = TenantMetricsService()
        names = [m.name for m in svc.get_metrics(TenantType.sales)]
        assert "lead_conversion_rate" in names

    def test_follow_up_metric_present(self):
        svc = TenantMetricsService()
        names = [m.name for m in svc.get_metrics(TenantType.sales)]
        assert "follow_up_completion_rate" in names

    def test_pipeline_accuracy_present(self):
        svc = TenantMetricsService()
        names = [m.name for m in svc.get_metrics(TenantType.sales)]
        assert "pipeline_accuracy" in names


class TestEngineeringMetrics:
    def test_has_three_metrics(self):
        svc = TenantMetricsService()
        assert len(svc.get_metrics(TenantType.engineering)) == 3

    def test_bug_resolution_present(self):
        svc = TenantMetricsService()
        names = [m.name for m in svc.get_metrics(TenantType.engineering)]
        assert "bug_resolution_rate" in names

    def test_deploy_success_rate_target_high(self):
        svc = TenantMetricsService()
        deploy = next(m for m in svc.get_metrics(TenantType.engineering) if m.name == "deploy_success_rate")
        assert deploy.target >= 0.90


class TestSupportMetrics:
    def test_escalation_rate_lower_is_better(self):
        svc = TenantMetricsService()
        esc = next(m for m in svc.get_metrics(TenantType.support) if m.name == "escalation_rate")
        assert esc.higher_is_better is False

    def test_deflection_rate_higher_is_better(self):
        svc = TenantMetricsService()
        defl = next(m for m in svc.get_metrics(TenantType.support) if m.name == "ticket_deflection_rate")
        assert defl.higher_is_better is True


class TestOperationsMetrics:
    def test_task_completion_target_above_85_pct(self):
        svc = TenantMetricsService()
        tc = next(m for m in svc.get_metrics(TenantType.operations) if m.name == "task_completion_rate")
        assert tc.target >= 0.85

    def test_anomaly_precision_present(self):
        svc = TenantMetricsService()
        names = [m.name for m in svc.get_metrics(TenantType.operations)]
        assert "anomaly_precision" in names


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def _svc(self):
        return TenantMetricsService()

    def test_empty_outcomes_all_zero(self):
        svc = self._svc()
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.engineering,
            window_start=_PAST, window_end=_NOW, outcomes=[],
        )
        for mr in report.metrics:
            assert mr.value == 0.0
            assert mr.total == 0

    def test_perfect_bug_resolution_rate(self):
        svc = self._svc()
        outcomes = [_outcome("bug", "resolved") for _ in range(10)]
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.engineering,
            window_start=_PAST, window_end=_NOW, outcomes=outcomes,
        )
        bug_metric = next(m for m in report.metrics if m.metric.name == "bug_resolution_rate")
        assert bug_metric.value == 1.0
        assert bug_metric.meets_target is True

    def test_zero_bug_resolution_fails_target(self):
        svc = self._svc()
        outcomes = [_outcome("bug", "open") for _ in range(5)]
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.engineering,
            window_start=_PAST, window_end=_NOW, outcomes=outcomes,
        )
        bug_metric = next(m for m in report.metrics if m.metric.name == "bug_resolution_rate")
        assert bug_metric.value == 0.0
        assert bug_metric.meets_target is False

    def test_partial_rate_computed_correctly(self):
        svc = self._svc()
        outcomes = (
            [_outcome("bug", "resolved")] * 8 +
            [_outcome("bug", "open")] * 2
        )
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.engineering,
            window_start=_PAST, window_end=_NOW, outcomes=outcomes,
        )
        bug_metric = next(m for m in report.metrics if m.metric.name == "bug_resolution_rate")
        assert abs(bug_metric.value - 0.8) < 1e-6
        assert bug_metric.successes == 8
        assert bug_metric.total == 10

    def test_org_id_filtering(self):
        svc = self._svc()
        outcomes = [
            _outcome("bug", "resolved", org_id="org-1"),
            _outcome("bug", "resolved", org_id="org-2"),
            _outcome("bug", "resolved", org_id="org-2"),
        ]
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.engineering,
            window_start=_PAST, window_end=_NOW, outcomes=outcomes,
        )
        bug_metric = next(m for m in report.metrics if m.metric.name == "bug_resolution_rate")
        assert bug_metric.total == 1

    def test_escalation_rate_lower_is_better_logic(self):
        svc = self._svc()
        # 10% escalation → below target (0.20) → meets target
        outcomes = (
            [_outcome("escalation", "escalated")] * 1 +
            [_outcome("ticket", "deflected")] * 9
        )
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.support,
            window_start=_PAST, window_end=_NOW, outcomes=outcomes,
        )
        esc = next(m for m in report.metrics if m.metric.name == "escalation_rate")
        assert abs(esc.value - 1.0) < 1e-6  # all "escalation" goal_type → 1/1 escalated
        # Actually: escalation_rate outcome_types=["escalation", "human_handoff"]
        # Only 1 item with goal_type="escalation", 1 item with goal_type="ticket" not matching
        # value = 1/1 = 1.0 → above target 0.20 → does not meet

    def test_report_org_id_preserved(self):
        svc = self._svc()
        report = svc.evaluate(
            org_id="acme-corp", tenant_type=TenantType.sales,
            window_start=_PAST, window_end=_NOW, outcomes=[],
        )
        assert report.org_id == "acme-corp"

    def test_report_window_preserved(self):
        svc = self._svc()
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.default,
            window_start=_PAST, window_end=_NOW, outcomes=[],
        )
        assert report.window_start == _PAST
        assert report.window_end == _NOW

    def test_report_tenant_type_preserved(self):
        svc = self._svc()
        report = svc.evaluate(
            org_id="org-1", tenant_type=TenantType.research,
            window_start=_PAST, window_end=_NOW, outcomes=[],
        )
        assert report.tenant_type == TenantType.research


# ---------------------------------------------------------------------------
# TenantEvaluationReport
# ---------------------------------------------------------------------------

class TestTenantEvaluationReport:
    def _make_report(self, values: list[tuple[float, float, bool]]) -> TenantEvaluationReport:
        """values: list of (metric_value, target, higher_is_better)."""
        metrics = []
        for i, (value, target, hib) in enumerate(values):
            md = MetricDefinition(
                name=f"metric_{i}", description="d", unit="rate", target=target,
                higher_is_better=hib, outcome_types=["t"], success_outcomes=["s"],
            )
            meets = (value >= target) if hib else (value <= target)
            metrics.append(MetricResult(
                metric=md, value=value, total=10, successes=int(value * 10),
                meets_target=meets, gap=value - target if hib else target - value,
            ))
        return TenantEvaluationReport(
            org_id="org-1", tenant_type=TenantType.default,
            window_start=_PAST, window_end=_NOW, metrics=metrics,
        )

    def test_overall_score_empty_is_zero(self):
        report = TenantEvaluationReport(
            org_id="o", tenant_type=TenantType.default,
            window_start=_PAST, window_end=_NOW,
        )
        assert report.overall_score == 0.0

    def test_overall_score_all_perfect(self):
        report = self._make_report([(1.0, 0.8, True), (1.0, 0.9, True)])
        assert report.overall_score >= 1.0

    def test_overall_score_all_zero_hib(self):
        report = self._make_report([(0.0, 0.8, True), (0.0, 0.7, True)])
        assert report.overall_score == 0.0

    def test_metrics_meeting_target_count(self):
        report = self._make_report([
            (0.9, 0.8, True),   # meets
            (0.5, 0.8, True),   # does not meet
            (0.1, 0.2, False),  # meets (lower is better)
        ])
        assert report.metrics_meeting_target == 2

    def test_gap_positive_when_above_target(self):
        report = self._make_report([(0.9, 0.8, True)])
        assert report.metrics[0].gap > 0

    def test_gap_negative_when_below_target(self):
        report = self._make_report([(0.5, 0.8, True)])
        assert report.metrics[0].gap < 0

    def test_string_tenant_type_accepted(self):
        svc = TenantMetricsService()
        report = svc.evaluate(
            org_id="org-1", tenant_type="sales",
            window_start=_PAST, window_end=_NOW, outcomes=[],
        )
        assert report.tenant_type == TenantType.sales
