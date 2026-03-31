"""Tests for StrategyGovernanceService (Phase 53 Slice 1)."""

from __future__ import annotations

from app.services.strategy_governance_service import (
    StrategyGovernanceService,
    StrategyWindowMetrics,
)


def _m(win: float, fp: float, n: int, drift: str = "none") -> StrategyWindowMetrics:
    return StrategyWindowMetrics(
        win_rate=win,
        false_positive_rate=fp,
        sample_count=n,
        drift_severity=drift,
    )


class TestStrategyGovernanceTransitions:
    def test_auto_revert_when_drift_is_critical(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_strategy_transition(
            current_stage="canary",
            baseline=_m(0.60, 0.20, 200),
            candidate=_m(0.70, 0.18, 200, drift="critical"),
        )

        assert decision.action == "revert"
        assert decision.auto_revert is True
        assert decision.next_stage == "stable"

    def test_hold_when_sample_size_too_low(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_strategy_transition(
            current_stage="canary",
            baseline=_m(0.60, 0.20, 200),
            candidate=_m(0.70, 0.15, 20),
        )

        assert decision.action == "hold"

    def test_promote_canary_to_stable_on_thresholds(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_strategy_transition(
            current_stage="canary",
            baseline=_m(0.50, 0.20, 200),
            candidate=_m(0.56, 0.15, 220),
        )

        assert decision.action == "promote"
        assert decision.next_stage == "stable"

    def test_demote_canary_on_false_positive_regression(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_strategy_transition(
            current_stage="canary",
            baseline=_m(0.60, 0.20, 220),
            candidate=_m(0.61, 0.25, 230),
        )

        assert decision.action == "demote"
        assert decision.next_stage == "experimental"


class TestPlaybookEfficacyGovernance:
    def test_promote_playbook_when_success_rate_high(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_playbook_efficacy(success_rate=0.81, sample_count=80)
        assert decision.status == "promote"

    def test_demote_playbook_when_success_rate_low(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_playbook_efficacy(success_rate=0.31, sample_count=80)
        assert decision.status == "demote"

    def test_keep_when_insufficient_samples(self):
        svc = StrategyGovernanceService()
        decision = svc.evaluate_playbook_efficacy(success_rate=0.95, sample_count=10)
        assert decision.status == "keep"


class TestABRoutingAndConvergence:
    def test_ab_routing_is_deterministic_per_subject(self):
        svc = StrategyGovernanceService()
        first = svc.route_ab_strategy(
            org_id="org-1",
            subject_id="session-123",
            stable_strategy_id="stable-v1",
            candidate_strategy_id="candidate-v2",
            candidate_rollout_pct=25,
        )
        second = svc.route_ab_strategy(
            org_id="org-1",
            subject_id="session-123",
            stable_strategy_id="stable-v1",
            candidate_strategy_id="candidate-v2",
            candidate_rollout_pct=25,
        )

        assert first == second

    def test_ab_routing_respects_zero_and_full_rollout(self):
        svc = StrategyGovernanceService()
        assert svc.route_ab_strategy(
            org_id="org-1",
            subject_id="s1",
            stable_strategy_id="stable-v1",
            candidate_strategy_id="candidate-v2",
            candidate_rollout_pct=0,
        ) == "stable-v1"
        assert svc.route_ab_strategy(
            org_id="org-1",
            subject_id="s1",
            stable_strategy_id="stable-v1",
            candidate_strategy_id="candidate-v2",
            candidate_rollout_pct=100,
        ) == "candidate-v2"

    def test_convergence_picks_candidate_when_uplift_strong(self):
        svc = StrategyGovernanceService()
        result = svc.evaluate_ab_convergence(
            stable=_m(0.50, 0.20, 200),
            candidate=_m(0.56, 0.19, 210),
        )

        assert result.converged is True
        assert result.winner == "candidate"

    def test_convergence_requires_sample_size(self):
        svc = StrategyGovernanceService()
        result = svc.evaluate_ab_convergence(
            stable=_m(0.50, 0.20, 40),
            candidate=_m(0.70, 0.10, 40),
        )

        assert result.converged is False
        assert result.winner is None


class TestAuditPayload:
    def test_build_audit_payload_contains_key_fields(self):
        svc = StrategyGovernanceService()
        baseline = _m(0.50, 0.20, 200)
        candidate = _m(0.56, 0.15, 220)
        decision = svc.evaluate_strategy_transition(
            current_stage="canary",
            baseline=baseline,
            candidate=candidate,
        )

        payload = svc.build_audit_payload(
            org_id="org-1",
            strategy_id="candidate-v2",
            stage="canary",
            decision=decision,
            baseline=baseline,
            candidate=candidate,
        )

        assert payload["org_id"] == "org-1"
        assert payload["strategy_id"] == "candidate-v2"
        assert payload["action"] in {"promote", "demote", "hold", "revert"}
        assert "baseline" in payload
        assert "candidate" in payload
