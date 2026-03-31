"""Tests for ProofScorecardService (Phase 54 Slice 1)."""

from __future__ import annotations

from app.services.proof_scorecard_service import ProofScorecardService


class TestProofScorecardService:
    def test_scorecard_computes_core_metrics(self):
        svc = ProofScorecardService()
        baseline = {
            "lead_time_hours": 10.0,
            "mttr_hours": 8.0,
            "false_escalation_rate": 0.25,
        }
        records = [
            {
                "incident_id": "i1",
                "lead_time_hours": 7.0,
                "mttr_hours": 6.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
            {
                "incident_id": "i2",
                "lead_time_hours": 8.0,
                "mttr_hours": 6.5,
                "avoided_sla_breach": False,
                "false_escalation": False,
            },
        ]

        out = svc.compute_scorecard(records=records, baseline=baseline)

        assert out.incidents_count == 2
        assert out.lead_time_gain_pct > 0
        assert out.mttr_delta_pct > 0
        assert out.sla_avoidance_rate == 0.5
        assert out.false_escalation_reduction_pct >= 0
        assert out.score >= 0

    def test_monthly_roi_report_computes_net_impact(self):
        svc = ProofScorecardService()
        baseline = {
            "lead_time_hours": 10.0,
            "mttr_hours": 8.0,
            "false_escalation_rate": 0.25,
            "sla_penalty_per_breach": 1500.0,
        }
        records = [
            {
                "incident_id": "i1",
                "lead_time_hours": 7.0,
                "mttr_hours": 5.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
            {
                "incident_id": "i2",
                "lead_time_hours": 8.0,
                "mttr_hours": 6.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
        ]

        out = svc.compute_monthly_roi_report(
            tenant_id="org-1",
            month="2026-03",
            records=records,
            baseline=baseline,
            labor_cost_per_hour=100.0,
            monthly_operating_cost=1000.0,
        )

        assert out.tenant_id == "org-1"
        assert out.month == "2026-03"
        assert out.incidents_count == 2
        assert out.estimated_savings > 0
        assert out.net_impact > 0
        assert out.roi_pct > 0

    def test_reproducibility_hash_is_deterministic(self):
        svc = ProofScorecardService()
        baseline = {
            "lead_time_hours": 10.0,
            "mttr_hours": 8.0,
            "false_escalation_rate": 0.25,
        }
        records_a = [
            {
                "incident_id": "i2",
                "lead_time_hours": 8.0,
                "mttr_hours": 6.0,
                "avoided_sla_breach": True,
                "false_escalation": False,
            },
            {
                "incident_id": "i1",
                "lead_time_hours": 7.0,
                "mttr_hours": 5.0,
                "avoided_sla_breach": False,
                "false_escalation": True,
            },
        ]
        records_b = [records_a[1], records_a[0]]

        h1 = svc.reproducibility_hash(records=records_a, baseline=baseline)
        h2 = svc.reproducibility_hash(records=records_b, baseline=baseline)

        assert h1 == h2
        assert len(h1) == 32

    def test_reproducibility_hash_changes_when_data_changes(self):
        svc = ProofScorecardService()
        baseline = {
            "lead_time_hours": 10.0,
            "mttr_hours": 8.0,
            "false_escalation_rate": 0.25,
        }
        records = [
            {
                "incident_id": "i1",
                "lead_time_hours": 7.0,
                "mttr_hours": 5.0,
                "avoided_sla_breach": False,
                "false_escalation": True,
            }
        ]

        h1 = svc.reproducibility_hash(records=records, baseline=baseline)
        records[0]["lead_time_hours"] = 7.5
        h2 = svc.reproducibility_hash(records=records, baseline=baseline)

        assert h1 != h2
