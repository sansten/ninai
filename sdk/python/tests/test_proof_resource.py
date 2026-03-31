from __future__ import annotations

from ninai.resources import ProofResource


class _DummyClient:
    def __init__(self):
        self.calls = []

    def _post(self, path, json=None):
        self.calls.append((path, json))
        if path == "/proof/scorecard":
            return {
                "lead_time_gain_pct": 25.0,
                "sla_avoidance_rate": 0.5,
                "mttr_delta_pct": 20.0,
                "false_escalation_reduction_pct": 10.0,
                "incidents_count": 2,
                "score": 21.25,
                "reproducibility_hash": "abc123abc123abc123abc123abc123ab",
            }

        return {
            "month": "2026-03",
            "tenant_id": "org-1",
            "incidents_count": 2,
            "lead_time_saved_hours": 5.0,
            "mttr_saved_hours": 4.0,
            "avoided_sla_penalty": 1000.0,
            "estimated_savings": 2200.0,
            "operating_cost": 1000.0,
            "net_impact": 1200.0,
            "roi_pct": 120.0,
            "reproducibility_hash": "def456def456def456def456def456de",
        }


def test_proof_scorecard_maps_to_typed_model():
    client = _DummyClient()
    resource = ProofResource(client)

    out = resource.scorecard(
        records=[{"incident_id": "i1", "lead_time_hours": 7.0}],
        baseline={"lead_time_hours": 10.0},
    )

    assert out.score == 21.25
    assert out.incidents_count == 2
    assert len(out.reproducibility_hash) == 32
    assert client.calls[0][0] == "/proof/scorecard"


def test_proof_monthly_impact_forwards_cost_overrides():
    client = _DummyClient()
    resource = ProofResource(client)

    out = resource.monthly_impact(
        month="2026-03",
        records=[{"incident_id": "i1", "lead_time_hours": 7.0}],
        baseline={"lead_time_hours": 10.0},
        labor_cost_per_hour=150.0,
        false_escalation_cost=300.0,
        monthly_operating_cost=1500.0,
    )

    assert out.tenant_id == "org-1"
    assert out.net_impact == 1200.0
    assert client.calls[0][0] == "/proof/monthly-impact"
    payload = client.calls[0][1]
    assert payload["labor_cost_per_hour"] == 150.0
    assert payload["false_escalation_cost"] == 300.0
    assert payload["monthly_operating_cost"] == 1500.0
