from __future__ import annotations

from app.services.prospective_simulation_service import (
    ProspectiveSimulationService,
    _steps_for_horizon,
    _variable_impact,
    _project_variables,
    _variable_summaries,
)


class TestProspectiveSimulationHelpers:

    def test_steps_for_horizon_short(self):
        # 7 days → 1 week → 3 steps
        assert _steps_for_horizon(7) == 3

    def test_steps_for_horizon_long(self):
        # Very long horizon is capped at MAX_STEPS=10
        assert _steps_for_horizon(365) == 10

    def test_steps_for_horizon_min(self):
        assert _steps_for_horizon(1) >= 1

    def test_variable_impact_risk_metric_increases(self):
        mult = _variable_impact("increase", 1.0, "incident_rate")
        assert mult > 1.0

    def test_variable_impact_quality_metric_falls_on_increase(self):
        mult = _variable_impact("increase", 1.0, "customer_satisfaction")
        assert mult < 1.0

    def test_variable_impact_stable_unchanged(self):
        assert _variable_impact("stable", 0.9, "incident_rate") == 1.0

    def test_project_variables_empty_episodes(self):
        result = _project_variables([], ["incident_rate"])
        assert result["incident_rate"] == []

    def test_project_variables_tracks_increase(self):
        episodes = [
            {"severity_change": "increase", "probability": 1.0},
            {"severity_change": "increase", "probability": 1.0},
        ]
        result = _project_variables(episodes, ["incident_rate"])
        vals = result["incident_rate"]
        assert vals[1] > vals[0] > 1.0   # incident_rate grows with each increase

    def test_variable_summaries_trend_detection(self):
        projections = {"incident_rate": [1.0, 1.1, 1.2, 1.3]}
        summaries = _variable_summaries(projections)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.variable == "incident_rate"
        assert s.trend == "increasing"
        assert s.peak_risk_step is not None


class TestProspectiveSimulationService:

    def test_basic_simulate_returns_result(self):
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="What if we delay the deployment by 2 weeks?",
            horizon_days=14,
            variables_to_watch=["incident_rate", "customer_satisfaction"],
        )
        assert result.scenario == "What if we delay the deployment by 2 weeks?"
        assert result.horizon_days == 14
        assert result.variables_to_watch == ["incident_rate", "customer_satisfaction"]
        assert len(result.simulation_timeline) >= 1
        assert 0.0 <= result.success_probability <= 1.0
        assert isinstance(result.recommended_precautions, list)
        assert result.simulated_at != ""

    def test_variable_projections_in_timeline(self):
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="Incident during maintenance window",
            horizon_days=7,
            variables_to_watch=["incident_rate"],
        )
        for entry in result.simulation_timeline:
            assert "incident_rate" in entry.variable_projections
            assert isinstance(entry.variable_projections["incident_rate"], float)

    def test_variable_summaries_returned(self):
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="Scale up infrastructure",
            horizon_days=30,
            variables_to_watch=["latency", "availability"],
        )
        names = {vs.variable for vs in result.variable_summaries}
        assert "latency" in names
        assert "availability" in names

    def test_horizon_days_1_minimal_timeline(self):
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="Emergency patch deployment",
            horizon_days=1,
            variables_to_watch=[],
        )
        assert len(result.simulation_timeline) >= 1

    def test_with_historical_episodes_improves_quality(self):
        history = [
            {
                "content": "delay deployment scheduled maintenance",
                "tags": ["maintenance"],
                "event_description": "Deployment delay caused minor latency spike",
            },
            {
                "content": "latency spike resolved after patch",
                "tags": ["resolved"],
                "event_description": "Latency resolved within 2 hours",
            },
        ]
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="What if we delay the deployment by 2 weeks?",
            horizon_days=14,
            variables_to_watch=["latency"],
            historical_episodes=history,
        )
        assert len(result.simulation_timeline) >= 1

    def test_no_variables_to_watch(self):
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="Run a cost analysis",
            horizon_days=14,
            variables_to_watch=[],
        )
        assert result.variable_summaries == []
        for entry in result.simulation_timeline:
            assert entry.variable_projections == {}

    def test_confidence_in_range(self):
        svc = ProspectiveSimulationService()
        result = svc.simulate(
            scenario="Gradual rollout with feature flags",
            horizon_days=21,
            variables_to_watch=["incident_rate"],
        )
        assert 0.0 <= result.confidence <= 1.0
