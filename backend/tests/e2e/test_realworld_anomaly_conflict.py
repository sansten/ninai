"""Real-world validation: anomaly detection and conflict detection.

Dataset: IT helpdesk and enterprise event records (Kaggle-style).
Validates that anomaly thresholds and conflict detection logic produce
semantically correct verdicts against realistic enrichment snapshots.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("AGENT_STRATEGY", "heuristic")

pytest.importorskip("ninai_enterprise", reason="Enterprise-only agent validation lives in ninai-enterprise")

from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.conflict_detection_agent import ConflictDetectionAgent
from tests.e2e.data import (
    CRISIS_ENRICHMENT_SNAPSHOTS,
    HEALTHY_ENRICHMENT_SNAPSHOTS,
    get_ticket,
    make_anomaly_context,
    make_conflict_context,
)


# ---------------------------------------------------------------------------
# Anomaly detection tests
# ---------------------------------------------------------------------------


class TestAnomalyWithRealData:
    """Validate anomaly thresholds produce correct verdicts on real enrichment data."""

    async def test_healthy_enrichment_no_anomaly(self) -> None:
        snap = HEALTHY_ENRICHMENT_SNAPSHOTS[0]
        agent = AnomalyDetectionAgent()
        result = await agent.run("mem-healthy-0", make_anomaly_context(snap))

        assert result.status == "success"
        assert result.outputs["anomaly_detected"] is False
        assert result.outputs["anomaly_type"] is None
        assert result.outputs["severity"] is None

    async def test_critical_credibility_spike_detected(self) -> None:
        enrichment = {
            "credibility_score": 0.08,  # well below 0.25 threshold
            "uncertainty_score": 0.20,
            "uncertainty_level": "low",
            "conflict_count": 0,
            "temporal_confidence": 0.80,
            "decay_factor": 0.50,
            "is_stale": False,
        }
        agent = AnomalyDetectionAgent()
        result = await agent.run("mem-cred-spike", make_anomaly_context(enrichment))

        assert result.outputs["anomaly_detected"] is True
        assert "credibility" in result.outputs["anomaly_type"], (
            f"Expected credibility-type anomaly, got {result.outputs['anomaly_type']!r}"
        )
        assert "credibility_score" in result.outputs["affected_fields"]

    async def test_critical_uncertainty_surge_detected(self) -> None:
        enrichment = {
            "credibility_score": 0.75,
            "uncertainty_score": 0.92,  # above 0.80 threshold
            "uncertainty_level": "critical",
            "conflict_count": 0,
            "temporal_confidence": 0.70,
            "decay_factor": 0.45,
            "is_stale": False,
        }
        agent = AnomalyDetectionAgent()
        result = await agent.run("mem-unc-surge", make_anomaly_context(enrichment))

        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["anomaly_type"] == "uncertainty_surge"

    async def test_conflict_flood_detected(self) -> None:
        enrichment = {
            "credibility_score": 0.70,
            "uncertainty_score": 0.25,
            "uncertainty_level": "low",
            "conflict_count": 8,  # above threshold of 5
            "temporal_confidence": 0.75,
            "decay_factor": 0.50,
            "is_stale": False,
        }
        agent = AnomalyDetectionAgent()
        result = await agent.run("mem-conflict-flood", make_anomaly_context(enrichment))

        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["anomaly_type"] == "conflict_flood"
        assert "conflict_count" in result.outputs["affected_fields"]

    async def test_stale_revival_detected(self) -> None:
        # decay_factor very low but is_stale=False — contradiction
        enrichment = {
            "credibility_score": 0.75,
            "uncertainty_score": 0.20,
            "uncertainty_level": "low",
            "conflict_count": 0,
            "temporal_confidence": 0.80,
            "decay_factor": 0.05,  # below 0.10 threshold
            "is_stale": False,     # not flagged stale → revival anomaly
        }
        agent = AnomalyDetectionAgent()
        result = await agent.run("mem-stale-revival", make_anomaly_context(enrichment))

        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["anomaly_type"] == "stale_revival"

    async def test_crisis_ticket_anomaly_severity_high(self) -> None:
        snap = CRISIS_ENRICHMENT_SNAPSHOTS[0]  # credibility=0.10, uncertainty=0.91, conflicts=8
        agent = AnomalyDetectionAgent()
        result = await agent.run("mem-crisis-0", make_anomaly_context(snap))

        assert result.outputs["anomaly_detected"] is True
        assert result.outputs["severity"] in {"critical", "high"}, (
            f"Full crisis enrichment should be critical or high severity, "
            f"got {result.outputs['severity']!r}"
        )
        assert result.outputs["anomaly_score"] >= 0.70

    async def test_healthy_batch_mostly_no_anomaly(self) -> None:
        agent = AnomalyDetectionAgent()
        anomaly_count = 0
        for i, snap in enumerate(HEALTHY_ENRICHMENT_SNAPSHOTS):
            result = await agent.run(f"mem-healthy-{i}", make_anomaly_context(snap))
            if result.outputs["anomaly_detected"]:
                anomaly_count += 1

        assert anomaly_count <= 1, (
            f"Expected at most 1 anomaly in 5 healthy tickets, got {anomaly_count}"
        )

    async def test_crisis_batch_high_anomaly_rate(self) -> None:
        agent = AnomalyDetectionAgent()
        anomaly_count = 0
        for i, snap in enumerate(CRISIS_ENRICHMENT_SNAPSHOTS):
            result = await agent.run(f"mem-crisis-{i}", make_anomaly_context(snap))
            if result.outputs["anomaly_detected"]:
                anomaly_count += 1

        assert anomaly_count >= 2, (
            f"Expected at least 2 anomalies in 3 crisis tickets, got {anomaly_count}"
        )


# ---------------------------------------------------------------------------
# Conflict detection tests
# ---------------------------------------------------------------------------


class TestConflictWithRealData:
    """Validate cross-silo conflict detection against realistic signal pairs."""

    def _auth_world_nodes(self) -> list[dict]:
        return [
            {
                "entity": "AuthService",
                "entity_type": "system",
                "domains": ["incident", "engineering"],
                "silo_count": 2,
                "state_version": 5,
                "node_confidence": 0.85,
            }
        ]

    async def test_contradicting_auth_signals_conflict_detected(self) -> None:
        # CON-001 says auth is healthy; CON-002 says auth is down — opposite sentiments
        ticket = get_ticket("CON-001")
        world_nodes = self._auth_world_nodes()
        signals = [
            {
                "content": "Auth service is fully operational; all endpoints stable and healthy.",
                "source_silo": "monitoring",
                "entities": ["AuthService"],
                "relevance_score": 0.9,
                "target_domains": ["incident"],
            },
            {
                "content": "Auth service broken — critical failures, all JWTs rejected and outage ongoing.",
                "source_silo": "engineering",
                "entities": ["AuthService"],
                "relevance_score": 0.9,
                "target_domains": ["incident"],
            },
        ]
        agent = ConflictDetectionAgent()
        ctx = make_conflict_context(ticket, world_nodes=world_nodes, propagated_signals=signals)
        result = await agent.run(ticket["ticket_id"], ctx)

        assert result.status == "success"
        assert result.outputs["conflict_count"] > 0, (
            "Contradicting auth signals should produce at least one conflict"
        )

    async def test_same_silo_signals_less_conflict(self) -> None:
        # Both signals from the same silo — conflict detection needs different silos
        ticket = get_ticket("CON-001")
        world_nodes = self._auth_world_nodes()
        signals = [
            {
                "content": "Auth service is fully operational and stable.",
                "source_silo": "monitoring",
                "entities": ["AuthService"],
                "relevance_score": 0.9,
                "target_domains": ["incident"],
            },
            {
                "content": "Auth service broken — critical outage and failures detected.",
                "source_silo": "monitoring",  # same silo as above
                "entities": ["AuthService"],
                "relevance_score": 0.9,
                "target_domains": ["incident"],
            },
        ]
        agent = ConflictDetectionAgent()
        ctx = make_conflict_context(ticket, world_nodes=world_nodes, propagated_signals=signals)
        result = await agent.run(ticket["ticket_id"], ctx)

        assert result.status == "success"
        # Same silo cannot conflict with itself — state_mismatch requires >= 2 different silos
        state_conflicts = [
            c for c in result.outputs["conflicts"] if c["conflict_type"] == "state_mismatch"
        ]
        assert len(state_conflicts) == 0, (
            "Same-silo contradicting signals should not produce state_mismatch conflicts"
        )

    async def test_deployment_vs_incident_conflict(self) -> None:
        # DEP-001 says auth stable; INC-001 says auth down — cross-silo gap
        ticket = get_ticket("INC-001")
        world_nodes = self._auth_world_nodes()
        signals = [
            {
                "content": "v2.4.0 deployed — auth service updated, confirmed healthy and stable. All endpoints green.",
                "source_silo": "cicd",
                "entities": ["AuthService"],
                "relevance_score": 0.85,
                "target_domains": ["deployment"],
            },
            {
                "content": "Auth service down — JWT validation failing. Error: invalid signature. Incident ongoing.",
                "source_silo": "engineering",
                "entities": ["AuthService"],
                "relevance_score": 0.95,
                "target_domains": ["incident"],
            },
        ]
        agent = ConflictDetectionAgent()
        ctx = make_conflict_context(ticket, world_nodes=world_nodes, propagated_signals=signals)
        result = await agent.run(ticket["ticket_id"], ctx)

        assert result.status == "success"
        assert result.outputs["conflict_count"] > 0, (
            "Deployment saying auth healthy vs incident saying auth down should conflict"
        )

    async def test_stable_signals_no_conflict(self) -> None:
        # Two signals from different silos both reporting the same healthy state
        ticket = get_ticket("CON-001")
        world_nodes = self._auth_world_nodes()
        signals = [
            {
                "content": "Auth service confirmed healthy; all health checks passed and stable.",
                "source_silo": "monitoring",
                "entities": ["AuthService"],
                "relevance_score": 0.88,
                "target_domains": ["incident"],
            },
            {
                "content": "Auth service deployed successfully; endpoints confirmed operational and stable.",
                "source_silo": "cicd",
                "entities": ["AuthService"],
                "relevance_score": 0.80,
                "target_domains": ["deployment"],
            },
        ]
        agent = ConflictDetectionAgent()
        ctx = make_conflict_context(ticket, world_nodes=world_nodes, propagated_signals=signals)
        result = await agent.run(ticket["ticket_id"], ctx)

        assert result.status == "success"
        state_conflicts = [
            c for c in result.outputs["conflicts"] if c["conflict_type"] == "state_mismatch"
        ]
        assert len(state_conflicts) == 0, (
            f"Two aligned positive signals should not produce state_mismatch, "
            f"got {state_conflicts}"
        )
        assert result.outputs.get("has_critical_conflict") is not True or \
               result.outputs["conflict_count"] == 0

    async def test_conflict_count_scales_with_contradictions(self) -> None:
        ticket = get_ticket("INC-001")

        def _make_signals(num_contradictions: int) -> list[dict]:
            sigs = []
            for i in range(num_contradictions):
                sigs.append({
                    "content": f"Service{i} is healthy, stable, and operational. All green.",
                    "source_silo": f"silo-pos-{i}",
                    "entities": [f"Service{i}"],
                    "relevance_score": 0.85,
                    "target_domains": ["incident"],
                })
                sigs.append({
                    "content": f"Service{i} is broken, failed, and experiencing critical outage.",
                    "source_silo": f"silo-neg-{i}",
                    "entities": [f"Service{i}"],
                    "relevance_score": 0.85,
                    "target_domains": ["incident"],
                })
            return sigs

        nodes_few = [
            {"entity": f"Service{i}", "entity_type": "system", "domains": ["incident"],
             "silo_count": 2, "state_version": 1, "node_confidence": 0.8}
            for i in range(2)
        ]
        nodes_many = [
            {"entity": f"Service{i}", "entity_type": "system", "domains": ["incident"],
             "silo_count": 2, "state_version": 1, "node_confidence": 0.8}
            for i in range(5)
        ]

        agent = ConflictDetectionAgent()
        ctx_few = make_conflict_context(
            ticket, world_nodes=nodes_few, propagated_signals=_make_signals(2)
        )
        ctx_many = make_conflict_context(
            ticket, world_nodes=nodes_many, propagated_signals=_make_signals(5)
        )

        r_few = await agent.run("conflict-few", ctx_few)
        r_many = await agent.run("conflict-many", ctx_many)

        assert r_many.outputs["conflict_count"] > r_few.outputs["conflict_count"], (
            f"More contradictions should yield higher conflict_count: "
            f"few={r_few.outputs['conflict_count']}, many={r_many.outputs['conflict_count']}"
        )

    async def test_all_sample_conflicts_valid_structure(self) -> None:
        # Run a few representative tickets through conflict detection and verify structure
        sample_tickets = [
            get_ticket("INC-001"),
            get_ticket("CON-001"),
            get_ticket("DEP-001"),
        ]
        world_nodes: list[dict] = []
        signals: list[dict] = []

        agent = ConflictDetectionAgent()
        for ticket in sample_tickets:
            ctx = make_conflict_context(
                ticket, world_nodes=world_nodes, propagated_signals=signals
            )
            result = await agent.run(ticket["ticket_id"], ctx)

            assert result.status == "success"
            outputs = result.outputs
            assert isinstance(outputs["conflicts"], list)
            assert isinstance(outputs["conflict_count"], int)
            assert outputs["conflict_count"] == len(outputs["conflicts"])
            assert isinstance(outputs["resolution_hints"], list)
            assert "high_severity_conflicts" in outputs
            for c in outputs["conflicts"]:
                assert "entity" in c
                assert "conflict_type" in c
                assert c["severity"] in {"high", "medium", "low"}
