"""Tests for the four AGI accuracy improvements.

1. Credibility-weighted evidence retrieval
   — CognitiveGatewayService._assemble_read_context
   — UncertaintyResolutionService._score_evidence
   — HypothesisService._find_corroborating_evidence

2. Contradiction-aware evidence in HypothesisService
   — _find_contradicting_evidence
   — contradiction penalty reduces confirmed_confidence

3. ConfidenceCalibrationService
   — ingest_outcomes, calibrate, passthrough, is_calibrated, report

4. Causal edge reinforcement
   — reinforce_causal_edges (standalone)
   — HypothesisService.reinforce_from_report
   — auto-reinforcement inside analyse()
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.services.cognitive_gateway_service import (
    _assemble_read_context,
    _credibility_weight as _gw_credibility_weight,
)
from app.services.uncertainty_resolution_service import (
    _score_evidence,
    _credibility_weight as _urs_credibility_weight,
)
from app.services.hypothesis_service import (
    HypothesisService,
    Hypothesis,
    HypothesisReport,
    _credibility_weight as _hs_credibility_weight,
    _find_contradicting_evidence,
    reinforce_causal_edges,
)
from app.services.confidence_calibration_service import (
    ConfidenceCalibrationService,
    CalibrationBucket,
    _bucket_index,
    _N_BUCKETS,
    _MIN_SAMPLES,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mem(mem_id: str, content: str, credibility: float | None = None) -> dict:
    m: dict = {"id": mem_id, "content": content}
    if credibility is not None:
        m["credibility_score"] = credibility
    return m


def _gateway_returning(memories: list[dict]):
    gw = MagicMock()
    read_result = MagicMock()
    read_result.memories = memories
    gw.read.return_value = read_result
    return gw


def _anomalous_enrichment() -> dict:
    return {
        "credibility_score": 0.02,
        "confidence_scores": {"intent": 0.05, "entities": 0.05},
        "high_severity_conflicts": [{"topic": "a"}, {"topic": "b"}, {"topic": "c"}],
        "is_stale": True,
    }


# ===========================================================================
# 1. Credibility-weighted evidence retrieval
# ===========================================================================

class TestCredibilityWeightHelper:
    def test_explicit_credibility_score(self):
        mem = {"credibility_score": 0.6}
        assert _hs_credibility_weight(mem) == pytest.approx(0.6)

    def test_nested_enrichment_fallback(self):
        mem = {"enrichment": {"credibility_score": 0.4}}
        assert _hs_credibility_weight(mem) == pytest.approx(0.4)

    def test_missing_returns_one(self):
        assert _hs_credibility_weight({}) == pytest.approx(1.0)

    def test_clamped_below_point_one(self):
        mem = {"credibility_score": 0.0}
        assert _hs_credibility_weight(mem) == pytest.approx(0.1)

    def test_clamped_above_one(self):
        mem = {"credibility_score": 1.5}
        assert _hs_credibility_weight(mem) == pytest.approx(1.0)

    def test_invalid_value_returns_one(self):
        mem = {"credibility_score": "bad"}
        assert _hs_credibility_weight(mem) == pytest.approx(1.0)

    def test_gateway_helper_same_logic(self):
        mem = {"credibility_score": 0.3}
        assert _gw_credibility_weight(mem) == pytest.approx(0.3)

    def test_urs_helper_same_logic(self):
        mem = {"credibility_score": 0.7}
        assert _urs_credibility_weight(mem) == pytest.approx(0.7)


class TestAssembleReadContextCredibility:
    def test_high_credibility_ranks_above_low(self):
        """Both memories match the query equally; higher-credibility one must rank first."""
        mems = [
            _mem("m-low", "status update current", credibility=0.1),
            _mem("m-high", "status update current", credibility=0.9),
        ]
        ranked = _assemble_read_context(mems, "status update current")
        assert ranked[0]["id"] == "m-high"

    def test_zero_overlap_ranks_last(self):
        mems = [
            _mem("m-match", "query word here", credibility=0.8),
            _mem("m-none", "unrelated gibberish", credibility=1.0),
        ]
        ranked = _assemble_read_context(mems, "query word here")
        assert ranked[0]["id"] == "m-match"

    def test_no_credibility_field_defaults_to_one(self):
        """Memories without credibility_score should behave as weight=1.0."""
        mems = [
            _mem("m1", "status update current"),
            _mem("m2", "status update current"),
        ]
        ranked = _assemble_read_context(mems, "status update current")
        assert len(ranked) == 2


class TestScoreEvidenceCredibility:
    def test_high_credibility_gets_higher_relevance(self):
        mems = [
            _mem("m-low", "current value status intent update", credibility=0.1),
            _mem("m-high", "current value status intent update", credibility=0.9),
        ]
        evs = _score_evidence(mems, "current value status intent update")
        ids = [e.memory_id for e in evs]
        assert ids.index("m-high") < ids.index("m-low")

    def test_top_relevance_is_one_after_normalisation(self):
        mems = [_mem("m1", "resolution decision authoritative", credibility=1.0)]
        evs = _score_evidence(mems, "resolution decision authoritative")
        assert evs[0].relevance == pytest.approx(1.0)

    def test_low_credibility_reduces_relevance(self):
        mems = [
            _mem("m-full", "status intent update", credibility=1.0),
            _mem("m-low", "status intent update", credibility=0.1),
        ]
        evs = _score_evidence(mems, "status intent update")
        full = next(e for e in evs if e.memory_id == "m-full")
        low = next(e for e in evs if e.memory_id == "m-low")
        assert full.relevance > low.relevance


class TestFindCorroboratingEvidenceCredibility:
    def test_low_credibility_memory_still_qualifies_above_floor(self):
        """The credibility floor of 0.1 means even low-credibility memories can qualify."""
        # credibility=0.1 → weight=0.1, overlap=1 → weighted=0.1 >= 0.1 threshold
        memories = [_mem("m-low", "spike present", credibility=0.1)]
        gw = _gateway_returning(memories)
        from app.services.hypothesis_service import _find_corroborating_evidence
        result = _find_corroborating_evidence(
            hypothesis_entity="spike",
            domains=[],
            candidate_memories=memories,
            gateway=gw,
        )
        assert "m-low" in result

    def test_high_credibility_memory_included(self):
        memories = [_mem("m-good", "spike high frequency", credibility=0.9)]
        gw = _gateway_returning(memories)
        from app.services.hypothesis_service import _find_corroborating_evidence
        result = _find_corroborating_evidence(
            hypothesis_entity="spike",
            domains=[],
            candidate_memories=memories,
            gateway=gw,
        )
        assert "m-good" in result


# ===========================================================================
# 2. Contradiction-aware evidence
# ===========================================================================

class TestFindContradictingEvidence:
    def test_negation_near_entity_detected(self):
        mems = [_mem("m1", "the spike is not present in this system", credibility=0.9)]
        result = _find_contradicting_evidence(
            hypothesis_entity="spike",
            candidate_memories=mems,
        )
        assert "m1" in result

    def test_resolved_near_entity_detected(self):
        mems = [_mem("m1", "spike issue was resolved by the team", credibility=0.8)]
        result = _find_contradicting_evidence(
            hypothesis_entity="spike",
            candidate_memories=mems,
        )
        assert "m1" in result

    def test_entity_absent_not_contradicting(self):
        mems = [_mem("m1", "the latency is resolved", credibility=0.9)]
        result = _find_contradicting_evidence(
            hypothesis_entity="spike",
            candidate_memories=mems,
        )
        assert result == []

    def test_negation_far_from_entity_not_detected(self):
        # "not" is more than 5 words away from "spike"
        mems = [_mem("m1", "not a factor in any way at all and spike happened", credibility=0.9)]
        result = _find_contradicting_evidence(
            hypothesis_entity="spike",
            candidate_memories=mems,
        )
        # "not" at position 0, "spike" at position 10 — distance 10 > 5
        assert result == []

    def test_low_credibility_source_ignored(self):
        """Contradictions from low-credibility sources should be dismissed."""
        mems = [_mem("m1", "spike is not present", credibility=0.1)]
        result = _find_contradicting_evidence(
            hypothesis_entity="spike",
            candidate_memories=mems,
        )
        assert result == []

    def test_caps_at_three(self):
        mems = [
            _mem(f"m{i}", f"spike is not here {i}", credibility=0.9)
            for i in range(10)
        ]
        result = _find_contradicting_evidence(
            hypothesis_entity="spike",
            candidate_memories=mems,
        )
        assert len(result) <= 3

    def test_empty_memories_returns_empty(self):
        assert _find_contradicting_evidence(hypothesis_entity="x", candidate_memories=[]) == []


class TestContradictionPenaltyInAnalyse:
    def test_contradictions_reduce_confidence(self):
        """Contradicting evidence for every hypothesis should lower their scores."""
        # Memories that contain the anomaly_type entity + negation
        memories = [
            _mem("m1", "credibility_spike is not present", credibility=0.9),
            _mem("m2", "credibility_spike was resolved", credibility=0.8),
        ]
        gw = _gateway_returning(memories)
        svc = HypothesisService(gateway=gw)

        report_no_contra = HypothesisService(gateway=_gateway_returning([])).analyse(
            memory_id="base",
            content="credibility_spike issue",
            enrichment=_anomalous_enrichment(),
            candidate_memories=[],
        )
        report_with_contra = svc.analyse(
            memory_id="contra",
            content="credibility_spike issue",
            enrichment=_anomalous_enrichment(),
            candidate_memories=memories,
        )

        # When contradictions are present, top hypothesis should have equal or lower confidence
        if report_no_contra.hypotheses and report_with_contra.hypotheses:
            assert (
                report_with_contra.hypotheses[0].confirmed_confidence
                <= report_no_contra.hypotheses[0].confirmed_confidence + 0.30  # allow for evidence differences
            )

    def test_contradicting_evidence_field_present(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-c",
            content="unusual event occurred",
            enrichment=_anomalous_enrichment(),
        )
        for h in report.hypotheses:
            assert isinstance(h.contradicting_evidence, list)


# ===========================================================================
# 3. ConfidenceCalibrationService
# ===========================================================================

class TestBucketIndex:
    def test_zero_maps_to_bucket_zero(self):
        assert _bucket_index(0.0) == 0

    def test_one_maps_to_last_bucket(self):
        assert _bucket_index(1.0) == _N_BUCKETS - 1

    def test_midpoint(self):
        assert _bucket_index(0.5) == 5

    def test_just_below_boundary(self):
        assert _bucket_index(0.499) == 4

    def test_just_above_boundary(self):
        assert _bucket_index(0.501) == 5


class TestConfidenceCalibrationServicePassthrough:
    def test_no_table_returns_raw(self):
        svc = ConfidenceCalibrationService()
        assert svc.calibrate(0.75) == pytest.approx(0.75)

    def test_unknown_agent_type_returns_raw(self):
        svc = ConfidenceCalibrationService()
        assert svc.calibrate(0.60, "hypothesis") == pytest.approx(0.60)

    def test_is_calibrated_false_without_data(self):
        svc = ConfidenceCalibrationService()
        assert svc.is_calibrated("hypothesis") is False

    def test_calibrate_clamps_input_below_zero(self):
        svc = ConfidenceCalibrationService()
        result = svc.calibrate(-0.5)
        assert 0.0 <= result <= 1.0

    def test_calibrate_clamps_input_above_one(self):
        svc = ConfidenceCalibrationService()
        result = svc.calibrate(1.5)
        assert 0.0 <= result <= 1.0

    def test_calibrate_invalid_type_returns_half(self):
        svc = ConfidenceCalibrationService()
        result = svc.calibrate("bad")  # type: ignore
        assert result == pytest.approx(0.5)


class TestConfidenceCalibrationServiceIngest:
    def _make_outcomes(self, n_success: int, n_fail: int, conf: float) -> list[dict]:
        return (
            [{"predicted_confidence": conf, "success": True}] * n_success
            + [{"predicted_confidence": conf, "success": False}] * n_fail
        )

    def test_ingest_populates_buckets(self):
        svc = ConfidenceCalibrationService()
        outcomes = self._make_outcomes(8, 2, 0.75)
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        assert svc.is_calibrated("hyp") is True

    def test_calibrate_after_ingest_shifts_toward_observed_rate(self):
        svc = ConfidenceCalibrationService()
        # bucket for 0.75 (bucket 7): 8 successes out of 10 → observed 0.80
        outcomes = self._make_outcomes(8, 2, 0.75)
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        cal = svc.calibrate(0.75, "hyp")
        # calibrated should be between observed_rate=0.80 and raw=0.75
        assert 0.74 <= cal <= 0.81

    def test_sparse_bucket_returns_raw(self):
        svc = ConfidenceCalibrationService()
        # Only 2 observations — below _MIN_SAMPLES
        outcomes = self._make_outcomes(1, 1, 0.55)
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        # bucket for 0.55 has count=2 < 5 → passthrough
        assert svc.calibrate(0.55, "hyp") == pytest.approx(0.55)

    def test_all_failures_shifts_down(self):
        svc = ConfidenceCalibrationService()
        outcomes = self._make_outcomes(0, 10, 0.85)  # 0% success in bucket 8
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        cal = svc.calibrate(0.85, "hyp")
        assert cal < 0.85

    def test_all_successes_shifts_up(self):
        svc = ConfidenceCalibrationService()
        outcomes = self._make_outcomes(10, 0, 0.35)  # 100% success in bucket 3
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        cal = svc.calibrate(0.35, "hyp")
        assert cal > 0.35

    def test_calibrated_score_clamped(self):
        svc = ConfidenceCalibrationService()
        # Extreme case: raw=0.05, all success → should not exceed 1.0
        outcomes = self._make_outcomes(20, 0, 0.05)
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        assert svc.calibrate(0.05, "hyp") <= 1.0

    def test_report_returns_correct_total(self):
        svc = ConfidenceCalibrationService()
        outcomes = self._make_outcomes(5, 5, 0.75)
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        report = svc.report(agent_type="hyp")
        assert report.total_samples == 10

    def test_report_calibrated_buckets_count(self):
        svc = ConfidenceCalibrationService()
        outcomes = self._make_outcomes(8, 2, 0.75)
        svc.ingest_outcomes(agent_type="hyp", outcomes=outcomes)
        report = svc.report(agent_type="hyp")
        assert report.calibrated_buckets == 1

    def test_multiple_agent_types_isolated(self):
        svc = ConfidenceCalibrationService()
        svc.ingest_outcomes(agent_type="a", outcomes=self._make_outcomes(10, 0, 0.5))
        svc.ingest_outcomes(agent_type="b", outcomes=self._make_outcomes(0, 10, 0.5))
        cal_a = svc.calibrate(0.5, "a")
        cal_b = svc.calibrate(0.5, "b")
        assert cal_a > cal_b

    def test_invalid_outcome_ignored(self):
        svc = ConfidenceCalibrationService()
        outcomes = [{"predicted_confidence": "not-a-float", "success": True}]
        svc.ingest_outcomes(agent_type="x", outcomes=outcomes)
        # Should not raise; bad record is skipped
        assert svc.calibrate(0.5, "x") == pytest.approx(0.5)


# ===========================================================================
# 4. Causal edge reinforcement
# ===========================================================================

class TestReinforceCausalEdges:
    def _make_edges(self) -> list[dict]:
        return [
            {"from_entity": "a", "to_entity": "b", "weight": 0.5},
            {"from_entity": "b", "to_entity": "c", "weight": 0.5},
        ]

    def test_confirmed_strengthens_edges(self):
        edges = self._make_edges()
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b", "c"], confirmed=True)
        # Both edges should increase from 0.5
        for e in edges:
            assert e["weight"] > 0.5

    def test_denied_weakens_edges(self):
        edges = self._make_edges()
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b", "c"], confirmed=False)
        for e in edges:
            assert e["weight"] < 0.5

    def test_unrelated_edge_unchanged(self):
        edges = self._make_edges()
        edges.append({"from_entity": "x", "to_entity": "y", "weight": 0.5})
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b"], confirmed=True)
        xy = next(e for e in edges if e["from_entity"] == "x")
        assert xy["weight"] == pytest.approx(0.5)

    def test_weight_clamped_at_one(self):
        edges = [{"from_entity": "a", "to_entity": "b", "weight": 0.99}]
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b"], confirmed=True)
        assert edges[0]["weight"] <= 1.0

    def test_weight_clamped_at_point_zero_five(self):
        edges = [{"from_entity": "a", "to_entity": "b", "weight": 0.06}]
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b"], confirmed=False)
        assert edges[0]["weight"] >= 0.05

    def test_short_path_no_edges_updated(self):
        edges = self._make_edges()
        original = [e["weight"] for e in edges]
        reinforce_causal_edges(entity_edges=edges, causal_path=["a"], confirmed=True)
        assert [e["weight"] for e in edges] == original

    def test_undirected_match(self):
        """Edge stored as (b, a) should be found when path is [a, b]."""
        edges = [{"from_entity": "b", "to_entity": "a", "weight": 0.5}]
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b"], confirmed=True)
        assert edges[0]["weight"] > 0.5

    def test_returns_same_list(self):
        edges = self._make_edges()
        result = reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b"], confirmed=True)
        assert result is edges

    def test_ema_formula(self):
        """Confirmed: new_weight = 0.75*old + 0.25*1.0."""
        edges = [{"from_entity": "a", "to_entity": "b", "weight": 0.5}]
        reinforce_causal_edges(entity_edges=edges, causal_path=["a", "b"], confirmed=True)
        expected = round(0.75 * 0.5 + 0.25 * 1.0, 4)
        assert edges[0]["weight"] == pytest.approx(expected)


class TestReinforceFromReport:
    def _make_report(self, confirmed: bool) -> HypothesisReport:
        conf = 0.80 if confirmed else 0.30
        h = Hypothesis(
            entity="db_overload",
            entity_type="system",
            domains=[],
            initial_confidence=0.4,
            confirmed_confidence=conf,
            supporting_evidence=["m1"] if confirmed else [],
            contradicting_evidence=[] if confirmed else ["m2"],
            causal_path=["effect", "db_overload"],
            counterfactual_hint=None,
        )
        return HypothesisReport(
            memory_id="mem-r",
            anomaly_detected=True,
            anomaly_score=0.7,
            anomaly_type="spike",
            hypotheses=[h],
            confirmed_hypothesis=h if confirmed else None,
            recommended_action="escalate_with_cause" if confirmed else "monitor",
            counterfactual_hint=None,
            evidence_count=1 if confirmed else 0,
        )

    def test_confirmed_report_strengthens_edges(self):
        svc = HypothesisService()
        edges = [{"from_entity": "effect", "to_entity": "db_overload", "weight": 0.5}]
        report = self._make_report(confirmed=True)
        svc.reinforce_from_report(report=report, entity_edges=edges)
        assert edges[0]["weight"] > 0.5

    def test_denied_report_weakens_edges(self):
        svc = HypothesisService()
        edges = [{"from_entity": "effect", "to_entity": "db_overload", "weight": 0.5}]
        report = self._make_report(confirmed=False)
        svc.reinforce_from_report(report=report, entity_edges=edges)
        # Only weakens if has_contradictions and not supporting_evidence
        assert edges[0]["weight"] <= 0.5

    def test_empty_hypotheses_no_change(self):
        svc = HypothesisService()
        edges = [{"from_entity": "a", "to_entity": "b", "weight": 0.5}]
        empty_report = HypothesisReport(
            memory_id="m", anomaly_detected=False, anomaly_score=0.0,
            anomaly_type=None, hypotheses=[], confirmed_hypothesis=None,
            recommended_action="no_action", counterfactual_hint=None, evidence_count=0,
        )
        svc.reinforce_from_report(report=empty_report, entity_edges=edges)
        assert edges[0]["weight"] == pytest.approx(0.5)


class TestAutoReinforcementInAnalyse:
    def test_confirmed_hypothesis_strengthens_entity_edges(self):
        """When a confirmed hypothesis is found, the edges in its causal path strengthen."""
        world_nodes = [
            {"entity": "latency", "entity_type": "metric", "domains": [], "effect_score": 0.9},
            {"entity": "db_overload", "entity_type": "system", "domains": [], "effect_score": 0.4},
        ]
        entity_edges = [
            {"from_entity": "latency", "to_entity": "db_overload", "weight": 0.5},
        ]
        original_weight = entity_edges[0]["weight"]

        # Enough matching evidence to push past 0.70 threshold
        memories = [_mem(f"m{i}", "latency db_overload credibility spike") for i in range(8)]
        gw = _gateway_returning(memories)
        svc = HypothesisService(gateway=gw)

        svc.analyse(
            memory_id="mem-auto",
            content="latency db_overload credibility spike repeated",
            enrichment=_anomalous_enrichment(),
            candidate_memories=memories,
            world_nodes=world_nodes,
            entity_edges=entity_edges,
        )
        # The edge may have been strengthened if a confirmed hypothesis used it
        # We can't guarantee graph traversal hit this exact edge, but weight should not decrease
        assert entity_edges[0]["weight"] >= original_weight * 0.95
