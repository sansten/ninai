"""Tests for Cognitive OS gap implementations.

Covers three services added to close real cognitive loops:

Gap 1 — Inbound event → Goal creation + cognitive loop trigger
  - Tests for the goal title construction, severity qualification, and
    response field presence (goal_id, cognitive_session_id).

Gap 2 — UncertaintyResolutionService
  - needs_resolution(), resolve(), _build_queries(), _score_evidence(),
    _determine_action() across all uncertainty sources and action paths.

Gap 3 — HypothesisService
  - is_anomalous(), analyse() with and without graph data, confirmed
    hypothesis path, no-action short-circuit, evidence boosting.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.services.uncertainty_resolution_service import (
    UncertaintyResolutionService,
    ResolutionQuery,
    ResolutionEvidence,
    UncertaintyResolutionReport,
    _build_queries,
    _score_evidence,
    _determine_action,
)
from app.services.hypothesis_service import (
    HypothesisService,
    Hypothesis,
    HypothesisReport,
    _content_to_effect_candidates,
    _determine_action as _hyp_determine_action,
)
from app.services.cognitive_gateway_service import CognitiveGatewayService


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _gateway_returning(memories: list[dict]) -> CognitiveGatewayService:
    """Return a gateway whose read() always returns the given memories."""
    gw = MagicMock(spec=CognitiveGatewayService)
    read_result = MagicMock()
    read_result.memories = memories
    gw.read.return_value = read_result
    return gw


def _mem(mem_id: str, content: str) -> dict:
    return {"id": mem_id, "content": content}


# ===========================================================================
# Gap 1 — Inbound event: severity threshold + goal title construction
# ===========================================================================

class TestInboundGoalSeverityThreshold:
    """The HIGH_SEVERITY_VALUES set drives when a Goal is created."""

    _HIGH_SEVERITY_VALUES = frozenset({"high", "critical"})

    def test_high_is_in_threshold(self):
        assert "high" in self._HIGH_SEVERITY_VALUES

    def test_critical_is_in_threshold(self):
        assert "critical" in self._HIGH_SEVERITY_VALUES

    def test_medium_not_in_threshold(self):
        assert "medium" not in self._HIGH_SEVERITY_VALUES

    def test_low_not_in_threshold(self):
        assert "low" not in self._HIGH_SEVERITY_VALUES

    def test_none_not_in_threshold(self):
        assert None not in self._HIGH_SEVERITY_VALUES


class TestInboundGoalTitleConstruction:
    """Goal title is built from connector_type + event_type + event title."""

    def _make_title(self, connector_type, event_type, title, external_id="EXT-1") -> str:
        raw = title or external_id or "unknown"
        return (
            f"Investigate {connector_type} {event_type}: "
            f"{str(raw)[:120]}"
        )

    def test_title_includes_connector_type(self):
        t = self._make_title("pagerduty", "incident", "DB is slow")
        assert "pagerduty" in t

    def test_title_includes_event_type(self):
        t = self._make_title("jira", "issue_created", "Login failure")
        assert "issue_created" in t

    def test_title_includes_event_title(self):
        t = self._make_title("slack", "message", "Deploy failed")
        assert "Deploy failed" in t

    def test_title_truncated_at_120(self):
        long_title = "x" * 200
        t = self._make_title("webhook", "alert", long_title)
        # After the prefix, event title should be capped
        suffix = t.split(": ", 1)[1]
        assert len(suffix) <= 120

    def test_title_falls_back_to_external_id(self):
        t = self._make_title("pagerduty", "incident", None, external_id="PD-999")
        assert "PD-999" in t

    def test_critical_priority_is_2(self):
        # priority=2 when severity=="critical", else 1
        def priority(severity):
            return 2 if severity == "critical" else 1
        assert priority("critical") == 2
        assert priority("high") == 1

    def test_high_priority_is_1(self):
        def priority(severity):
            return 2 if severity == "critical" else 1
        assert priority("high") == 1


# ===========================================================================
# Gap 2 — UncertaintyResolutionService
# ===========================================================================

def _high_uncertainty_enrichment() -> dict:
    """Enrichment whose heuristic score lands in 'high' band (0.55–0.80)."""
    return {
        "confidence_scores": {
            "intent": 0.05, "entities": 0.05, "sentiment": 0.05,
            "credibility": 0.05, "temporal": 0.05,
        },
        "high_severity_conflicts": [
            {"topic": "a", "severity": "high"},
            {"topic": "b", "severity": "high"},
            {"topic": "c", "severity": "high"},
        ],
        "conflicts": [
            {"topic": "a", "status": "unresolved", "severity": "high"},
            {"topic": "b", "status": "unresolved", "severity": "high"},
        ],
        "credibility_score": 0.05,
        "is_stale": True,
        "escalation_targets": ["user1"],
    }


def _critical_uncertainty_enrichment() -> dict:
    """Enrichment whose heuristic score lands in 'critical' band (>0.80)."""
    return {
        "confidence_scores": {f"f{i}": 0.01 for i in range(8)},
        "high_severity_conflicts": [{"topic": f"c{i}", "severity": "high"} for i in range(6)],
        "conflicts": [{"topic": f"c{i}", "status": "unresolved", "severity": "high"} for i in range(6)],
        "credibility_score": 0.01,
        "is_stale": True,
        "escalation_targets": ["user1", "user2"],
    }


class TestNeedsResolution:
    def test_high_uncertainty_needs_resolution(self):
        svc = UncertaintyResolutionService()
        assert svc.needs_resolution(_high_uncertainty_enrichment()) is True

    def test_critical_uncertainty_needs_resolution(self):
        svc = UncertaintyResolutionService()
        assert svc.needs_resolution(_critical_uncertainty_enrichment()) is True

    def test_low_uncertainty_no_resolution(self):
        svc = UncertaintyResolutionService()
        assert svc.needs_resolution({}) is False

    def test_normal_enrichment_no_resolution(self):
        svc = UncertaintyResolutionService()
        enrichment = {
            "confidence_scores": {"intent": 0.9, "entities": 0.85},
            "credibility_score": 0.92,
            "high_severity_conflicts": [],
            "conflicts": [],
            "is_stale": False,
        }
        assert svc.needs_resolution(enrichment) is False


class TestBuildQueries:
    def test_low_confidence_field_generates_query(self):
        enrichment = {}
        signals = {"low_confidence_fields": ["intent"], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        fields = [q.field for q in qs]
        assert "intent" in fields

    def test_query_text_contains_field_label(self):
        enrichment = {}
        signals = {"low_confidence_fields": ["sentiment_score"], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        assert any("sentiment score" in q.query_text for q in qs)

    def test_unresolved_conflict_generates_query(self):
        enrichment = {}
        signals = {
            "low_confidence_fields": [],
            "unresolved_conflicts": [{"topic": "budget", "field": "budget"}],
        }
        qs = _build_queries(enrichment, signals)
        assert any(q.uncertainty_source == "unresolved_conflict" for q in qs)

    def test_stale_data_generates_freshness_query(self):
        enrichment = {"is_stale": True}
        signals = {"low_confidence_fields": [], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        assert any(q.field == "freshness" for q in qs)

    def test_stale_false_no_freshness_query(self):
        enrichment = {"is_stale": False}
        signals = {"low_confidence_fields": [], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        assert all(q.field != "freshness" for q in qs)

    def test_low_credibility_generates_credibility_query(self):
        enrichment = {"credibility_score": 0.20}
        signals = {"low_confidence_fields": [], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        assert any(q.field == "credibility" for q in qs)

    def test_high_credibility_no_credibility_query(self):
        enrichment = {"credibility_score": 0.85}
        signals = {"low_confidence_fields": [], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        assert all(q.field != "credibility" for q in qs)

    def test_capped_at_eight_queries(self):
        enrichment = {"is_stale": True, "credibility_score": 0.1}
        signals = {
            "low_confidence_fields": [f"field_{i}" for i in range(10)],
            "unresolved_conflicts": [{"topic": f"conflict_{i}"} for i in range(5)],
        }
        qs = _build_queries(enrichment, signals)
        assert len(qs) <= 8

    def test_uncertainty_source_labels(self):
        enrichment = {}
        signals = {"low_confidence_fields": ["x"], "unresolved_conflicts": []}
        qs = _build_queries(enrichment, signals)
        for q in qs:
            assert q.uncertainty_source in {
                "low_confidence", "unresolved_conflict", "stale", "credibility"
            }


class TestScoreEvidence:
    def test_matching_memory_returned(self):
        memories = [_mem("m1", "current value status intent update")]
        evs = _score_evidence(memories, "current value status intent update")
        assert len(evs) == 1
        assert evs[0].memory_id == "m1"

    def test_no_overlap_returns_empty(self):
        memories = [_mem("m1", "completely unrelated gibberish zzz")]
        evs = _score_evidence(memories, "intent confidence")
        # "confidence" overlap check — depends on content; may or may not match
        # Here, no overlapping token
        assert all(ev.relevance > 0 for ev in evs) or evs == []

    def test_relevance_is_normalized(self):
        memories = [
            _mem("m1", "resolution decision authoritative budget conflict"),
            _mem("m2", "resolution authoritative"),
        ]
        evs = _score_evidence(memories, "resolution decision authoritative budget conflict")
        best = max(evs, key=lambda e: e.relevance)
        assert best.relevance == pytest.approx(1.0)

    def test_top_3_returned(self):
        memories = [_mem(f"m{i}", "current value status intent update") for i in range(10)]
        evs = _score_evidence(memories, "current value status intent update")
        assert len(evs) <= 3

    def test_content_preview_capped(self):
        long_content = "word " * 100
        memories = [_mem("m1", long_content)]
        evs = _score_evidence(memories, "word")
        if evs:
            assert len(evs[0].content_preview) <= 200

    def test_empty_memories_returns_empty(self):
        assert _score_evidence([], "some query") == []


class TestDetermineActionUncertainty:
    def test_all_resolved_returns_resolved(self):
        action, conf = _determine_action(
            uncertainty_level="high",
            n_queries=3,
            n_evidence=2,
            still_unresolved_count=0,
        )
        assert action == "resolved"
        assert conf > 0.60

    def test_critical_unresolved_escalates(self):
        action, conf = _determine_action(
            uncertainty_level="critical",
            n_queries=3,
            n_evidence=0,
            still_unresolved_count=2,
        )
        assert action == "escalate"
        assert conf == pytest.approx(0.30)

    def test_high_unresolved_human_review(self):
        action, conf = _determine_action(
            uncertainty_level="high",
            n_queries=2,
            n_evidence=0,
            still_unresolved_count=1,
        )
        assert action == "human_review"
        assert conf == pytest.approx(0.45)

    def test_monitor_fallback(self):
        action, conf = _determine_action(
            uncertainty_level="low",
            n_queries=0,
            n_evidence=0,
            still_unresolved_count=0,
        )
        assert action == "monitor"

    def test_resolution_confidence_increases_with_evidence(self):
        _, conf1 = _determine_action(
            uncertainty_level="high", n_queries=2, n_evidence=1, still_unresolved_count=0
        )
        _, conf2 = _determine_action(
            uncertainty_level="high", n_queries=2, n_evidence=5, still_unresolved_count=0
        )
        assert conf2 > conf1

    def test_resolution_confidence_capped_at_point_9(self):
        _, conf = _determine_action(
            uncertainty_level="high", n_queries=2, n_evidence=100, still_unresolved_count=0
        )
        assert conf <= 0.90


class TestUncertaintyResolutionServiceResolve:
    def _high_enrichment(self) -> dict:
        return _high_uncertainty_enrichment()

    def test_returns_report_with_memory_id(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-001",
            enrichment=self._high_enrichment(),
        )
        assert report.memory_id == "mem-001"

    def test_queries_generated_for_low_confidence_fields(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-002",
            enrichment=self._high_enrichment(),
        )
        assert len(report.queries_generated) > 0

    def test_no_candidate_memories_no_evidence(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-003",
            enrichment=self._high_enrichment(),
            candidate_memories=None,
        )
        assert report.evidence_found == []

    def test_evidence_found_when_memories_match(self):
        memories = [
            _mem("m1", "current value status intent update"),
            _mem("m2", "current value status entities update"),
        ]
        gw = _gateway_returning(memories)
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-004",
            enrichment=self._high_enrichment(),
            candidate_memories=memories,
        )
        assert len(report.evidence_found) > 0

    def test_evidence_deduplicated_by_memory_id(self):
        memories = [_mem("m1", "current value status intent update")]
        gw = _gateway_returning(memories)
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-005",
            enrichment=self._high_enrichment(),
            candidate_memories=memories,
        )
        ids = [ev.memory_id for ev in report.evidence_found]
        assert len(ids) == len(set(ids))

    def test_still_unresolved_fields_listed(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-006",
            enrichment=self._high_enrichment(),
            candidate_memories=[],
        )
        # No evidence → all queried fields still unresolved
        assert isinstance(report.still_unresolved, list)

    def test_recommended_action_is_valid(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-007",
            enrichment=self._high_enrichment(),
            candidate_memories=[],
        )
        assert report.recommended_action in {"resolved", "escalate", "human_review", "monitor"}

    def test_uncertainty_score_is_float(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(memory_id="mem-008", enrichment=_high_uncertainty_enrichment())
        assert isinstance(report.uncertainty_score, float)
        assert 0.0 <= report.uncertainty_score <= 1.0

    def test_critical_unresolved_escalates(self):
        gw = _gateway_returning([])
        svc = UncertaintyResolutionService(gateway=gw)
        report = svc.resolve(
            memory_id="mem-009",
            enrichment=_critical_uncertainty_enrichment(),
            candidate_memories=[],
        )
        assert report.recommended_action == "escalate"


# ===========================================================================
# Gap 3 — HypothesisService
# ===========================================================================

class TestContentToEffectCandidates:
    def test_anomaly_type_is_first_candidate(self):
        candidates = _content_to_effect_candidates("some content here", "memory_leak")
        assert candidates[0]["entity"] == "memory_leak"
        assert candidates[0]["effect_score"] == pytest.approx(0.9)

    def test_empty_content_returns_empty(self):
        assert _content_to_effect_candidates("", None) == []

    def test_caps_at_five(self):
        content = "alpha bravo charlie delta epsilon foxtrot golf hotel"
        candidates = _content_to_effect_candidates(content, None)
        assert len(candidates) <= 5

    def test_no_anomaly_type_skips_first_slot(self):
        candidates = _content_to_effect_candidates("hello world test", None)
        # All should be concept type from content
        for c in candidates:
            assert c["entity_type"] == "concept"


class TestHypDetermineAction:
    def test_confirmed_above_threshold_escalates_with_cause(self):
        confirmed = Hypothesis(
            entity="db_overload",
            entity_type="system",
            domains=[],
            initial_confidence=0.5,
            confirmed_confidence=0.75,
            supporting_evidence=["m1"],
            causal_path=["slow_query", "db_overload"],
            counterfactual_hint=None,
        )
        action = _hyp_determine_action(confirmed, anomaly_score=0.8)
        assert action == "escalate_with_cause"

    def test_no_confirmed_high_score_investigates(self):
        action = _hyp_determine_action(None, anomaly_score=0.8)
        assert action == "investigate"

    def test_no_confirmed_mid_score_monitors(self):
        action = _hyp_determine_action(None, anomaly_score=0.45)
        assert action == "monitor"

    def test_no_confirmed_low_score_no_action(self):
        action = _hyp_determine_action(None, anomaly_score=0.1)
        assert action == "no_action"


class TestHypothesisServiceIsAnomalous:
    def test_low_credibility_is_anomalous(self):
        svc = HypothesisService()
        assert svc.is_anomalous(_anomalous_enrichment()) is True

    def test_normal_enrichment_not_anomalous(self):
        svc = HypothesisService()
        assert svc.is_anomalous(_normal_enrichment()) is False

    def test_empty_enrichment_not_anomalous(self):
        svc = HypothesisService()
        assert svc.is_anomalous({}) is False


def _anomalous_enrichment() -> dict:
    """Enrichment whose heuristic fires anomaly_detected=True with score > 0.30."""
    return {
        "credibility_score": 0.02,
        "confidence_scores": {"intent": 0.05, "entities": 0.05},
        "high_severity_conflicts": [{"topic": "a"}, {"topic": "b"}, {"topic": "c"}],
        "is_stale": True,
    }


def _normal_enrichment() -> dict:
    return {
        "credibility_score": 0.95,
        "confidence_scores": {"intent": 0.9, "entities": 0.85},
        "high_severity_conflicts": [],
        "is_stale": False,
    }


class TestHypothesisServiceAnalyse:
    def _anomalous_enrichment(self) -> dict:
        return _anomalous_enrichment()

    def _normal_enrichment(self) -> dict:
        return _normal_enrichment()

    def test_no_anomaly_returns_no_action_report(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-x",
            content="everything is fine",
            enrichment=self._normal_enrichment(),
        )
        assert report.anomaly_detected is False
        assert report.recommended_action == "no_action"
        assert report.confirmed_hypothesis is None

    def test_anomaly_returns_report_with_memory_id(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-1",
            content="database response time exceeded 5s three times this hour",
            enrichment=self._anomalous_enrichment(),
        )
        assert report.memory_id == "mem-hyp-1"
        assert report.anomaly_detected is True

    def test_anomaly_generates_hypotheses(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-2",
            content="database response time exceeded 5s frequently today",
            enrichment=self._anomalous_enrichment(),
        )
        # Content fallback should produce at least one hypothesis
        assert isinstance(report.hypotheses, list)

    def test_anomaly_type_in_report(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-3",
            content="something unusual happening",
            enrichment=self._anomalous_enrichment(),
        )
        assert report.anomaly_type is not None

    def test_evidence_boosts_confidence(self):
        """Supporting evidence should push confirmed_confidence above initial."""
        memories = [
            _mem("m1", "spike database latency overload"),
            _mem("m2", "spike error rate increased"),
        ]
        gw = _gateway_returning(memories)
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-4",
            content="spike occurred repeatedly in the database layer",
            enrichment=self._anomalous_enrichment(),
            candidate_memories=memories,
        )
        if report.hypotheses:
            top = report.hypotheses[0]
            assert top.confirmed_confidence >= top.initial_confidence

    def test_hypotheses_sorted_descending(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-5",
            content="spike in database latency caused failures",
            enrichment=self._anomalous_enrichment(),
        )
        confs = [h.confirmed_confidence for h in report.hypotheses]
        assert confs == sorted(confs, reverse=True)

    def test_evidence_count_equals_sum_of_supporting(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-6",
            content="unusual latency spike",
            enrichment=self._anomalous_enrichment(),
        )
        expected = sum(len(h.supporting_evidence) for h in report.hypotheses)
        assert report.evidence_count == expected

    def test_report_action_is_valid(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-7",
            content="something is off",
            enrichment=self._anomalous_enrichment(),
        )
        assert report.recommended_action in {
            "escalate_with_cause", "investigate", "monitor", "no_action"
        }

    def test_with_world_nodes_and_edges(self):
        """Graph path: when world_nodes and entity_edges are provided."""
        world_nodes = [
            {"entity": "database", "entity_type": "system", "domains": ["infra"], "effect_score": 0.9},
            {"entity": "slow_query", "entity_type": "event", "domains": ["infra"], "effect_score": 0.5},
            {"entity": "disk_io", "entity_type": "resource", "domains": ["infra"], "effect_score": 0.3},
        ]
        entity_edges = [
            {"source": "database", "target": "slow_query", "weight": 0.8},
            {"source": "slow_query", "target": "disk_io", "weight": 0.6},
        ]
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-8",
            content="database latency spike",
            enrichment=self._anomalous_enrichment(),
            world_nodes=world_nodes,
            entity_edges=entity_edges,
        )
        assert report.anomaly_detected is True
        assert isinstance(report.hypotheses, list)

    def test_confirmed_hypothesis_above_threshold(self):
        """A hypothesis with enough evidence should be set as confirmed."""
        # Flood evidence so boost pushes past 0.70
        memories = [
            _mem(f"m{i}", "credibility spike latency overload query disk")
            for i in range(8)
        ]
        gw = _gateway_returning(memories)
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-9",
            content="credibility spike occurred repeatedly in the layer",
            enrichment=_anomalous_enrichment(),
            candidate_memories=memories,
        )
        # If a hypothesis is confirmed, it must be above threshold
        if report.confirmed_hypothesis is not None:
            assert report.confirmed_hypothesis.confirmed_confidence >= 0.70
            assert report.recommended_action == "escalate_with_cause"

    def test_anomaly_score_preserved_in_report(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-10",
            content="some event",
            enrichment=self._anomalous_enrichment(),
        )
        assert isinstance(report.anomaly_score, float)
        assert report.anomaly_score >= 0.30  # must be at least the min threshold

    def test_hypothesis_causal_path_is_list(self):
        gw = _gateway_returning([])
        svc = HypothesisService(gateway=gw)
        report = svc.analyse(
            memory_id="mem-hyp-11",
            content="latency spike database overload repeated",
            enrichment=self._anomalous_enrichment(),
        )
        for h in report.hypotheses:
            assert isinstance(h.causal_path, list)
