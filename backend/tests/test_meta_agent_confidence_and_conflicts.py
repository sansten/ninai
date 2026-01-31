from __future__ import annotations

import pytest

from app.services.meta_agent.confidence_aggregator import (
    AggregationInputs,
    ConfidenceAggregator,
    normalize_signal_weights,
)
from app.services.meta_agent.conflict_resolver import (
    ClassificationCandidate,
    detect_classification_conflict,
    resolve_classification,
    resolve_classification_candidates,
)


def test_normalize_signal_weights_sums_to_one():
    w = normalize_signal_weights({"w_agent_confidence": 2, "w_evidence_strength": 2})
    assert pytest.approx(sum(w.values()), rel=1e-6) == 1.0


def test_normalize_signal_weights_rejects_negative():
    with pytest.raises(ValueError):
        normalize_signal_weights({"w_agent_confidence": -0.1})


def test_confidence_aggregation_basic_behavior():
    agg = ConfidenceAggregator(
        signal_weights={
            "w_agent_confidence": 1.0,
            "w_evidence_strength": 0.0,
            "w_historical_accuracy": 0.0,
            "w_consistency_score": 0.0,
            "w_contradiction_penalty": 0.0,
        }
    )
    res = agg.aggregate(
        AggregationInputs(
            agent_confidence=0.9,
            evidence_strength=0.0,
            historical_accuracy=0.0,
            consistency_score=0.0,
            contradiction_penalty=0.0,
        )
    )
    assert pytest.approx(res.overall_confidence, rel=1e-6) == 0.9
    assert 0.0 <= res.risk_score <= 1.0


def test_resolve_classification_most_restrictive():
    assert resolve_classification(["public", "confidential"]) == "confidential"
    assert resolve_classification(["internal", "restricted", "confidential"]) == "restricted"


def test_resolve_classification_candidates_confidence_gap_overrides():
    # Default would be most restrictive (restricted), but public wins with huge confidence gap.
    resolved = resolve_classification_candidates(
        [
            ClassificationCandidate("restricted", 0.1),
            ClassificationCandidate("public", 0.95),
        ],
        confidence_gap_threshold=0.60,
    )
    assert resolved == "public"


def test_detect_classification_conflict():
    c = detect_classification_conflict(["internal", "internal"])
    assert c.has_conflict is False

    c2 = detect_classification_conflict(["internal", "confidential"])
    assert c2.has_conflict is True
    assert c2.conflict_type == "classification"
    assert "candidates" in (c2.details or {})
