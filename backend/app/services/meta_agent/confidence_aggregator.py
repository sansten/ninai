from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
    "w_agent_confidence": 0.35,
    "w_evidence_strength": 0.25,
    "w_historical_accuracy": 0.20,
    "w_consistency_score": 0.15,
    "w_contradiction_penalty": 0.05,
}


def normalize_signal_weights(weights: dict[str, float] | None) -> dict[str, float]:
    weights = {**DEFAULT_SIGNAL_WEIGHTS, **(weights or {})}

    keys = [
        "w_agent_confidence",
        "w_evidence_strength",
        "w_historical_accuracy",
        "w_consistency_score",
        "w_contradiction_penalty",
    ]

    # Fail-closed on non-numeric / negative weights
    cleaned: dict[str, float] = {}
    for key in keys:
        value = weights.get(key)
        if value is None:
            value = 0.0
        value = float(value)
        if value < 0:
            raise ValueError("Signal weights must be non-negative")
        cleaned[key] = value

    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_SIGNAL_WEIGHTS)

    return {k: v / total for k, v in cleaned.items()}


@dataclass(frozen=True)
class AggregationInputs:
    agent_confidence: float
    evidence_strength: float
    historical_accuracy: float
    consistency_score: float
    contradiction_penalty: float


@dataclass(frozen=True)
class AggregationResult:
    overall_confidence: float
    risk_score: float


class ConfidenceAggregator:
    def __init__(self, *, signal_weights: dict[str, float] | None = None):
        self.signal_weights = normalize_signal_weights(signal_weights)

    def aggregate(self, inputs: AggregationInputs) -> AggregationResult:
        w = self.signal_weights

        overall = (
            w["w_agent_confidence"] * _clamp01(inputs.agent_confidence)
            + w["w_evidence_strength"] * _clamp01(inputs.evidence_strength)
            + w["w_historical_accuracy"] * _clamp01(inputs.historical_accuracy)
            + w["w_consistency_score"] * _clamp01(inputs.consistency_score)
            - w["w_contradiction_penalty"] * _clamp01(inputs.contradiction_penalty)
        )
        overall = _clamp01(overall)

        # Simple risk model: contradiction dominates, then low confidence.
        risk = _clamp01(0.7 * _clamp01(inputs.contradiction_penalty) + 0.3 * (1.0 - overall))

        return AggregationResult(overall_confidence=overall, risk_score=risk)
