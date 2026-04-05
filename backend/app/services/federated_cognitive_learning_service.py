"""Federated Cognitive Learning Service (Feature 24.8).

Extends the FederatedMemoryAgent foundations with:
  1. Gradient-free federated aggregation of heuristic agent weights
     (FedAvg-style weighted averaging without raw data exchange)
  2. Per-org differential privacy noise calibration
     (epsilon budget scaled by org's sensitivity tier)
  3. Multi-metric cross-org benchmark insights
     (no raw data leaves any org — only DP-noised aggregates)

References:
  McMahan et al. 2017 "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" — adapted here for heuristic weight vectors
  instead of gradient updates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from app.services.differential_privacy_service import DifferentialPrivacyService


# ---------------------------------------------------------------------------
# Sensitivity tiers — map org data-sensitivity label → epsilon
# ---------------------------------------------------------------------------

_SENSITIVITY_EPSILON: dict[str, float] = {
    "high":   0.1,   # strict privacy — lots of noise
    "medium": 1.0,   # balanced
    "low":    5.0,   # relaxed — less noise, more signal
}
_DEFAULT_EPSILON = 1.0


def _epsilon_for_org(sensitivity_tier: str) -> float:
    return _SENSITIVITY_EPSILON.get(str(sensitivity_tier).lower(), _DEFAULT_EPSILON)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AggregatedWeights:
    """Federated aggregate of heuristic agent weight vectors."""
    agent_name: str
    global_weights: dict[str, float]        # DP-noised federated average
    contributing_orgs: int
    aggregation_epsilon: float              # effective epsilon used
    privacy_budget_spent: float


@dataclass
class OrgBenchmarkInsight:
    metric: str
    org_value: float
    global_private_mean: float              # DP mean across orgs
    percentile_rank: float                  # 0..1
    gap_to_p75: float                       # positive = below target
    status: str                             # "on_track" | "below_target"


@dataclass
class FederatedLearningResult:
    aggregated_weights: list[AggregatedWeights]
    benchmark_insights: list[OrgBenchmarkInsight]
    sharing_recommendation: str
    total_privacy_budget_spent: float
    federation_confidence: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _percentile_rank(value: float, peers: list[float]) -> float:
    if not peers:
        return 0.0
    return sum(1 for v in peers if v <= value) / len(peers)


def _fedavg(
    org_weights: list[dict[str, float]],
    org_sample_counts: list[int],
) -> dict[str, float]:
    """FedAvg: weighted average of weight dicts by org sample count.

    Gradient-free variant: direct weight averaging with sample-size weighting.
    """
    total_samples = sum(org_sample_counts)
    if total_samples == 0:
        total_samples = max(1, len(org_weights))
        org_sample_counts = [1] * len(org_weights)

    all_keys: set[str] = set()
    for w in org_weights:
        all_keys.update(w.keys())

    averaged: dict[str, float] = {}
    for key in all_keys:
        weighted_sum = 0.0
        for w, n in zip(org_weights, org_sample_counts):
            weighted_sum += float(w.get(key, 0.0)) * n
        averaged[key] = weighted_sum / total_samples

    return averaged


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FederatedCognitiveLearningService:
    """Feature 24.8: Federated weight aggregation + multi-metric benchmarking.

    All raw org data stays local; only DP-noised aggregates travel.
    """

    def aggregate_agent_weights(
        self,
        *,
        agent_name: str,
        org_weight_submissions: list[dict[str, Any]],
    ) -> AggregatedWeights:
        """Aggregate per-org heuristic agent weight vectors (FedAvg-style).

        Each submission must have:
          weights: dict[str, float]  — agent weight fields
          sample_count: int          — size of org's local dataset
          sensitivity_tier: str      — "high" | "medium" | "low"

        Returns a DP-averaged global weight vector and budget bookkeeping.
        """
        if not org_weight_submissions:
            return AggregatedWeights(
                agent_name=agent_name,
                global_weights={},
                contributing_orgs=0,
                aggregation_epsilon=_DEFAULT_EPSILON,
                privacy_budget_spent=0.0,
            )

        raw_weights: list[dict[str, float]] = []
        sample_counts: list[int] = []
        epsilons: list[float] = []

        for sub in org_weight_submissions:
            weights = {k: float(v) for k, v in (sub.get("weights") or {}).items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            n = max(1, int(sub.get("sample_count") or 1))
            eps = _epsilon_for_org(sub.get("sensitivity_tier") or "medium")
            raw_weights.append(weights)
            sample_counts.append(n)
            epsilons.append(eps)

        # Use the strictest (smallest) epsilon across participating orgs
        effective_epsilon = min(epsilons)
        dp = DifferentialPrivacyService(epsilon=effective_epsilon)

        # FedAvg
        averaged = _fedavg(raw_weights, sample_counts)

        # Add DP noise (Laplace) to each weight dimension
        keys = list(averaged.keys())
        noisy_values = dp.add_laplace_noise(
            [averaged[k] for k in keys],
            sensitivity=1.0,
            epsilon=effective_epsilon,
        )
        noisy_weights = dict(zip(keys, noisy_values))

        # Privacy budget = one aggregation query
        budget_spent = effective_epsilon  # compositional: 1 query * epsilon

        return AggregatedWeights(
            agent_name=agent_name,
            global_weights={k: round(v, 6) for k, v in noisy_weights.items()},
            contributing_orgs=len(org_weight_submissions),
            aggregation_epsilon=effective_epsilon,
            privacy_budget_spent=budget_spent,
        )

    def benchmark_org(
        self,
        *,
        org_metrics: dict[str, float],
        peer_metric_submissions: list[dict[str, Any]],
        sensitivity_tier: str = "medium",
    ) -> list[OrgBenchmarkInsight]:
        """Produce DP-safe multi-metric benchmark insights for one org.

        Each peer submission: dict of {metric_name: float_value}.
        Raw peer values are only used to compute a noised aggregate — they
        are never returned or stored.

        Returns one OrgBenchmarkInsight per metric present in org_metrics.
        """
        epsilon = _epsilon_for_org(sensitivity_tier)
        dp = DifferentialPrivacyService(epsilon=epsilon)

        # Collect peer samples per metric
        peer_pools: dict[str, list[float]] = {}
        for sub in peer_metric_submissions:
            for metric, value in sub.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    peer_pools.setdefault(metric, []).append(float(value))

        insights: list[OrgBenchmarkInsight] = []
        for metric, org_value in org_metrics.items():
            peers = peer_pools.get(metric) or []
            if not peers:
                insights.append(OrgBenchmarkInsight(
                    metric=metric,
                    org_value=org_value,
                    global_private_mean=org_value,
                    percentile_rank=0.0,
                    gap_to_p75=0.0,
                    status="no_peer_data",
                ))
                continue

            # DP mean of peer values
            private_mean = float(dp.aggregate_with_privacy(
                org_data_dict={"peers": peers},
                aggregation_type="mean",
                epsilon=epsilon,
            ))

            pr = _percentile_rank(org_value, peers)
            p75 = sorted(peers)[int(0.75 * len(peers))]
            gap = round(p75 - org_value, 4)

            insights.append(OrgBenchmarkInsight(
                metric=metric,
                org_value=round(org_value, 4),
                global_private_mean=round(private_mean, 4),
                percentile_rank=round(pr, 4),
                gap_to_p75=gap,
                status="on_track" if pr >= 0.75 else "below_target",
            ))

        return insights

    def synthesize(
        self,
        *,
        agent_name: str,
        org_weight_submissions: list[dict[str, Any]],
        org_metrics: dict[str, float],
        peer_metric_submissions: list[dict[str, Any]],
        sensitivity_tier: str = "medium",
    ) -> FederatedLearningResult:
        """Full federated learning pass: aggregate weights + benchmark.

        Convenience method that runs both operations and returns a
        unified result with a confidence score and sharing recommendation.
        """
        agg_weights = self.aggregate_agent_weights(
            agent_name=agent_name,
            org_weight_submissions=org_weight_submissions,
        )
        benchmark_insights = self.benchmark_org(
            org_metrics=org_metrics,
            peer_metric_submissions=peer_metric_submissions,
            sensitivity_tier=sensitivity_tier,
        )

        total_budget = agg_weights.privacy_budget_spent + _epsilon_for_org(sensitivity_tier)

        # Confidence: rises with more contributing orgs and tighter privacy
        confidence = _clip(
            0.3
            + 0.15 * min(1.0, agg_weights.contributing_orgs / 5)
            + 0.25 * (1.0 - _clip(agg_weights.aggregation_epsilon / 10.0))
            + 0.30 * (len(benchmark_insights) > 0)
        )

        on_track_count = sum(1 for b in benchmark_insights if b.status == "on_track")
        if agg_weights.contributing_orgs < 3:
            recommendation = "hold: insufficient_contributors"
        elif agg_weights.aggregation_epsilon <= 0.1:
            recommendation = "share: high_privacy_aggregate_only"
        elif on_track_count == len(benchmark_insights) and benchmark_insights:
            recommendation = "share: peer_group_exchange"
        else:
            recommendation = "share: controlled_improvement_insights"

        return FederatedLearningResult(
            aggregated_weights=[agg_weights],
            benchmark_insights=benchmark_insights,
            sharing_recommendation=recommendation,
            total_privacy_budget_spent=round(total_budget, 4),
            federation_confidence=round(confidence, 4),
        )
