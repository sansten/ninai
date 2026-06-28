"""Agent Contribution Estimator — Phase 94.

Estimates each agent's marginal contribution to pipeline outcomes using a
Monte Carlo approximation of Shapley values.

The exact Shapley computation requires 2^n evaluations for n agents — infeasible
at n=85. Monte Carlo Shapley approximates it with random permutation sampling:

    For each agent a:
        for _ in range(n_samples):
            perm = random_permutation(all_agents)
            coalition_without = {agents before a in perm}
            coalition_with = coalition_without ∪ {a}
            marginal = metric_fn(coalition_with, outcomes) - metric_fn(coalition_without, outcomes)
        shapley[a] = mean(marginals)

This converges in O(n * n_samples) instead of O(2^n), which is practical for
85-agent pipelines with n_samples=50–200.

The metric_fn must accept a frozenset of agent_ids and a list of outcomes and
return a scalar performance score (higher = better). Callers supply the metric;
this service handles the sampling and averaging.

Usage::

    svc = AgentContributionEstimatorService(n_samples=100, seed=42)

    def metric(coalition, outcomes):
        # fraction of outcomes contributed by any agent in coalition
        relevant = [o for o in outcomes if o.get("agent_id") in coalition]
        return len(relevant) / max(len(outcomes), 1)

    outcomes = [{"agent_id": "a1", "success": True}, ...]
    scores = svc.estimate(agents=["a1", "a2", "a3"], outcomes=outcomes, metric_fn=metric)
    # scores: {"a1": 0.42, "a2": 0.18, "a3": 0.38, ...}
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

MetricFn = Callable[[frozenset[str], list[dict[str, Any]]], float]


@dataclass
class ContributionReport:
    agents: list[str]
    shapley_values: dict[str, float]       # agent_id → Shapley value
    normalised_values: dict[str, float]    # as fraction of total
    n_samples: int
    n_agents: int
    top_contributors: list[str]            # sorted by contribution desc
    low_contributors: list[str]            # bottom 20%, sorted asc
    total_value: float

    @property
    def most_valuable(self) -> str | None:
        return self.top_contributors[0] if self.top_contributors else None


@dataclass
class _SampleState:
    """Running mean state for each agent."""
    total: float = 0.0
    count: int = 0

    def update(self, marginal: float) -> None:
        self.total += marginal
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


class AgentContributionEstimatorService:
    """Monte Carlo Shapley estimator for multi-agent pipeline contribution.

    Parameters
    ----------
    n_samples:
        Number of random permutations to sample per agent.
        50 is adequate for directional rankings; 200 gives tighter estimates.
    seed:
        Optional RNG seed for reproducibility.
    top_fraction:
        Fraction of agents classified as "top contributors" (default 0.20).
    low_fraction:
        Fraction of agents classified as "low contributors" (default 0.20).
    """

    def __init__(
        self,
        n_samples: int = 100,
        seed: int | None = None,
        top_fraction: float = 0.20,
        low_fraction: float = 0.20,
    ) -> None:
        self._n_samples = max(10, int(n_samples))
        self._rng = random.Random(seed)
        self._top_fraction = max(0.0, min(1.0, float(top_fraction)))
        self._low_fraction = max(0.0, min(1.0, float(low_fraction)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        agents: list[str],
        outcomes: list[dict[str, Any]],
        metric_fn: MetricFn,
    ) -> ContributionReport:
        """Estimate Shapley values for all agents.

        Parameters
        ----------
        agents:
            List of agent identifiers participating in the pipeline.
        outcomes:
            List of outcome records (dicts with at minimum an "agent_id" key).
        metric_fn:
            Callable (coalition: frozenset[str], outcomes: list[dict]) → float.
            Returns the performance score for the given coalition.
        """
        if not agents:
            return self._empty_report()

        n = len(agents)
        states: dict[str, _SampleState] = {a: _SampleState() for a in agents}

        for _ in range(self._n_samples):
            perm = list(agents)
            self._rng.shuffle(perm)

            coalition: set[str] = set()
            prev_score = metric_fn(frozenset(), outcomes)

            for agent_id in perm:
                coalition.add(agent_id)
                score = metric_fn(frozenset(coalition), outcomes)
                marginal = score - prev_score
                states[agent_id].update(marginal)
                prev_score = score

        shapley: dict[str, float] = {a: states[a].mean for a in agents}
        total = sum(shapley.values())

        if total > 0:
            normalised = {a: v / total for a, v in shapley.items()}
        else:
            n_agents = len(agents)
            normalised = {a: 1.0 / n_agents for a in agents}

        sorted_agents = sorted(agents, key=lambda a: shapley[a], reverse=True)
        top_n = max(1, round(len(agents) * self._top_fraction))
        low_n = max(1, round(len(agents) * self._low_fraction))

        return ContributionReport(
            agents=list(agents),
            shapley_values=shapley,
            normalised_values=normalised,
            n_samples=self._n_samples,
            n_agents=len(agents),
            top_contributors=sorted_agents[:top_n],
            low_contributors=sorted_agents[max(0, len(sorted_agents) - low_n):],
            total_value=total,
        )

    def estimate_incremental(
        self,
        agents: list[str],
        outcomes: list[dict[str, Any]],
        metric_fn: MetricFn,
        exclude: set[str] | None = None,
    ) -> dict[str, float]:
        """Estimate with a subset excluded (coalition ablation).

        Useful for measuring how much a specific sub-team contributes by
        comparing full-pipeline score vs. score without that sub-team.

        Returns per-agent Shapley values for the included agents only.
        """
        excluded = set(exclude or [])
        active = [a for a in agents if a not in excluded]
        if not active:
            return {}
        report = self.estimate(active, outcomes, metric_fn)
        return report.shapley_values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_report(self) -> ContributionReport:
        return ContributionReport(
            agents=[],
            shapley_values={},
            normalised_values={},
            n_samples=self._n_samples,
            n_agents=0,
            top_contributors=[],
            low_contributors=[],
            total_value=0.0,
        )
