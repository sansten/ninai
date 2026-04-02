"""Uncertainty propagation engine (Phase 64)."""

from __future__ import annotations


class UncertaintyPropagationService:
    PROPAGATION_DECAY = 0.7
    MIN_PROPAGATED = 0.05

    def propagate(
        self,
        *,
        source_uncertainty: float,
        inference_hops: int,
        corroborating_evidence_count: int,
    ) -> float:
        source = min(1.0, max(0.0, float(source_uncertainty)))
        hops = max(0, int(inference_hops))
        evidence = max(0, int(corroborating_evidence_count))

        attenuated = source * (self.PROPAGATION_DECAY ** hops)
        corroboration_factor = 1.0 / (1.0 + 0.2 * evidence)
        propagated = attenuated * corroboration_factor
        return max(self.MIN_PROPAGATED, round(propagated, 6))

    def propagate_chain(
        self,
        *,
        sources: list[dict],
        chain_length: int,
    ) -> float:
        if not sources:
            return self.MIN_PROPAGATED

        hops = max(0, int(chain_length))
        values: list[float] = []
        for source in sources:
            propagated = self.propagate(
                source_uncertainty=float(source.get("uncertainty", 0.0) or 0.0),
                inference_hops=hops,
                corroborating_evidence_count=int(source.get("corroborating_evidence_count", 0) or 0),
            )
            values.append(propagated)

        if not values:
            return self.MIN_PROPAGATED

        product = 1.0
        for value in values:
            product *= max(self.MIN_PROPAGATED, value)
        geometric_mean = product ** (1.0 / len(values))
        return max(self.MIN_PROPAGATED, round(geometric_mean, 6))

    def uncertainty_label(self, uncertainty: float) -> str:
        value = min(1.0, max(0.0, float(uncertainty)))
        if value < 0.1:
            return "certain"
        if value < 0.3:
            return "likely"
        if value < 0.6:
            return "uncertain"
        return "speculative"

    def should_flag_for_review(
        self,
        *,
        propagated_uncertainty: float,
        decision_stakes: str,
    ) -> bool:
        value = min(1.0, max(0.0, float(propagated_uncertainty)))
        stakes = str(decision_stakes or "low").strip().lower()

        thresholds = {
            "critical": 0.1,
            "high": 0.25,
            "medium": 0.5,
            "low": 0.75,
        }
        threshold = thresholds.get(stakes, thresholds["low"])
        return value > threshold
