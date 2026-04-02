"""Multi-agent voting engine (Phase 75)."""

from __future__ import annotations

import math
from statistics import pstdev
from typing import Any


class MultiAgentVotingEngine:
    AGENT_WEIGHTS = {
        "credibility_agent": 0.90,
        "anomaly_detection_agent": 0.85,
        "conflict_detection_agent": 0.85,
        "uncertainty_reporting_agent": 0.80,
        "hypothesis_service": 0.75,
        "causal_reasoning_agent": 0.80,
        "default": 0.70,
    }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _weight_for(self, agent_name: str) -> float:
        return float(self.AGENT_WEIGHTS.get(str(agent_name or "").lower(), self.AGENT_WEIGHTS["default"]))

    def vote(self, *, ballots: list[dict]) -> dict[str, Any]:
        rows = ballots or []
        if not rows:
            return {
                "consensus": False,
                "weighted_true": 0.0,
                "weighted_false": 0.0,
                "margin": 0.0,
                "agreement_rate": 0.0,
                "dissenting_agents": [],
            }

        weighted_true = 0.0
        weighted_false = 0.0
        for ballot in rows:
            weight = self._weight_for(str(ballot.get("agent_name", "")))
            confidence = self._clamp(float(ballot.get("confidence", 0.0) or 0.0))
            if bool(ballot.get("verdict", False)):
                weighted_true += weight * confidence
            else:
                weighted_false += weight * confidence

        consensus = weighted_true > weighted_false
        total_weighted = weighted_true + weighted_false
        margin = abs(weighted_true - weighted_false) / total_weighted if total_weighted > 0 else 0.0

        votes_matching_consensus = 0
        dissenting: list[str] = []
        for ballot in rows:
            agent_name = str(ballot.get("agent_name", "unknown"))
            verdict = bool(ballot.get("verdict", False))
            if verdict == consensus:
                votes_matching_consensus += 1
            else:
                dissenting.append(agent_name)

        agreement_rate = votes_matching_consensus / len(rows)

        return {
            "consensus": consensus,
            "weighted_true": round(weighted_true, 6),
            "weighted_false": round(weighted_false, 6),
            "margin": round(margin, 6),
            "agreement_rate": round(agreement_rate, 6),
            "dissenting_agents": dissenting,
        }

    def resolve_numeric(self, *, ballots: list[dict]) -> dict[str, float]:
        rows = ballots or []
        if not rows:
            return {
                "consensus_value": 0.0,
                "std_deviation": 0.0,
                "min_value": 0.0,
                "max_value": 0.0,
            }

        weighted_sum = 0.0
        weight_total = 0.0
        values: list[float] = []

        for ballot in rows:
            value = float(ballot.get("value", 0.0) or 0.0)
            confidence = self._clamp(float(ballot.get("confidence", 0.0) or 0.0))
            weight = self._weight_for(str(ballot.get("agent_name", ""))) * confidence
            values.append(value)
            weighted_sum += weight * value
            weight_total += weight

        consensus_value = weighted_sum / weight_total if weight_total > 0 else 0.0
        std_deviation = pstdev(values) if len(values) > 1 else 0.0

        return {
            "consensus_value": round(consensus_value, 6),
            "std_deviation": round(float(std_deviation), 6),
            "min_value": round(min(values), 6),
            "max_value": round(max(values), 6),
        }

    def is_contested(self, *, margin: float) -> bool:
        return float(margin) < 0.2
