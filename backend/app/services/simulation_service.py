"""SimulationService.

Deterministic (non-LLM) plan simulation for the CognitiveLoop.

Design goals:
- Fast and side-effect free.
- Does not require raw memory content (uses summary-only evidence cards).
"""

from __future__ import annotations

import math
from typing import Any

from app.schemas.cognitive import PlannerOutput
from app.schemas.simulation import (
    SimulationOutput,
    SimulationAddStep,
    SimulationPlanRisk,
    SimulationRecommendedPlanPatch,
    SimulationRiskFactor,
)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _logistic(x: float) -> float:
    # numerically stable enough for our small magnitudes
    return float(1.0 / (1.0 + math.exp(-float(x))))


class SimulationService:
    """Compute deterministic risk signals + a conservative patch suggestion."""

    def simulate_memory_promotion(
        self,
        *,
        memory_content: str,
        access_count: int,
        importance_score: float,
        memory_scope: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Simulate a memory promotion decision to assess risk and suitability.
        
        Returns:
            dict with keys:
                - should_promote (bool): Recommendation based on simulation
                - confidence (float): Confidence in the decision (0-1)
                - risk_factors (list[str]): Any detected risks
                - metadata (dict): Simulation details for audit trail
        """
        tags = tags or []
        
        # Base promotion score from access patterns and importance
        access_factor = min(1.0, access_count / 5.0)  # Normalize against 5 accesses
        combined_score = 0.6 * importance_score + 0.4 * access_factor
        
        # Risk assessment
        risk_factors = []
        risk_score = 0.0
        
        # Check for sensitive keywords
        sensitive_keywords = ["password", "secret", "api_key", "token", "credential"]
        content_lower = memory_content.lower()
        if any(kw in content_lower for kw in sensitive_keywords):
            risk_factors.append("Contains sensitive keywords - may need restricted classification")
            risk_score += 0.3
        
        # Check scope
        if memory_scope in ["restricted", "confidential"]:
            risk_factors.append(f"High-privilege scope ({memory_scope}) requires careful promotion")
            risk_score += 0.2
        
        # Check content length (very short content may not be worth promoting)
        if len(memory_content) < 20:
            risk_factors.append("Very short content - may not provide sufficient value for LTM")
            risk_score += 0.15
        
        # Final decision
        adjusted_score = combined_score - (0.5 * risk_score)
        should_promote = adjusted_score >= 0.5  # Conservative threshold
        confidence = _clamp01(adjusted_score)
        
        return {
            "should_promote": should_promote,
            "confidence": confidence,
            "risk_factors": risk_factors,
            "metadata": {
                "access_count": access_count,
                "importance_score": importance_score,
                "scope": memory_scope,
                "combined_score": combined_score,
                "risk_score": risk_score,
                "adjusted_score": adjusted_score,
                "content_length": len(memory_content),
            },
        }

    def compute_evidence_strength(self, evidence_cards: list[dict[str, Any]]) -> float:
        if not evidence_cards:
            return 0.0

        scores: list[float] = []
        for c in evidence_cards:
            try:
                s = float(c.get("score") or 0.0)
            except Exception:
                s = 0.0
            # Qdrant scores may exceed 1.0 depending on distance; clamp to 0..1.
            scores.append(_clamp01(s))

        mean_score = sum(scores) / max(1, len(scores))
        count_factor = _clamp01(len(evidence_cards) / 5.0)
        return _clamp01(0.5 * mean_score + 0.5 * count_factor)

    def compute_tool_failure_risk(self, required_tools: list[str], self_model: dict[str, Any] | None) -> float:
        if not required_tools:
            return 0.05

        tool_stats = (self_model or {}).get("tool_reliability") if isinstance(self_model, dict) else None
        if not isinstance(tool_stats, dict):
            tool_stats = {}

        rates: list[float] = []
        for tool in required_tools:
            stats = tool_stats.get(tool)
            if isinstance(stats, dict) and stats.get("success_rate_30d") is not None:
                try:
                    rates.append(_clamp01(float(stats.get("success_rate_30d"))))
                except Exception:
                    rates.append(0.85)
            else:
                rates.append(0.85)

        avg_rate = sum(rates) / max(1, len(rates))
        return _clamp01(1.0 - avg_rate)

    def compute_policy_risk(self, plan: PlannerOutput, evidence_cards: list[dict[str, Any]]) -> float:
        # Conservative heuristic based on evidence classification + tool usage.
        class_weight = {"public": 0.0, "internal": 0.05, "confidential": 0.15, "restricted": 0.30}
        max_cls = 0.0
        for c in evidence_cards or []:
            cls = str(c.get("classification") or "internal").lower().strip()
            max_cls = max(max_cls, float(class_weight.get(cls, 0.10)))

        tool_steps = sum(1 for s in (plan.steps or []) if s.tool)
        tool_risk = 0.0 if tool_steps <= 1 else _clamp01(0.05 * (tool_steps - 1))
        return _clamp01(max_cls + tool_risk)

    def simulate_plan(
        self,
        *,
        plan: PlannerOutput,
        evidence_cards: list[dict[str, Any]],
        self_model: dict[str, Any] | None = None,
        policies: dict[str, Any] | None = None,
    ) -> SimulationOutput:
        _ = policies

        evidence_strength = self.compute_evidence_strength(evidence_cards)
        tool_failure_prob = self.compute_tool_failure_risk(plan.required_tools or [], self_model)
        policy_risk = self.compute_policy_risk(plan, evidence_cards)

        # Simple bounded mapping.
        # Higher evidence strength and tool reliability increase success.
        success = _logistic(
            2.2 * (evidence_strength - 0.5)
            + 1.8 * ((1.0 - tool_failure_prob) - 0.8)
            - 2.0 * policy_risk
        )

        policy_violation = _clamp01(_logistic(3.0 * (policy_risk - 0.15)))
        data_leak = _clamp01(0.15 * policy_risk)

        risk_factors: list[SimulationRiskFactor] = []
        patch = SimulationRecommendedPlanPatch()

        if evidence_strength < 0.50:
            risk_factors.append(
                SimulationRiskFactor(
                    type="insufficient_evidence",
                    description="Evidence strength is low; plan likely needs more supporting memories before execution.",
                    affected_steps=[s.step_id for s in (plan.steps or []) if s.step_id],
                    mitigation="Add an evidence-gathering step and re-plan once evidence improves.",
                )
            )

            # Recommend adding a conservative evidence gathering step (no new tools).
            patch.add_steps.append(
                SimulationAddStep(
                    step_id="S_EVIDENCE",
                    action="Retrieve additional evidence from memory before proceeding.",
                    tool="memory.search",
                    tool_input_hint={"query": plan.objective, "limit": 12, "hybrid": True},
                    expected_output="Additional evidence cards (summaries only) relevant to the objective.",
                    success_criteria=["At least 5 relevant evidence cards."],
                )
            )

        if tool_failure_prob > 0.25:
            risk_factors.append(
                SimulationRiskFactor(
                    type="tool_unreliable",
                    description="One or more required tools appear unreliable based on self-model stats.",
                    affected_steps=[s.step_id for s in (plan.steps or []) if s.tool],
                    mitigation="Prefer alternative tools or add retries/human confirmation.",
                )
            )

        if policy_risk > 0.25:
            risk_factors.append(
                SimulationRiskFactor(
                    type="policy_risk",
                    description="Plan touches higher-classification evidence or has multiple tool steps; policy risk elevated.",
                    affected_steps=[s.step_id for s in (plan.steps or []) if s.step_id],
                    mitigation="Reduce scope, add justification, and ensure required permissions before execution.",
                )
            )

        confidence = _clamp01(0.35 + 0.45 * evidence_strength + 0.25 * (1.0 - tool_failure_prob) - 0.30 * policy_risk)

        evidence_ids = [str(c.get("id")) for c in (evidence_cards or []) if c.get("id")]

        return SimulationOutput(
            plan_risk=SimulationPlanRisk(
                success_probability=_clamp01(success),
                policy_violation_probability=_clamp01(policy_violation),
                data_leak_probability=_clamp01(data_leak),
                tool_failure_probability=_clamp01(tool_failure_prob),
            ),
            risk_factors=risk_factors,
            recommended_plan_patch=patch,
            confidence=confidence,
            evidence_memory_ids=evidence_ids,
        )
