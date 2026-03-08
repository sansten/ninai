"""
PR-6: Meta-Cognitive Planning Service

Executive control layer that decides how deeply and carefully the agent should think.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta_cognitive import StrategySelected
from app.models.meta_agent import MetaAgentRun
from app.models.self_model import SelfModelProfile


class MetaCognitiveService:
    """Service for query complexity estimation and strategy selection."""

    def __init__(self, session: Optional[AsyncSession] = None, org_id: Optional[str] = None):
        self.session = session
        self.org_id = org_id

    async def estimate_query_complexity(self, query: str) -> float:
        """
        Estimate query complexity on a 0-1 scale.

        Uses lightweight heuristics over query length and complexity signals.
        """
        if not query or not query.strip():
            return 0.0

        normalized = query.strip().lower()
        token_count = len(normalized.split())

        score = min(0.5, token_count / 60.0)

        complexity_terms = [
            "why",
            "compare",
            "tradeoff",
            "optimize",
            "counterfactual",
            "forecast",
            "prove",
            "multi-step",
            "uncertain",
            "irreversible",
            "risk",
            "strategy",
        ]
        signal_hits = sum(1 for term in complexity_terms if term in normalized)
        score += min(0.35, signal_hits * 0.06)

        if "?" in normalized:
            score += 0.05

        return round(min(1.0, score), 3)

    async def allocate_retrieval_budget(
        self,
        complexity: float,
        time_budget_seconds: int,
    ) -> int:
        """Allocate memory retrieval budget from complexity and time limits."""
        complexity = max(0.0, min(1.0, complexity))
        time_budget_seconds = max(5, time_budget_seconds)

        base = 8 + int(complexity * 42)
        time_cap = max(6, int(time_budget_seconds / 2))
        return max(6, min(base, time_cap))

    async def allocate_reasoning_depth(self, complexity: float) -> int:
        """Allocate reasoning chain depth from complexity score."""
        complexity = max(0.0, min(1.0, complexity))
        if complexity < 0.2:
            return 1
        if complexity < 0.45:
            return 2
        if complexity < 0.7:
            return 3
        if complexity < 0.9:
            return 4
        return 5

    async def select_strategy(
        self,
        query: str,
        complexity: float,
        time_budget_seconds: int = 30,
        confidence_threshold: float = 0.7,
    ) -> Dict[str, object]:
        """Select heuristic/deliberative/mixed/escalate strategy for a query."""
        complexity = max(0.0, min(1.0, complexity))
        retrieval_budget = await self.allocate_retrieval_budget(complexity, time_budget_seconds)
        reasoning_depth = await self.allocate_reasoning_depth(complexity)

        q = (query or "").lower()
        high_risk = any(term in q for term in ["irreversible", "compliance", "legal", "safety"])

        if high_risk and complexity >= 0.7:
            strategy = StrategySelected.ESCALATE.value
        elif complexity < 0.25 and time_budget_seconds <= 15:
            strategy = StrategySelected.HEURISTIC.value
        elif complexity >= 0.75:
            strategy = StrategySelected.DELIBERATIVE.value
        else:
            strategy = StrategySelected.MIXED.value

        expected_answer_quality = max(0.5, min(0.98, 0.55 + complexity * 0.4))
        verification_required = high_risk or complexity >= 0.7

        return {
            "query": query,
            "complexity_estimated": round(complexity, 3),
            "strategy_selected": strategy,
            "retrieval_budget": retrieval_budget,
            "reasoning_depth": reasoning_depth,
            "verification_required": verification_required,
            "confidence_threshold": confidence_threshold,
            "time_budget_seconds": time_budget_seconds,
            "expected_answer_quality": round(expected_answer_quality, 3),
            "created_at": datetime.utcnow().isoformat(),
        }

    async def should_verify(self, query: str, answer: str, confidence: float) -> bool:
        """Determine whether an answer should be cross-checked before return."""
        q = (query or "").lower()
        a = (answer or "").lower()

        risky = any(term in q for term in ["irreversible", "safety", "legal", "financial", "compliance"])
        uncertain_answer = any(term in a for term in ["maybe", "might", "possibly", "uncertain"])

        return risky or confidence < 0.72 or uncertain_answer

    async def uncertainty_aware_response(
        self,
        answer: str,
        confidence: float,
        threshold: float,
    ) -> str:
        """Wrap low-confidence responses with explicit confidence signaling."""
        if confidence >= threshold:
            return answer

        confidence_pct = int(round(confidence * 100))
        return (
            f"I'm about {confidence_pct}% confident in this answer. "
            f"Treat it as directional guidance and verify critical details.\n\n{answer}"
        )

    async def learn_strategy_effectiveness(
        self,
        strategy_id: str,
        actual_confidence: float,
        expected_answer_quality: float = 0.7,
    ) -> Dict[str, object]:
        """Compute post-hoc strategy effectiveness score for learning loops."""
        actual_confidence = max(0.0, min(1.0, actual_confidence))
        expected_answer_quality = max(0.0, min(1.0, expected_answer_quality))

        gap = abs(actual_confidence - expected_answer_quality)
        effectiveness = max(0.0, 1.0 - gap)

        return {
            "strategy_id": strategy_id,
            "actual_confidence": round(actual_confidence, 3),
            "expected_answer_quality": round(expected_answer_quality, 3),
            "strategy_effectiveness": round(effectiveness, 3),
            "learned_at": datetime.utcnow().isoformat(),
        }

    async def get_confidence_calibration(self) -> Dict[str, float]:
        """Return confidence calibration computed from MetaAgentRun outcomes.

        confidence_calibration: avg final_confidence across recent runs
        surprise_frequency: fraction of runs that were confident (>0.7) yet failed
        Fallback to neutral defaults if no session or no data.
        """
        if not self.session or not self.org_id:
            return {"confidence_calibration": 0.70, "surprise_frequency": 0.0, "well_calibrated": True}

        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        stmt = (
            select(MetaAgentRun)
            .where(
                and_(
                    MetaAgentRun.organization_id == self.org_id,
                    MetaAgentRun.created_at >= cutoff,
                    MetaAgentRun.final_confidence.isnot(None),
                )
            )
            .order_by(MetaAgentRun.created_at.desc())
            .limit(200)
        )
        result = await self.session.execute(stmt)
        runs = result.scalars().all()

        if not runs:
            return {"confidence_calibration": 0.70, "surprise_frequency": 0.0, "well_calibrated": True}

        confidences = [r.final_confidence for r in runs if r.final_confidence is not None]
        avg_confidence = sum(confidences) / len(confidences)

        # Surprised = agent was confident (>0.7) but run failed
        surprises = sum(
            1 for r in runs if (r.final_confidence or 0.0) > 0.7 and r.status == "failed"
        )
        surprise_frequency = surprises / len(runs)

        return {
            "confidence_calibration": round(avg_confidence, 3),
            "surprise_frequency": round(surprise_frequency, 3),
            "well_calibrated": surprise_frequency < 0.25,
        }

    async def get_epistemic_state(self) -> Dict[str, object]:
        """Return epistemic snapshot derived from SelfModelProfile domain_confidence.

        known_domains: confidence >= 0.70
        uncertain_domains: 0.40 <= confidence < 0.70
        unknown_domains: confidence < 0.40
        Calibration data re-uses get_confidence_calibration() to avoid duplication.
        """
        known_domains: list[str] = []
        uncertain_domains: list[str] = []
        unknown_domains: list[str] = []

        if self.session and self.org_id:
            stmt = select(SelfModelProfile).where(SelfModelProfile.organization_id == self.org_id)
            result = await self.session.execute(stmt)
            profile = result.scalar_one_or_none()

            if profile and profile.domain_confidence:
                for domain, conf in profile.domain_confidence.items():
                    conf_val = float(conf) if isinstance(conf, (int, float)) else 0.5
                    if conf_val >= 0.70:
                        known_domains.append(domain)
                    elif conf_val >= 0.40:
                        uncertain_domains.append(domain)
                    else:
                        unknown_domains.append(domain)

        calibration = await self.get_confidence_calibration()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "known_domains": known_domains,
            "uncertain_domains": uncertain_domains,
            "unknown_domains": unknown_domains,
            "confidence_calibration": calibration["confidence_calibration"],
            "surprise_frequency": calibration["surprise_frequency"],
        }
