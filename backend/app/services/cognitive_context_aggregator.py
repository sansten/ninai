"""Cross-Service Intelligence Bus — CognitiveContextAggregator.

Collects signals from MetaCognitiveService and IntrinsicMotivationService and
packages them as a structured dict for injection into the PlannerAgent prompt.

Design principles:
- Each service call is fail-isolated: one service failure never blocks the rest.
- Returns an empty dict if all services are unavailable or raise.
- Compact output only — no raw DB objects, no large payloads.
- Called once per cognitive session (before the planning loop).
- Uses sequential awaits because the wired services typically share one
  AsyncSession; concurrent DB use on a single session is unsafe.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.meta_cognitive_service import MetaCognitiveService
    from app.services.intrinsic_motivation_service import IntrinsicMotivationService

logger = logging.getLogger(__name__)

_MAX_GAPS = 3  # maximum knowledge gaps to surface in context


class CognitiveContextAggregator:
    """Aggregates cross-service intelligence signals for the planner."""

    def __init__(
        self,
        *,
        meta_cognitive: MetaCognitiveService | None = None,
        intrinsic_motivation: IntrinsicMotivationService | None = None,
    ) -> None:
        self.meta_cognitive = meta_cognitive
        self.intrinsic_motivation = intrinsic_motivation

    async def aggregate(
        self,
        *,
        org_id: str,
        goal: str,
    ) -> dict[str, Any]:
        """Collect intelligence signals from all wired services.

        Returns a dict ready to be serialised as ``{cognitive_context_json}`` in
        the planner prompt.  Keys present only when the underlying service is
        available and returns non-empty data.
        """
        context: dict[str, Any] = {}

        # These services usually share one AsyncSession via dependency wiring,
        # so we fetch sequentially to avoid concurrent use of the same session.
        epistemic = await self._fetch_epistemic_state(goal)
        gaps = await self._fetch_knowledge_gaps(org_id)

        if epistemic:
            context["epistemic_state"] = epistemic
        if gaps:
            context["knowledge_gaps"] = gaps

        return context

    # ------------------------------------------------------------------
    # Private fetchers (each self-contained with exception handling)
    # ------------------------------------------------------------------

    async def _fetch_epistemic_state(self, goal: str) -> dict[str, Any] | None:
        if not self.meta_cognitive:
            return None
        try:
            state = await self.meta_cognitive.get_epistemic_state()
            complexity = await self.meta_cognitive.estimate_query_complexity(goal)
            return {
                "known_domains": list(state.get("known_domains") or []),
                "uncertain_domains": list(state.get("uncertain_domains") or []),
                "unknown_domains": list(state.get("unknown_domains") or []),
                "confidence_calibration": round(float(state.get("confidence_calibration", 0.7)), 3),
                "query_complexity": round(float(complexity), 3),
            }
        except Exception:
            logger.debug("CognitiveContextAggregator: epistemic_state fetch failed", exc_info=True)
            return None

    async def _fetch_knowledge_gaps(self, org_id: str) -> list[dict[str, Any]] | None:
        if not self.intrinsic_motivation:
            return None
        try:
            gaps = await self.intrinsic_motivation.detect_knowledge_gaps(org_id)
            if not gaps:
                return None
            top = sorted(gaps, key=lambda g: float(g.get("confidence_in_gap") or 0), reverse=True)
            top = top[:_MAX_GAPS]
            return [
                {
                    "gap_type": g.get("gap_type"),
                    "domain": g.get("domain"),
                    "description": g.get("description"),
                    "confidence": round(float(g.get("confidence_in_gap") or 0), 3),
                }
                for g in top
            ]
        except Exception:
            logger.debug("CognitiveContextAggregator: knowledge_gaps fetch failed", exc_info=True)
            return None
