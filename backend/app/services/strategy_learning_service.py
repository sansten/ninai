"""Strategy Learning Service — Outcome-Driven Strategy Learning.

After every completed cognitive session, extracts a decision fingerprint
(goal_type, domain, tool_sequence, plan_confidence, iteration_count) and
updates an EMA-smoothed StrategyLibraryEntry.

Before each new planning call, retrieves the top-3 matching strategies and
returns them as warm-start hints for the PlannerAgent prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_iteration import CognitiveIteration
from app.models.cognitive_session import CognitiveSession
from app.models.strategy_library import StrategyLibraryEntry
from app.services.self_model_service import ema as _ema

logger = logging.getLogger(__name__)

_EMA_ALPHA = 0.25  # EMA smoothing factor for success rate updates
_MIN_SAMPLES = 3   # minimum sample count before a strategy is surfaced as a hint


class StrategyLearningService:
    """Outcome-driven strategy learning tied to the cognitive loop."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_session_outcome(self, org_id: str, session_id: str) -> None:
        """Extract a strategy fingerprint from a completed session and upsert it.

        Called post-session (fire-and-forget safe — exceptions are caught and logged).
        """
        try:
            await self._do_ingest(org_id, session_id)
        except Exception:
            logger.exception("StrategyLearningService.ingest_session_outcome failed for session %s", session_id)

    async def get_strategy_hints(
        self,
        org_id: str,
        goal_type: str,
        domain: str,
    ) -> dict[str, Any] | None:
        """Return the top-3 proven strategies for (org_id, goal_type, domain).

        Returns None if there are no qualifying entries (sample_count < _MIN_SAMPLES).
        Returns a dict ready to be serialised into the planner prompt.
        """
        stmt = (
            select(StrategyLibraryEntry)
            .where(
                and_(
                    StrategyLibraryEntry.organization_id == org_id,
                    StrategyLibraryEntry.goal_type == goal_type,
                    StrategyLibraryEntry.domain == domain,
                    StrategyLibraryEntry.sample_count >= _MIN_SAMPLES,
                )
            )
            .order_by(StrategyLibraryEntry.success_rate.desc())
            .limit(3)
        )
        result = await self.session.execute(stmt)
        entries = result.scalars().all()

        if not entries:
            return None

        strategies = [
            {
                "tool_sequence": entry.tool_sequence,
                "success_rate": round(entry.success_rate, 3),
                "avg_iterations": round(entry.avg_iterations, 1),
                "avg_plan_confidence": round(entry.avg_plan_confidence, 3),
                "sample_count": entry.sample_count,
            }
            for entry in entries
        ]

        return {
            "goal_type": goal_type,
            "domain": domain,
            "top_strategies": strategies,
            "note": (
                "These tool sequences have historically high success rates for similar goals. "
                "Prefer them when multiple tools could accomplish the same step."
            ),
        }

    async def recompute_strategy_stats(self, org_id: str) -> None:
        """No-op placeholder: EMA is updated incrementally in ingest_session_outcome.

        Can be extended later to recompute from the last-N sessions.
        """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _do_ingest(self, org_id: str, session_id: str) -> None:
        """Core ingest logic — loads session + iterations and upserts the entry."""
        # Load cognitive session
        sess_result = await self.session.execute(
            select(CognitiveSession).where(
                and_(
                    CognitiveSession.id == session_id,
                    CognitiveSession.organization_id == org_id,
                )
            )
        )
        sess = sess_result.scalar_one_or_none()
        if sess is None or sess.status not in ("succeeded", "failed", "aborted"):
            return  # only learn from completed sessions

        # Load all iterations for this session
        iter_result = await self.session.execute(
            select(CognitiveIteration)
            .where(CognitiveIteration.session_id == session_id)
            .order_by(CognitiveIteration.iteration_num.asc())
        )
        iterations = iter_result.scalars().all()
        if not iterations:
            return

        # --- Build fingerprint ---
        goal_type = _classify_goal_type(str(sess.goal or ""))
        domain = str((sess.context_snapshot or {}).get("domain", "general")).strip() or "general"

        # tool_sequence: collect unique ordered tools from the last iteration's plan
        last_iter = iterations[-1]
        plan_steps = (last_iter.plan_json or {}).get("steps", [])
        tool_sequence = [
            str(s["tool"])
            for s in plan_steps
            if isinstance(s, dict) and s.get("tool")
        ]

        # plan confidence: from last critique
        plan_confidence = float(
            (last_iter.critique_json or {}).get("confidence", 0.5) or 0.5
        )

        # evidence pattern: topic categories from evidence memory ids (simplified as count)
        evidence_memory_ids = list(
            (last_iter.metrics or {}).get("evidence_memory_ids", []) or []
        )
        evidence_pattern = evidence_memory_ids[:5]  # store up to 5 evidence IDs as pattern proxy

        iteration_count = len(iterations)
        outcome_value = 1.0 if sess.status == "succeeded" else 0.0

        # --- Upsert StrategyLibraryEntry with EMA ---
        stmt = select(StrategyLibraryEntry).where(
            and_(
                StrategyLibraryEntry.organization_id == org_id,
                StrategyLibraryEntry.goal_type == goal_type,
                StrategyLibraryEntry.domain == domain,
            )
        )
        existing_result = await self.session.execute(stmt)
        entry = existing_result.scalar_one_or_none()

        if entry is None:
            entry = StrategyLibraryEntry(
                organization_id=org_id,
                goal_type=goal_type,
                domain=domain,
                tool_sequence=tool_sequence,
                evidence_pattern=evidence_pattern,
                avg_plan_confidence=plan_confidence,
                avg_iterations=float(iteration_count),
                success_rate=outcome_value,
                sample_count=1,
            )
            self.session.add(entry)
        else:
            entry.success_rate = _ema(previous=entry.success_rate, observed=outcome_value, alpha=_EMA_ALPHA)
            entry.avg_plan_confidence = _ema(previous=entry.avg_plan_confidence, observed=plan_confidence, alpha=_EMA_ALPHA)
            # avg_iterations is unbounded (not [0,1]) so raw EMA is used directly
            entry.avg_iterations = (1 - _EMA_ALPHA) * entry.avg_iterations + _EMA_ALPHA * float(iteration_count)
            # Update tool sequence only when success rate improves above prior
            if outcome_value > entry.success_rate and tool_sequence:
                entry.tool_sequence = tool_sequence
                entry.evidence_pattern = evidence_pattern
            entry.sample_count += 1

        await self.session.flush()


# ------------------------------------------------------------------
# Goal type classifier (keyword heuristic)
# ------------------------------------------------------------------

_GOAL_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("information_retrieval", ["search", "find", "retrieve", "look up", "fetch", "get"]),
    ("analysis", ["analyze", "analyse", "explain", "understand", "why", "reason", "summarize"]),
    ("generation", ["create", "generate", "write", "draft", "compose", "build"]),
    ("evaluation", ["compare", "evaluate", "assess", "rank", "prioritize", "score"]),
    ("planning", ["plan", "strategy", "how", "steps", "roadmap", "schedule"]),
    ("self_improvement", ["improve", "optimize", "fix", "debug", "calibrate"]),
]


def _classify_goal_type(goal_text: str) -> str:
    """Classify a goal string into one of the known goal type buckets."""
    normalized = goal_text.lower()
    for goal_type, keywords in _GOAL_TYPE_KEYWORDS:
        if any(kw in normalized for kw in keywords):
            return goal_type
    return "general"
