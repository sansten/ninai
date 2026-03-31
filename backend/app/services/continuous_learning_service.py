"""Continuous Learning Pipeline services — Phase 50.

Implements the autonomous closed loop:
- Track action outcomes per strategy domain
- Evolve strategy library entries (promote/prune)
- Track playbook effectiveness
- Surface drift alerts when win rates drop sharply
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.action_execution_record import ActionExecutionRecord
from app.models.cognitive_session import CognitiveSession
from app.models.strategy_library import StrategyLibraryEntry
from app.services.strategy_learning_service import _classify_goal_type


@dataclass(frozen=True)
class StrategyMetric:
    goal_type: str
    domain: str
    total: int
    successes: int
    success_rate: float


@dataclass(frozen=True)
class PlaybookMetric:
    playbook_id: str
    total: int
    successes: int
    success_rate: float


@dataclass(frozen=True)
class StrategyDriftAlert:
    strategy_key: str
    previous_success_rate: float
    current_success_rate: float
    delta: float
    severity: str


@dataclass(frozen=True)
class EvolutionResult:
    promoted: list[str]
    pruned: list[str]
    strategy_metrics: list[StrategyMetric]
    playbook_metrics: list[PlaybookMetric]
    drift_alerts: list[StrategyDriftAlert]


def _is_success_status(status: str | None) -> bool:
    return str(status or "").strip().lower() == "success"


def _aggregate_strategy_metrics(outcomes: list[dict[str, Any]]) -> list[StrategyMetric]:
    buckets: dict[tuple[str, str], dict[str, int]] = {}
    for row in outcomes:
        goal_type = str(row.get("goal_type") or "general").strip() or "general"
        domain = str(row.get("domain") or "general").strip() or "general"
        key = (goal_type, domain)

        acc = buckets.setdefault(key, {"total": 0, "successes": 0})
        acc["total"] += 1
        if bool(row.get("success")):
            acc["successes"] += 1

    result: list[StrategyMetric] = []
    for (goal_type, domain), acc in buckets.items():
        total = int(acc["total"])
        successes = int(acc["successes"])
        rate = float(successes / total) if total else 0.0
        result.append(
            StrategyMetric(
                goal_type=goal_type,
                domain=domain,
                total=total,
                successes=successes,
                success_rate=round(rate, 4),
            )
        )
    return sorted(result, key=lambda x: (x.goal_type, x.domain))


def _aggregate_playbook_metrics(outcomes: list[dict[str, Any]]) -> list[PlaybookMetric]:
    buckets: dict[str, dict[str, int]] = {}
    for row in outcomes:
        playbook_id = str(row.get("matched_playbook_id") or "").strip()
        if not playbook_id:
            continue

        acc = buckets.setdefault(playbook_id, {"total": 0, "successes": 0})
        acc["total"] += 1
        if bool(row.get("success")):
            acc["successes"] += 1

    result: list[PlaybookMetric] = []
    for playbook_id, acc in buckets.items():
        total = int(acc["total"])
        successes = int(acc["successes"])
        rate = float(successes / total) if total else 0.0
        result.append(
            PlaybookMetric(
                playbook_id=playbook_id,
                total=total,
                successes=successes,
                success_rate=round(rate, 4),
            )
        )
    return sorted(result, key=lambda x: x.playbook_id)


def _build_drift_alerts(
    *,
    previous: list[StrategyMetric],
    current: list[StrategyMetric],
    min_drop: float = 0.20,
    min_previous_rate: float = 0.60,
) -> list[StrategyDriftAlert]:
    prev_map = {(m.goal_type, m.domain): m for m in previous}
    alerts: list[StrategyDriftAlert] = []

    for now in current:
        prior = prev_map.get((now.goal_type, now.domain))
        if prior is None:
            continue
        delta = float(now.success_rate - prior.success_rate)
        if prior.success_rate >= min_previous_rate and delta <= -abs(min_drop):
            severity = "critical" if delta <= -0.30 else "warning"
            alerts.append(
                StrategyDriftAlert(
                    strategy_key=f"{now.goal_type}:{now.domain}",
                    previous_success_rate=round(prior.success_rate, 4),
                    current_success_rate=round(now.success_rate, 4),
                    delta=round(delta, 4),
                    severity=severity,
                )
            )

    return sorted(alerts, key=lambda a: a.delta)


class OutcomeTrackerService:
    """Loads action outcomes and computes per-strategy effectiveness."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def load_outcomes(
        self,
        *,
        org_id: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(
                ActionExecutionRecord.status,
                ActionExecutionRecord.payload_summary,
                CognitiveSession.goal,
                CognitiveSession.context_snapshot,
            )
            .select_from(ActionExecutionRecord)
            .join(
                CognitiveSession,
                CognitiveSession.id == ActionExecutionRecord.session_id,
                isouter=True,
            )
            .where(
                and_(
                    ActionExecutionRecord.organization_id == org_id,
                    ActionExecutionRecord.created_at >= start,
                    ActionExecutionRecord.created_at < end,
                )
            )
        )

        res = await self.session.execute(stmt)
        rows = res.all()

        outcomes: list[dict[str, Any]] = []
        for status, payload_summary, goal, context_snapshot in rows:
            context = context_snapshot or {}
            domain = str(context.get("domain") or "general").strip() or "general"
            goal_type = _classify_goal_type(str(goal or ""))
            payload = payload_summary or {}

            outcomes.append(
                {
                    "goal_type": goal_type,
                    "domain": domain,
                    "success": _is_success_status(status),
                    "matched_playbook_id": payload.get("matched_playbook_id"),
                }
            )

        return outcomes


class StrategyEvolutionService:
    """Nightly evolution of strategy library entries using action outcomes."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.tracker = OutcomeTrackerService(session)

    async def run(
        self,
        *,
        org_id: str,
        window_days: int = 30,
        promote_threshold: float = 0.80,
        prune_threshold: float = 0.20,
        min_samples: int = 5,
    ) -> EvolutionResult:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=max(1, int(window_days)))

        outcomes = await self.tracker.load_outcomes(org_id=org_id, start=start, end=now)
        strategy_metrics = _aggregate_strategy_metrics(outcomes)
        playbook_metrics = _aggregate_playbook_metrics(outcomes)

        prev_start = start - timedelta(days=7)
        prev_end = start
        prev_outcomes = await self.tracker.load_outcomes(org_id=org_id, start=prev_start, end=prev_end)
        prev_metrics = _aggregate_strategy_metrics(prev_outcomes)
        drift_alerts = _build_drift_alerts(previous=prev_metrics, current=strategy_metrics)

        stmt = select(StrategyLibraryEntry).where(StrategyLibraryEntry.organization_id == org_id)
        result = await self.session.execute(stmt)
        entries = result.scalars().all()
        by_key = {(e.goal_type, e.domain): e for e in entries}

        promoted: list[str] = []
        pruned: list[str] = []

        for metric in strategy_metrics:
            key = (metric.goal_type, metric.domain)
            entry = by_key.get(key)
            if entry is None:
                continue

            # Keep strategy stats aligned with real outcome window.
            entry.success_rate = float(metric.success_rate)
            entry.sample_count = max(int(entry.sample_count or 0), int(metric.total))

            if metric.total >= min_samples and metric.success_rate >= promote_threshold:
                promoted.append(str(entry.id))
            elif metric.total >= min_samples and metric.success_rate <= prune_threshold:
                pruned.append(str(entry.id))
                await self.session.delete(entry)

        await self.session.flush()

        return EvolutionResult(
            promoted=sorted(promoted),
            pruned=sorted(pruned),
            strategy_metrics=strategy_metrics,
            playbook_metrics=playbook_metrics,
            drift_alerts=drift_alerts,
        )
