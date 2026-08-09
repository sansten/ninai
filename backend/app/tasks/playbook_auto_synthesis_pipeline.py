"""Nightly playbook auto-synthesis pipeline — Phase 81.

Queries AutonomousGoalOutcome records from the last N days, clusters them
by impact-description fingerprint, and writes qualifying Playbook rows.
Runs at 04:00 UTC (after strategy-evolution at 03:30).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select

from app.agents.playbook_auto_synthesis_agent import synthesize_playbooks
from app.core.celery_app import celery_app
from app.core.database import async_session_factory
from app.models.autonomous_goal_outcome import AutonomousGoalOutcome
from app.models.playbook import Playbook, PlaybookScopeType
from app.tasks.async_runtime import run_async

logger = logging.getLogger(__name__)


async def _run_for_org(org_id: str, window_days: int) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    async with async_session_factory() as db:
        async with db.begin():
            rows = (
                await db.execute(
                    select(AutonomousGoalOutcome)
                    .where(
                        AutonomousGoalOutcome.organization_id == org_id,
                        AutonomousGoalOutcome.created_at >= cutoff,
                    )
                    .order_by(AutonomousGoalOutcome.created_at.desc())
                    .limit(500)
                )
            ).scalars().all()

            if not rows:
                return {"org_id": org_id, "synthesized": 0, "skipped": 0}

            outcome_records = [
                {
                    "outcome_type": r.outcome_type,
                    "impact_description": r.impact_description or "",
                    "goal_id": str(r.goal_id) if r.goal_id else None,
                }
                for r in rows
            ]

            candidates = synthesize_playbooks(outcome_records)
            synthesized = 0
            for pb in candidates:
                sig_hash = pb["signature_hash"]
                existing = (
                    await db.execute(
                        select(Playbook).where(
                            Playbook.organization_id == org_id,
                            Playbook.signature_hash == sig_hash,
                            Playbook.scope_type == PlaybookScopeType.ORGANIZATION,
                        )
                    )
                ).scalar_one_or_none()

                if existing:
                    # merge evidence counts
                    ev = dict(existing.evidence or {})
                    ev["outcome_count"] = pb["evidence"]["outcome_count"]
                    ev["valuable_count"] = pb["evidence"]["valuable_count"]
                    existing.success_rate = pb["success_rate"]
                    existing.steps = [{"action": s} for s in pb["steps"]]
                    existing.evidence = ev
                else:
                    db.add(
                        Playbook(
                            id=str(uuid4()),
                            organization_id=org_id,
                            scope_type=PlaybookScopeType.ORGANIZATION,
                            scope_id=None,
                            title=pb["title"],
                            problem_signature=pb["problem_signature"],
                            signature_hash=sig_hash,
                            steps=[{"action": s} for s in pb["steps"]],
                            constraints={"source": "auto_synthesis"},
                            success_rate=pb["success_rate"],
                            evidence=pb["evidence"],
                        )
                    )
                    synthesized += 1

            return {"org_id": org_id, "synthesized": synthesized, "candidates": len(candidates)}


async def _resolve_orgs(window_days: int) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    async with async_session_factory() as db:
        result = await db.execute(
            select(AutonomousGoalOutcome.organization_id)
            .where(AutonomousGoalOutcome.created_at >= cutoff)
            .distinct()
            .limit(200)
        )
        return [str(r[0]) for r in result.all()]


@celery_app.task(
    name="app.tasks.playbook_auto_synthesis_pipeline.playbook_auto_synthesis_task",
    bind=True,
    max_retries=3,
    retry_backoff=True,
)
def playbook_auto_synthesis_task(self, *, org_id: str | None = None, window_days: int = 30) -> dict:
    """Auto-synthesize playbooks from recurring high-success outcome patterns."""

    async def _run() -> dict:
        target_orgs = [org_id] if org_id else await _resolve_orgs(window_days)
        if not target_orgs:
            return {"orgs_processed": 0, "total_synthesized": 0}

        results = await asyncio.gather(*[_run_for_org(o, window_days) for o in target_orgs])
        total = sum(r.get("synthesized", 0) for r in results)
        logger.info("playbook_auto_synthesis: %d orgs, %d new playbooks", len(target_orgs), total)
        return {"orgs_processed": len(target_orgs), "total_synthesized": total, "details": list(results)}

    return run_async(_run())
