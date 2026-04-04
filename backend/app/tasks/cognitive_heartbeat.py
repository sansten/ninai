"""Cognitive OS heartbeat task.

Runs every 5 minutes. For each org with recent activity it:
  1. Scans for unresolved anomalies (high anomaly_score, no loop session run)
  2. Scans for stale active goals (active > 2h, no recent session)
  3. If either count exceeds threshold, autonomously creates a goal and
     fires a cognitive_loop_task — without any human prompt.
  4. Upserts SystemCognitionState so the per-org "what am I thinking about"
     record is always fresh.

This is what makes Ninai a Cognitive OS rather than a memory store: the
system has an ongoing internal life independent of external API calls.
"""

from __future__ import annotations

import asyncio
import uuid as _uuid
from datetime import datetime, timedelta, timezone

from celery.utils.log import get_task_logger
from sqlalchemy import cast, func, select, text
from sqlalchemy.dialects.postgresql import JSONB

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.models.autonomous_goal import AutonomousGoal
from app.models.cognitive_session import CognitiveSession
from app.models.memory import MemoryMetadata
from app.services.system_cognition_state import (
    CognitionStateUpdate,
    SystemCognitionStateService,
)
from app.services.cognitive_autonomy_control_service import get_cognitive_autonomy_control_service

logger = get_task_logger(__name__)

# Synthetic actor for heartbeat-initiated sessions
_HEARTBEAT_ACTOR = "00000000-0000-0000-0000-000000000001"
# Anomaly score threshold for autonomous session firing
_ANOMALY_THRESHOLD = 0.5
# Goal age before considered stale
_GOAL_STALE_HOURS = 2
# Lookback window for "recent anomalies"
_ANOMALY_LOOKBACK_MINUTES = 30
# Cognitive load threshold above which a session is fired
_LOAD_FIRE_THRESHOLD = 0.1


def _broker_available() -> bool:
    broker = celery_app.conf.broker_url
    return bool(broker) and not str(broker).startswith("memory://")


def _acquire_lock(org_id: str, ttl_seconds: int = 240) -> bool:
    """Acquire a Redis lock for this org's heartbeat. Returns True if acquired."""
    try:
        from app.core.redis import get_redis_client  # type: ignore[import]
        r = get_redis_client()
        key = f"heartbeat:lock:{org_id}"
        return bool(r.set(key, "1", nx=True, ex=ttl_seconds))
    except Exception:
        # If Redis is unavailable, proceed without lock (idempotency best-effort)
        return True


@celery_app.task(name="app.tasks.cognitive_heartbeat.cognitive_heartbeat_task")
def cognitive_heartbeat_task(*, org_id: str | None = None) -> dict:
    """Run the cognitive heartbeat for one org or all recently active orgs."""

    async def _resolve_active_orgs() -> list[str]:
        if org_id:
            return [org_id]
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        async with async_session_factory() as db:
            res = await db.execute(
                select(MemoryMetadata.organization_id)
                .where(MemoryMetadata.created_at >= cutoff)
                .distinct()
            )
            return [str(x) for x in res.scalars().all() if x]

    async def _count_unresolved_anomalies(db, target_org_id: str) -> int:
        """Count memories with high anomaly score and no loop session fired."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=_ANOMALY_LOOKBACK_MINUTES)
        try:
            # anomaly_score lives in extra_metadata JSONB on MemoryMetadata
            res = await db.execute(
                text(
                    """
                    SELECT COUNT(*) FROM memory_metadata
                    WHERE organization_id = :org_id
                      AND created_at >= :cutoff
                      AND (extra_metadata->>'anomaly_score')::float >= :threshold
                    """
                ),
                {
                    "org_id": target_org_id,
                    "cutoff": cutoff,
                    "threshold": _ANOMALY_THRESHOLD,
                },
            )
            return int(res.scalar() or 0)
        except Exception:
            return 0

    async def _count_stale_goals(db, target_org_id: str) -> int:
        """Count active goals older than GOAL_STALE_HOURS with no recent session."""
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=_GOAL_STALE_HOURS)
        try:
            res = await db.execute(
                select(func.count())
                .select_from(AutonomousGoal)
                .where(
                    AutonomousGoal.organization_id == target_org_id,
                    AutonomousGoal.status == "active",
                    AutonomousGoal.activated_at <= stale_cutoff,
                )
            )
            return int(res.scalar() or 0)
        except Exception:
            return 0

    async def _run_for_org(target_org_id: str) -> dict:
        enabled, reason = get_cognitive_autonomy_control_service().is_enabled(org_id=target_org_id)
        if not enabled:
            return {
                "org_id": target_org_id,
                "status": "skipped",
                "reason": reason or "cognitive_autonomy_disabled",
            }

        if not _acquire_lock(target_org_id):
            return {"org_id": target_org_id, "status": "skipped", "reason": "lock_held"}

        spawned_sessions: list[str] = []
        focus_parts: list[str] = []

        async with async_session_factory() as db:
            async with db.begin():
                await set_tenant_context(db, _HEARTBEAT_ACTOR, target_org_id)

                anomaly_count = await _count_unresolved_anomalies(db, target_org_id)
                stale_count = await _count_stale_goals(db, target_org_id)

                if anomaly_count:
                    focus_parts.append(f"{anomaly_count} unresolved anomaly/anomalies")
                if stale_count:
                    focus_parts.append(f"{stale_count} stale goal(s)")

                cognitive_load = min(
                    1.0, (anomaly_count + stale_count * 0.5) / 10.0
                )
                current_focus = (
                    "Monitoring: " + ", ".join(focus_parts)
                    if focus_parts
                    else "Idle — no active signals"
                )
                next_action: str | None = None

                if cognitive_load > _LOAD_FIRE_THRESHOLD:
                    title_parts = []
                    if anomaly_count:
                        title_parts.append(f"investigate {anomaly_count} anomaly/anomalies")
                    if stale_count:
                        title_parts.append(f"advance {stale_count} stale goal(s)")
                    goal_title = "Heartbeat: " + " and ".join(title_parts)
                    next_action = f"Autonomous session queued: {goal_title}"

                    if _broker_available():
                        session_id = str(_uuid.uuid4())
                        
                        # Create the CognitiveSession row with is_autonomous=True
                        cog_session = CognitiveSession(
                            id=session_id,
                            organization_id=target_org_id,
                            user_id=_HEARTBEAT_ACTOR,
                            status="running",
                            goal=goal_title,
                            goal_id=None,
                            agent_id=None,
                            context_snapshot={
                                "source": "heartbeat_autonomy",
                                "cognitive_load": cognitive_load,
                                "anomaly_count": anomaly_count,
                                "stale_goal_count": stale_count,
                            },
                            trace_id=None,
                            is_autonomous=True,  # Mark as autonomous session
                        )
                        db.add(cog_session)
                        await db.flush()
                        
                        spawned_sessions.append(session_id)
                        from app.tasks.cognitive_loop import cognitive_loop_task  # noqa: PLC0415
                        cognitive_loop_task.apply_async(
                            kwargs={
                                "org_id": target_org_id,
                                "session_id": session_id,
                                "initiator_user_id": _HEARTBEAT_ACTOR,
                                "justification": "cognitive_heartbeat",
                            },
                            queue="q.cognitive_loop",
                        )
                        logger.info(
                            "heartbeat fired autonomous session %s for org %s (load=%.2f)",
                            session_id,
                            target_org_id,
                            cognitive_load,
                        )

                # Upsert the org's live cognitive state
                svc = SystemCognitionStateService(db)
                await svc.upsert(
                    target_org_id,
                    CognitionStateUpdate(
                        current_focus=current_focus,
                        active_session_ids=spawned_sessions,
                        unresolved_anomalies_count=anomaly_count,
                        stale_goals_count=stale_count,
                        cognitive_load=cognitive_load,
                        next_scheduled_action=next_action,
                    ),
                )

        return {
            "org_id": target_org_id,
            "status": "ok",
            "cognitive_load": cognitive_load,
            "anomaly_count": anomaly_count,
            "stale_goal_count": stale_count,
            "sessions_spawned": len(spawned_sessions),
            "current_focus": current_focus,
        }

    async def _run() -> dict:
        active_orgs = await _resolve_active_orgs()
        if not active_orgs:
            return {"status": "no-op", "processed_orgs": 0, "results": []}

        results = []
        for oid in active_orgs:
            try:
                results.append(await _run_for_org(oid))
            except Exception as exc:
                logger.exception("heartbeat failed for org %s", oid)
                results.append({
                    "org_id": oid,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        return {
            "status": "ok",
            "processed_orgs": len(active_orgs),
            "results": results,
        }

    return asyncio.run(_run())
