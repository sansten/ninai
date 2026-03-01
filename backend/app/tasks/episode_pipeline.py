"""Episode routing and summarization pipeline tasks (PR2).

Implements:
- episode_router_task(memory_id): attach memory to matching episode or create new
- episode_summarizer_task(episode_id): refresh episode summary from recent events

All heavy work is async-capable and runs via Celery task wrappers.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from celery.utils.log import get_task_logger
from sqlalchemy import and_, desc, select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.models.episode import Episode, EpisodeStatus
from app.models.episode_event import EpisodeActorType, EpisodeEvent, EpisodeEventType
from app.models.memory import MemoryMetadata


logger = get_task_logger(__name__)

ROUTER_CANDIDATE_LIMIT = 50
ROUTER_MATCH_THRESHOLD = 0.42
ROUTER_OWNER_BOOST = 0.18
ROUTER_TIME_WEIGHT = 0.22
SUMMARY_MAX_EVENTS = 8


def _run_async(coro):
    """Run an async coroutine from a synchronous Celery task."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()

    return asyncio.run(coro)


def _text_tokens(*parts: Any) -> set[str]:
    text = " ".join(str(part or "") for part in parts).lower()
    return set(re.findall(r"[a-z0-9_\-]+", text))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def _time_proximity_score(ts_a: datetime | None, ts_b: datetime | None) -> float:
    if ts_a is None or ts_b is None:
        return 0.0
    delta_hours = abs((ts_a - ts_b).total_seconds()) / 3600.0
    # Smooth decay: near in time = high score, far away = low score
    return 1.0 / (1.0 + (delta_hours / 24.0))


def _episode_text(ep: Episode) -> set[str]:
    return _text_tokens(ep.title, ep.summary, ep.tags or [], ep.entities or {})


def _memory_text(mem: MemoryMetadata) -> set[str]:
    return _text_tokens(mem.title, mem.content_preview, mem.tags or [], mem.entities or {})


def _episode_score(memory: MemoryMetadata, episode: Episode) -> float:
    lexical = _jaccard_similarity(_memory_text(memory), _episode_text(episode))
    owner_boost = ROUTER_OWNER_BOOST if (memory.owner_id and memory.owner_id == episode.owner_user_id) else 0.0
    time_score = _time_proximity_score(memory.created_at, episode.last_event_at)
    return (0.60 * lexical) + owner_boost + (ROUTER_TIME_WEIGHT * time_score)


def _derive_episode_title(memory: MemoryMetadata) -> str:
    if memory.title:
        return memory.title
    preview = (memory.content_preview or "").strip()
    if not preview:
        return "Untitled Episode"
    return preview[:120]


async def route_memory_to_episode(
    *,
    org_id: str,
    memory_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    """Attach memory to best matching open episode or create a new episode."""

    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            memory = await db.get(MemoryMetadata, memory_id)
            if not memory or memory.organization_id != org_id:
                raise ValueError(f"Memory {memory_id} not found for organization {org_id}")

            candidates_stmt = (
                select(Episode)
                .where(
                    and_(
                        Episode.organization_id == org_id,
                        Episode.status == EpisodeStatus.OPEN,
                    )
                )
                .order_by(desc(Episode.last_event_at))
                .limit(ROUTER_CANDIDATE_LIMIT)
            )
            candidates = list((await db.execute(candidates_stmt)).scalars().all())

            best_episode = None
            best_score = -1.0
            for candidate in candidates:
                score = _episode_score(memory, candidate)
                if score > best_score:
                    best_score = score
                    best_episode = candidate

            attached_existing = bool(best_episode and best_score >= ROUTER_MATCH_THRESHOLD)

            if attached_existing:
                episode = best_episode
            else:
                now = memory.created_at or datetime.now(timezone.utc)
                episode = Episode(
                    id=str(uuid4()),
                    organization_id=org_id,
                    scope_type=memory.scope,
                    scope_id=memory.scope_id,
                    owner_user_id=memory.owner_id,
                    episode_type="memory_case",
                    status=EpisodeStatus.OPEN,
                    title=_derive_episode_title(memory),
                    summary=None,
                    started_at=now,
                    last_event_at=now,
                    resolved_at=None,
                    tags=memory.tags,
                    entities=memory.entities,
                )
                db.add(episode)
                await db.flush()

            event_ts = memory.created_at or datetime.now(timezone.utc)
            event = EpisodeEvent(
                id=str(uuid4()),
                organization_id=org_id,
                episode_id=episode.id,
                memory_id=memory.id,
                event_type=EpisodeEventType.USER_REPORT,
                event_ts=event_ts,
                actor_type=EpisodeActorType.USER,
                actor_id=memory.owner_id,
                content=memory.content_preview,
                payload={"source": "episode_router_task", "memory_title": memory.title},
            )
            db.add(event)

            episode.last_event_at = max(episode.last_event_at or event_ts, event_ts)
            await db.flush()

            return {
                "episode_id": episode.id,
                "memory_id": memory.id,
                "attached_existing": attached_existing,
                "match_score": round(best_score if best_score > 0 else 0.0, 4),
                "event_id": event.id,
            }


async def summarize_episode(
    *,
    org_id: str,
    episode_id: str,
    actor_user_id: str,
) -> dict[str, Any]:
    """Generate/update a lightweight summary from recent episode events."""

    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            episode = await db.get(Episode, episode_id)
            if not episode or episode.organization_id != org_id:
                raise ValueError(f"Episode {episode_id} not found for organization {org_id}")

            events_stmt = (
                select(EpisodeEvent)
                .where(
                    and_(
                        EpisodeEvent.organization_id == org_id,
                        EpisodeEvent.episode_id == episode_id,
                    )
                )
                .order_by(desc(EpisodeEvent.event_ts))
                .limit(SUMMARY_MAX_EVENTS)
            )
            events = list((await db.execute(events_stmt)).scalars().all())

            if not events:
                episode.summary = "No activity yet."
                await db.flush()
                return {"episode_id": episode_id, "summary": episode.summary, "event_count": 0}

            ordered = list(reversed(events))
            snippets = []
            for ev in ordered:
                text = (ev.content or "").strip()
                if text:
                    snippets.append(text[:140])

            status_text = f"status={episode.status}"
            title_text = episode.title or "Untitled Episode"
            key_points = "; ".join(snippets[:3]) if snippets else "Events recorded"
            summary = f"{title_text} ({status_text}). Recent timeline: {key_points}."

            episode.summary = summary[:1200]
            await db.flush()

            return {
                "episode_id": episode_id,
                "summary": episode.summary,
                "event_count": len(events),
            }


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def episode_router_task(
    self,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    """Route memory into an episode and trigger summary refresh."""

    if storage != "long_term":
        return {
            "status": "skipped",
            "reason": "memory_not_long_term",
            "org_id": org_id,
            "memory_id": memory_id,
            "trace_id": trace_id,
        }

    actor_user_id = initiator_user_id or "00000000-0000-0000-0000-000000000001"
    route_result = _run_async(route_memory_to_episode(org_id=org_id, memory_id=memory_id, actor_user_id=actor_user_id))
    summary_result = _run_async(
        summarize_episode(org_id=org_id, episode_id=route_result["episode_id"], actor_user_id=actor_user_id)
    )

    return {
        "status": "ok",
        "org_id": org_id,
        "memory_id": memory_id,
        "trace_id": trace_id,
        "route": route_result,
        "summary": summary_result,
    }


@celery_app.task(
    bind=True,
    max_retries=5,
    autoretry_for=(Exception,),
    dont_autoretry_for=(ValueError,),
    retry_backoff=True,
)
def episode_summarizer_task(
    self,
    org_id: str,
    episode_id: str,
    initiator_user_id: str | None = None,
):
    """Refresh summary for an existing episode."""

    actor_user_id = initiator_user_id or "00000000-0000-0000-0000-000000000001"
    summary_result = _run_async(summarize_episode(org_id=org_id, episode_id=episode_id, actor_user_id=actor_user_id))
    return {
        "status": "ok",
        "org_id": org_id,
        "episode_id": episode_id,
        "summary": summary_result,
    }


def enqueue_episode_pipeline(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    """Enqueue episode routing when Celery broker is enabled."""

    broker = celery_app.conf.broker_url
    if not broker or str(broker).startswith("memory://"):
        return None

    return episode_router_task.si(
        org_id=org_id,
        memory_id=memory_id,
        initiator_user_id=initiator_user_id,
        trace_id=trace_id,
        storage=storage,
    ).apply_async()
