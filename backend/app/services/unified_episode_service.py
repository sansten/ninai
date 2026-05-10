from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode
from app.models.episode_event import EpisodeEvent
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership


@dataclass
class UnifiedEpisodeView:
    episode_id: str
    source: str
    title: str | None
    summary: str | None
    status: str
    started_at: str | None
    ended_at: str | None
    message_count: int | None
    scope: str | None
    scope_id: str | None
    topic_id: str | None = None
    tags: list[str] = field(default_factory=list)
    entities: dict[str, Any] = field(default_factory=dict)
    linked_memory_ids: list[str] = field(default_factory=list)
    sort_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("sort_at", None)
        return data


class UnifiedEpisodeService:
    """Normalizes both episode systems behind one read interface."""

    def __init__(self, session: AsyncSession, *, org_id: str):
        self.session = session
        self.org_id = org_id

    async def list_for_memory_ids(
        self,
        memory_ids: list[str],
        *,
        limit_per_memory: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        if not memory_ids:
            return {}

        index: dict[str, list[UnifiedEpisodeView]] = {str(mid): [] for mid in memory_ids}
        await self._load_case_episodes(memory_ids, index)
        await self._load_memory_episodes(memory_ids, index)

        result: dict[str, list[dict[str, Any]]] = {}
        for memory_id, items in index.items():
            ordered = sorted(
                items,
                key=lambda item: item.sort_at or datetime.min,
                reverse=True,
            )
            result[memory_id] = [item.as_dict() for item in ordered[: max(1, int(limit_per_memory or 1))]]
        return result

    async def _load_case_episodes(
        self,
        memory_ids: list[str],
        index: dict[str, list[UnifiedEpisodeView]],
    ) -> None:
        event_stmt = select(
            EpisodeEvent.memory_id,
            EpisodeEvent.episode_id,
            EpisodeEvent.event_ts,
        ).where(
            EpisodeEvent.organization_id == self.org_id,
            EpisodeEvent.memory_id.in_(memory_ids),
        )
        event_rows = (await self.session.execute(event_stmt)).all()
        if not event_rows:
            return

        episode_ids = sorted({str(row.episode_id) for row in event_rows if row.episode_id})
        if not episode_ids:
            return

        episodes_stmt = select(Episode).where(
            Episode.organization_id == self.org_id,
            Episode.id.in_(episode_ids),
        )
        episodes = {
            str(row.id): row
            for row in (await self.session.execute(episodes_stmt)).scalars().all()
        }

        by_pair: dict[tuple[str, str], datetime | None] = {}
        for memory_id, episode_id, event_ts in event_rows:
            key = (str(memory_id), str(episode_id))
            previous = by_pair.get(key)
            if previous is None or (event_ts and event_ts > previous):
                by_pair[key] = event_ts

        for (memory_id, episode_id), event_ts in by_pair.items():
            episode = episodes.get(episode_id)
            if episode is None:
                continue
            index.setdefault(memory_id, []).append(
                UnifiedEpisodeView(
                    episode_id=episode_id,
                    source="case_episode",
                    title=episode.title,
                    summary=episode.summary,
                    status=getattr(episode.status, "value", episode.status),
                    started_at=episode.started_at.isoformat() if episode.started_at else None,
                    ended_at=episode.resolved_at.isoformat() if episode.resolved_at else None,
                    message_count=None,
                    scope=getattr(episode.scope_type, "value", episode.scope_type) if episode.scope_type else None,
                    scope_id=episode.scope_id,
                    tags=list(episode.tags or []),
                    entities=dict(episode.entities or {}),
                    linked_memory_ids=[memory_id],
                    sort_at=event_ts or episode.last_event_at or episode.started_at,
                )
            )

    async def _load_memory_episodes(
        self,
        memory_ids: list[str],
        index: dict[str, list[UnifiedEpisodeView]],
    ) -> None:
        membership_stmt = select(
            MemoryEpisodeMembership.memory_id,
            MemoryEpisodeMembership.episode_id,
        ).where(
            MemoryEpisodeMembership.organization_id == self.org_id,
            MemoryEpisodeMembership.memory_id.in_(memory_ids),
        )
        membership_rows = (await self.session.execute(membership_stmt)).all()
        if not membership_rows:
            return

        episode_ids = sorted({str(row.episode_id) for row in membership_rows if row.episode_id})
        if not episode_ids:
            return

        episodes_stmt = select(MemoryEpisode).where(
            MemoryEpisode.organization_id == self.org_id,
            MemoryEpisode.id.in_(episode_ids),
        )
        episodes = {
            str(row.id): row
            for row in (await self.session.execute(episodes_stmt)).scalars().all()
        }

        memory_links: dict[str, list[str]] = {}
        for memory_id, episode_id in membership_rows:
            memory_links.setdefault(str(episode_id), []).append(str(memory_id))

        for memory_id, episode_id in membership_rows:
            episode = episodes.get(str(episode_id))
            if episode is None:
                continue
            linked_memory_ids = list(dict.fromkeys(memory_links.get(str(episode_id), [str(memory_id)])))
            metadata = episode.extra_metadata or {}
            if not isinstance(metadata, dict):
                metadata = {}
            entities = metadata.get("entities")
            if isinstance(entities, dict) and entities:
                normalized_entities = entities
            else:
                normalized_entities = dict(metadata)
            tags = metadata.get("tags")
            if not isinstance(tags, list):
                tags = []
            index.setdefault(str(memory_id), []).append(
                UnifiedEpisodeView(
                    episode_id=str(episode_id),
                    source="memory_episode",
                    title=episode.title,
                    summary=episode.narrative_summary,
                    status=str(episode.status or ""),
                    started_at=episode.boundary_start.isoformat() if episode.boundary_start else None,
                    ended_at=episode.boundary_end.isoformat() if episode.boundary_end else None,
                    message_count=int(episode.message_count or 0),
                    scope=episode.scope,
                    scope_id=episode.scope_id,
                    topic_id=episode.topic_id,
                    tags=list(tags),
                    entities=normalized_entities,
                    linked_memory_ids=linked_memory_ids,
                    sort_at=episode.boundary_end or episode.updated_at or episode.created_at,
                )
            )
