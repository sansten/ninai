from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode, EpisodeStatus
from app.models.memory import MemoryMetadata
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership


class EpisodeConvergenceService:
    """Projects case episodes into the durable memory-episode substrate."""

    PROJECTION_SOURCE = "case_episode_projection"

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def sync_case_episode_projection(
        self,
        *,
        episode: Episode,
        memory: MemoryMetadata | None = None,
        actor_user_id: str | None = None,
        event_ts: datetime | None = None,
        boundary_reason: str = "case_episode_projection",
    ) -> MemoryEpisode:
        owner_id = self._resolve_owner_id(
            episode=episode,
            memory=memory,
            actor_user_id=actor_user_id,
        )

        projected = await self._get_or_create_projection(
            episode=episode,
            owner_id=owner_id,
            boundary_reason=boundary_reason,
        )

        projected.owner_id = owner_id
        projected.scope = getattr(episode.scope_type, "value", episode.scope_type) or "personal"
        projected.scope_id = episode.scope_id
        projected.title = episode.title
        projected.narrative_summary = episode.summary
        projected.boundary_start = self._min_dt(
            projected.boundary_start,
            episode.started_at,
            memory.created_at if memory else None,
            event_ts,
        )
        projected.boundary_end = self._max_dt(
            projected.boundary_end,
            episode.last_event_at,
            episode.resolved_at,
            memory.created_at if memory else None,
            event_ts,
        )
        projected.boundary_reason = boundary_reason or projected.boundary_reason or "case_episode_projection"
        projected.boundary_confidence = 1.0
        projected.status = self._map_status(episode.status)
        projected.extra_metadata = self._build_projection_metadata(
            episode=episode,
            memory=memory,
            existing=projected.extra_metadata,
        )

        await self.session.flush()

        if memory is not None:
            await self._ensure_membership(
                episode_id=projected.id,
                memory=memory,
            )

        projected.message_count = await self._membership_count(projected.id)
        await self.session.flush()
        return projected

    async def _get_or_create_projection(
        self,
        *,
        episode: Episode,
        owner_id: str,
        boundary_reason: str,
    ) -> MemoryEpisode:
        projected = (
            await self.session.execute(
                select(MemoryEpisode).where(
                    MemoryEpisode.id == episode.id,
                    MemoryEpisode.organization_id == episode.organization_id,
                )
            )
        ).scalar_one_or_none()
        if projected is not None:
            return projected

        projected = MemoryEpisode(
            id=episode.id,
            organization_id=episode.organization_id,
            owner_id=owner_id,
            scope=getattr(episode.scope_type, "value", episode.scope_type) or "personal",
            scope_id=episode.scope_id,
            title=episode.title,
            narrative_summary=episode.summary,
            boundary_start=episode.started_at,
            boundary_end=episode.last_event_at or episode.resolved_at,
            message_count=0,
            boundary_reason=boundary_reason or "case_episode_projection",
            boundary_confidence=1.0,
            status=self._map_status(episode.status),
            created_by="episode_projection",
            extra_metadata=self._build_projection_metadata(episode=episode, memory=None, existing=None),
        )
        self.session.add(projected)
        await self.session.flush()
        return projected

    async def _ensure_membership(
        self,
        *,
        episode_id: str,
        memory: MemoryMetadata,
    ) -> None:
        existing = (
            await self.session.execute(
                select(MemoryEpisodeMembership).where(
                    MemoryEpisodeMembership.organization_id == memory.organization_id,
                    MemoryEpisodeMembership.episode_id == episode_id,
                    MemoryEpisodeMembership.memory_id == memory.id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return

        next_position = await self._next_position(episode_id)
        self.session.add(
            MemoryEpisodeMembership(
                id=str(uuid4()),
                organization_id=memory.organization_id,
                memory_id=memory.id,
                episode_id=episode_id,
                position=next_position,
                created_by="episode_projection",
            )
        )
        await self.session.flush()

    async def _next_position(self, episode_id: str) -> int:
        stmt = select(func.max(MemoryEpisodeMembership.position)).where(
            MemoryEpisodeMembership.episode_id == episode_id,
        )
        current_max = await self.session.scalar(stmt)
        return int(current_max or -1) + 1

    async def _membership_count(self, episode_id: str) -> int:
        stmt = select(func.count()).select_from(MemoryEpisodeMembership).where(
            MemoryEpisodeMembership.episode_id == episode_id,
        )
        total = await self.session.scalar(stmt)
        return int(total or 0)

    def _build_projection_metadata(
        self,
        *,
        episode: Episode,
        memory: MemoryMetadata | None,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(existing or {})
        merged_entities = dict(metadata.get("entities") or {})
        merged_entities.update(dict(episode.entities or {}))
        if memory is not None:
            merged_entities.update(dict(memory.entities or {}))

        merged_tags: list[str] = []
        tag_sources = list(metadata.get("tags") or []) + list(episode.tags or [])
        if memory is not None:
            tag_sources += list(memory.tags or [])
        for tag in tag_sources:
            if tag and tag not in merged_tags:
                merged_tags.append(tag)

        metadata.update(
            {
                "projection_source": self.PROJECTION_SOURCE,
                "source_case_episode_id": str(episode.id),
                "case_episode_type": episode.episode_type,
                "case_episode_status": getattr(episode.status, "value", episode.status),
                "tags": merged_tags,
                "entities": merged_entities,
                "projected_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if memory is not None:
            metadata["latest_memory_id"] = str(memory.id)
            metadata["latest_memory_created_at"] = (
                memory.created_at.isoformat() if memory.created_at else None
            )
        return metadata

    @staticmethod
    def _resolve_owner_id(
        *,
        episode: Episode,
        memory: MemoryMetadata | None,
        actor_user_id: str | None,
    ) -> str:
        owner_id = episode.owner_user_id or (memory.owner_id if memory is not None else None) or actor_user_id
        if not owner_id:
            raise ValueError(f"Episode {episode.id} has no owner to project into memory_episodes")
        return str(owner_id)

    @staticmethod
    def _map_status(status: EpisodeStatus | str | None) -> str:
        value = getattr(status, "value", status)
        return "open" if value == EpisodeStatus.OPEN.value else "closed"

    @staticmethod
    def _min_dt(*values: datetime | None) -> datetime | None:
        filtered = [value for value in values if value is not None]
        return min(filtered) if filtered else None

    @staticmethod
    def _max_dt(*values: datetime | None) -> datetime | None:
        filtered = [value for value in values if value is not None]
        return max(filtered) if filtered else None
