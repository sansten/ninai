"""
Episode Service (PR1: Advanced Memory Features)
================================================

Service for episode case continuity operations.
Handles episodes, episode events, and episode links with RLS enforcement.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode, EpisodeStatus
from app.models.episode_event import EpisodeEvent
from app.models.episode_link import EpisodeLink
from app.schemas.episode import (
    EpisodeCreate,
    EpisodeEventCreate,
    EpisodeLinkCreate,
    EpisodeUpdate,
)
from app.services.audit_service import AuditService


class EpisodeService:
    """
    Episode operations service.

    Handles all episode CRUD operations with:
    - RLS enforcement (organization_id filtering)
    - Audit logging for all operations
    - Timeline tracking (last_event_at updates)
    - Idempotent operations
    """

    def __init__(
        self,
        session: AsyncSession,
        user_id: str,
        org_id: str,
    ):
        """
        Initialize episode service.

        Args:
            session: Database session with tenant context set
            user_id: Current user's UUID
            org_id: Current organization's UUID
        """
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.audit_service = AuditService(session)

    # ═════════════════════════════════════════════════════════════════════
    # Episode CRUD
    # ═════════════════════════════════════════════════════════════════════

    async def create_episode(
        self,
        data: EpisodeCreate,
    ) -> Episode:
        """
        Create a new episode.

        Args:
            data: Episode creation data

        Returns:
            Created episode
        """
        now = datetime.now(timezone.utc)

        episode = Episode(
            id=str(uuid4()),
            organization_id=self.org_id,
            scope_type=data.scope_type,
            scope_id=data.scope_id,
            owner_user_id=data.owner_user_id or self.user_id,
            episode_type=data.episode_type,
            status=EpisodeStatus.OPEN,
            title=data.title,
            summary=None,  # Will be populated by async summarizer task
            started_at=now,
            last_event_at=now,
            resolved_at=None,
            tags=data.tags,
            entities=data.entities,
        )

        self.session.add(episode)
        await self.session.flush()

        # Audit log
        await self.audit_service.log_event(
            event_type="episode_created",
            actor_id=self.user_id,
            organization_id=self.org_id,
            resource_type="episode",
            resource_id=episode.id,
            details={
                "episode_type": episode.episode_type,
                "title": episode.title,
                "scope_type": episode.scope_type,
            },
        )

        return episode

    async def get_episode(
        self,
        episode_id: str,
    ) -> Optional[Episode]:
        """
        Get episode by ID (RLS enforced).

        Args:
            episode_id: Episode UUID

        Returns:
            Episode if found and accessible, None otherwise
        """
        stmt = select(Episode).where(
            and_(
                Episode.id == episode_id,
                Episode.organization_id == self.org_id,
            )
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_episodes(
        self,
        status: Optional[str] = None,
        episode_type: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        started_after: Optional[datetime] = None,
        started_before: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Episode], int]:
        """
        List episodes with filters (RLS enforced).

        Args:
            status: Filter by status
            episode_type: Filter by type
            owner_user_id: Filter by owner
            tags: Filter by tags (ANY match)
            started_after: Filter by start date (after)
            started_before: Filter by start date (before)
            page: Page number (1-indexed)
            page_size: Items per page

        Returns:
            Tuple of (episodes, total_count)
        """
        conditions = [Episode.organization_id == self.org_id]

        if status:
            conditions.append(Episode.status == status)
        if episode_type:
            conditions.append(Episode.episode_type == episode_type)
        if owner_user_id:
            conditions.append(Episode.owner_user_id == owner_user_id)
        if tags:
            conditions.append(Episode.tags.overlap(tags))
        if started_after:
            conditions.append(Episode.started_at >= started_after)
        if started_before:
            conditions.append(Episode.started_at <= started_before)

        # Count query
        count_stmt = select(func.count()).select_from(Episode).where(and_(*conditions))
        total = await self.session.scalar(count_stmt)

        # Data query
        stmt = (
            select(Episode)
            .where(and_(*conditions))
            .order_by(desc(Episode.last_event_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        result = await self.session.execute(stmt)
        episodes = list(result.scalars().all())

        return episodes, total

    async def update_episode(
        self,
        episode_id: str,
        data: EpisodeUpdate,
    ) -> Optional[Episode]:
        """
        Update episode (RLS enforced).

        Args:
            episode_id: Episode UUID
            data: Update data

        Returns:
            Updated episode if found and accessible, None otherwise
        """
        episode = await self.get_episode(episode_id)
        if not episode:
            return None

        # Apply updates
        update_dict = data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(episode, key, value)

        await self.session.flush()

        # Audit log
        await self.audit_service.log_event(
            event_type="episode_updated",
            actor_id=self.user_id,
            organization_id=self.org_id,
            resource_type="episode",
            resource_id=episode.id,
            details={"updates": update_dict},
        )

        return episode

    async def resolve_episode(
        self,
        episode_id: str,
    ) -> Optional[Episode]:
        """
        Mark episode as resolved (RLS enforced).

        Args:
            episode_id: Episode UUID

        Returns:
            Updated episode if found and accessible, None otherwise
        """
        episode = await self.get_episode(episode_id)
        if not episode:
            return None

        episode.status = EpisodeStatus.RESOLVED
        episode.resolved_at = datetime.now(timezone.utc)
        await self.session.flush()

        # Audit log
        await self.audit_service.log_event(
            event_type="episode_resolved",
            actor_id=self.user_id,
            organization_id=self.org_id,
            resource_type="episode",
            resource_id=episode.id,
            details={"resolved_at": episode.resolved_at.isoformat()},
        )

        return episode

    # ═════════════════════════════════════════════════════════════════════
    # Episode Event CRUD
    # ═════════════════════════════════════════════════════════════════════

    async def create_episode_event(
        self,
        data: EpisodeEventCreate,
    ) -> Optional[EpisodeEvent]:
        """
        Create an episode event.

        Also updates episode.last_event_at for proper timeline ordering.

        Args:
            data: Event creation data

        Returns:
            Created event if episode accessible, None otherwise
        """
        # Verify episode access
        episode = await self.get_episode(data.episode_id)
        if not episode:
            return None

        now = datetime.now(timezone.utc)
        event_ts = data.event_ts or now

        event = EpisodeEvent(
            id=str(uuid4()),
            organization_id=self.org_id,
            episode_id=data.episode_id,
            memory_id=data.memory_id,
            event_type=data.event_type,
            event_ts=event_ts,
            actor_type=data.actor_type,
            actor_id=data.actor_id or self.user_id,
            content=data.content,
            payload=data.payload,
        )

        self.session.add(event)

        # Update episode.last_event_at
        episode.last_event_at = event_ts
        await self.session.flush()

        # Audit log
        await self.audit_service.log_event(
            event_type="episode_event_created",
            actor_id=self.user_id,
            organization_id=self.org_id,
            resource_type="episode_event",
            resource_id=event.id,
            details={
                "episode_id": data.episode_id,
                "event_type": data.event_type,
            },
        )

        return event

    async def list_episode_events(
        self,
        episode_id: str,
        limit: int = 100,
    ) -> list[EpisodeEvent]:
        """
        List events for an episode (RLS enforced).

        Args:
            episode_id: Episode UUID
            limit: Max number of events to return

        Returns:
            List of events (empty if episode not accessible)
        """
        # Verify episode access
        episode = await self.get_episode(episode_id)
        if not episode:
            return []

        stmt = (
            select(EpisodeEvent)
            .where(
                and_(
                    EpisodeEvent.episode_id == episode_id,
                    EpisodeEvent.organization_id == self.org_id,
                )
            )
            .order_by(EpisodeEvent.event_ts)
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ═════════════════════════════════════════════════════════════════════
    # Episode Link CRUD
    # ═════════════════════════════════════════════════════════════════════

    async def create_episode_link(
        self,
        data: EpisodeLinkCreate,
    ) -> Optional[EpisodeLink]:
        """
        Create a link between two episodes (RLS enforced).

        Args:
            data: Link creation data

        Returns:
            Created link if both episodes accessible, None otherwise
        """
        # Verify both episodes exist and are accessible
        from_episode = await self.get_episode(data.from_episode_id)
        to_episode = await self.get_episode(data.to_episode_id)

        if not from_episode or not to_episode:
            return None

        # Check if link already exists
        existing_stmt = select(EpisodeLink).where(
            and_(
                EpisodeLink.from_episode_id == data.from_episode_id,
                EpisodeLink.to_episode_id == data.to_episode_id,
                EpisodeLink.relation == data.relation,
                EpisodeLink.organization_id == self.org_id,
            )
        )
        existing = await self.session.scalar(existing_stmt)
        if existing:
            return existing  # Idempotent

        link = EpisodeLink(
            id=str(uuid4()),
            organization_id=self.org_id,
            from_episode_id=data.from_episode_id,
            to_episode_id=data.to_episode_id,
            relation=data.relation,
            confidence=data.confidence,
            evidence=data.evidence,
        )

        self.session.add(link)
        await self.session.flush()

        # Audit log
        await self.audit_service.log_event(
            event_type="episode_link_created",
            actor_id=self.user_id,
            organization_id=self.org_id,
            resource_type="episode_link",
            resource_id=link.id,
            details={
                "from_episode_id": data.from_episode_id,
                "to_episode_id": data.to_episode_id,
                "relation": data.relation,
            },
        )

        return link

    async def list_episode_links(
        self,
        episode_id: str,
    ) -> list[EpisodeLink]:
        """
        List all links connected to an episode (RLS enforced).

        Args:
            episode_id: Episode UUID

        Returns:
            List of links (empty if episode not accessible)
        """
        # Verify episode access
        episode = await self.get_episode(episode_id)
        if not episode:
            return []

        stmt = select(EpisodeLink).where(
            and_(
                or_(
                    EpisodeLink.from_episode_id == episode_id,
                    EpisodeLink.to_episode_id == episode_id,
                ),
                EpisodeLink.organization_id == self.org_id,
            )
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())
