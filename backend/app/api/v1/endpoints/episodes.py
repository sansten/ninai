"""
Episode Endpoints (PR1: Advanced Memory Features)
==================================================

CRUD operations for episodes (case continuity), episode events, and episode links.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.tenant_context import TenantContext, get_tenant_context
from app.schemas.episode import (
    EpisodeCreate,
    EpisodeEventCreate,
    EpisodeEventListResponse,
    EpisodeEventResponse,
    EpisodeLinkCreate,
    EpisodeLinkListResponse,
    EpisodeLinkResponse,
    EpisodeListResponse,
    EpisodeResponse,
    EpisodeUpdate,
)
from app.services.episode_service import EpisodeService

router = APIRouter()


def get_episode_service(
    tenant: TenantContext = Depends(get_tenant_context),
    db: AsyncSession = Depends(get_db),
) -> EpisodeService:
    """Dependency to get configured EpisodeService."""
    return EpisodeService(
        session=db,
        user_id=tenant.user_id,
        org_id=tenant.org_id,
    )


# ═════════════════════════════════════════════════════════════════════════
# Episode CRUD
# ═════════════════════════════════════════════════════════════════════════


@router.post(
    "",
    response_model=EpisodeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create episode",
    description="Create a new episode for case/ticket/thread tracking",
)
async def create_episode(
    data: EpisodeCreate,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeResponse:
    """Create a new episode."""
    episode = await service.create_episode(data)
    await service.session.commit()
    return EpisodeResponse.model_validate(episode)


@router.get(
    "/{episode_id}",
    response_model=EpisodeResponse,
    summary="Get episode",
    description="Get episode by ID (RLS enforced)",
)
async def get_episode(
    episode_id: str,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeResponse:
    """Get episode by ID."""
    episode = await service.get_episode(episode_id)
    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found or access denied",
        )
    return EpisodeResponse.model_validate(episode)


@router.get(
    "",
    response_model=EpisodeListResponse,
    summary="List episodes",
    description="List episodes with filters (RLS enforced)",
)
async def list_episodes(
    status_filter: Optional[str] = Query(None, alias="status"),
    episode_type: Optional[str] = Query(None),
    owner_user_id: Optional[str] = Query(None),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeListResponse:
    """List episodes with filters."""
    tags_list = tags.split(",") if tags else None

    episodes, total = await service.list_episodes(
        status=status_filter,
        episode_type=episode_type,
        owner_user_id=owner_user_id,
        tags=tags_list,
        page=page,
        page_size=page_size,
    )

    return EpisodeListResponse(
        episodes=[EpisodeResponse.model_validate(e) for e in episodes],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.patch(
    "/{episode_id}",
    response_model=EpisodeResponse,
    summary="Update episode",
    description="Update episode (RLS enforced)",
)
async def update_episode(
    episode_id: str,
    data: EpisodeUpdate,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeResponse:
    """Update episode."""
    episode = await service.update_episode(episode_id, data)
    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found or access denied",
        )
    await service.session.commit()
    return EpisodeResponse.model_validate(episode)


@router.post(
    "/{episode_id}/resolve",
    response_model=EpisodeResponse,
    summary="Resolve episode",
    description="Mark episode as resolved (RLS enforced)",
)
async def resolve_episode(
    episode_id: str,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeResponse:
    """Mark episode as resolved."""
    episode = await service.resolve_episode(episode_id)
    if not episode:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found or access denied",
        )
    await service.session.commit()
    return EpisodeResponse.model_validate(episode)


# ═════════════════════════════════════════════════════════════════════════
# Episode Event CRUD
# ═════════════════════════════════════════════════════════════════════════


@router.post(
    "/events",
    response_model=EpisodeEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create episode event",
    description="Create a new event in an episode timeline",
)
async def create_episode_event(
    data: EpisodeEventCreate,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeEventResponse:
    """Create a new episode event."""
    event = await service.create_episode_event(data)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Episode not found or access denied",
        )
    await service.session.commit()
    return EpisodeEventResponse.model_validate(event)


@router.get(
    "/{episode_id}/events",
    response_model=EpisodeEventListResponse,
    summary="List episode events",
    description="List events for an episode (RLS enforced)",
)
async def list_episode_events(
    episode_id: str,
    limit: int = Query(100, ge=1, le=1000),
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeEventListResponse:
    """List events for an episode."""
    events = await service.list_episode_events(episode_id, limit=limit)
    return EpisodeEventListResponse(
        events=[EpisodeEventResponse.model_validate(e) for e in events],
        total=len(events),
    )


# ═════════════════════════════════════════════════════════════════════════
# Episode Link CRUD
# ═════════════════════════════════════════════════════════════════════════


@router.post(
    "/links",
    response_model=EpisodeLinkResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create episode link",
    description="Create a relationship between two episodes",
)
async def create_episode_link(
    data: EpisodeLinkCreate,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeLinkResponse:
    """Create a link between two episodes."""
    link = await service.create_episode_link(data)
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or both episodes not found or access denied",
        )
    await service.session.commit()
    return EpisodeLinkResponse.model_validate(link)


@router.get(
    "/{episode_id}/links",
    response_model=EpisodeLinkListResponse,
    summary="List episode links",
    description="List all links connected to an episode (RLS enforced)",
)
async def list_episode_links(
    episode_id: str,
    service: EpisodeService = Depends(get_episode_service),
) -> EpisodeLinkListResponse:
    """List all links connected to an episode."""
    links = await service.list_episode_links(episode_id)
    return EpisodeLinkListResponse(
        links=[EpisodeLinkResponse.model_validate(link) for link in links],
        total=len(links),
    )
