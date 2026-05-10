from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.requester_context import RequesterContext
from app.schemas.memory import MemoryCreate
from app.services.identity_policy_service import ResolvedActorContext
from app.services.memory_service import MemoryService
from app.services.usage_service import UsageService
from app.tasks.embed_task import enqueue_embed_and_index
from app.tasks.episode_pipeline import enqueue_episode_pipeline
from app.tasks.fact_pipeline import enqueue_fact_pipeline
from app.tasks.memory_pipeline import enqueue_memory_pipeline


@dataclass
class CognitiveIngestionResult:
    memory: Any
    storage: str
    pipelines_enqueued: list[str] = field(default_factory=list)


class CognitiveIngestionService:
    """Canonical durable ingestion path shared by memory and gateway writes."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        user_id: str,
        org_id: str,
        clearance_level: int = 0,
        roles_string: str = "",
        memory_service: MemoryService | None = None,
    ) -> None:
        self.session = session
        self.user_id = user_id
        self.org_id = org_id
        self.clearance_level = clearance_level
        self.roles_string = roles_string
        self.memory_service = memory_service or MemoryService(
            session=session,
            user_id=user_id,
            org_id=org_id,
            clearance_level=clearance_level,
        )

    @staticmethod
    def _requester_metadata(
        metadata: dict[str, Any] | None,
        requester: RequesterContext | None,
    ) -> dict[str, Any]:
        merged = dict(metadata or {})
        if requester is None:
            return merged

        merged.setdefault("_requester_job_role", requester.job_role)
        merged.setdefault("_requester_timezone", requester.timezone)
        merged.setdefault("_requester_urgency", requester.urgency_signal)
        if requester.location:
            merged.setdefault("_requester_location", requester.location)
        if requester.dominant_domains:
            merged.setdefault("_requester_domains", list(requester.dominant_domains[:6]))
        return merged

    @staticmethod
    def build_gateway_memory_create(
        *,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        requester: RequesterContext | None = None,
        context_id: str | None = None,
    ) -> MemoryCreate:
        body = dict(payload or {})
        extra_metadata = CognitiveIngestionService._requester_metadata(metadata, requester)
        if context_id:
            extra_metadata.setdefault("gateway_context_id", context_id)

        source_id = body.get("source_id") or context_id
        return MemoryCreate(
            content=content,
            title=title or None,
            scope=body.get("scope", "personal"),
            scope_id=body.get("scope_id"),
            memory_type=body.get("memory_type", "long_term"),
            classification=body.get("classification", "internal"),
            required_clearance=body.get("required_clearance"),
            tags=tags or body.get("tags"),
            entities=body.get("entities"),
            extra_metadata=extra_metadata or None,
            source_type=body.get("source_type") or "cognitive_gateway",
            source_id=source_id,
            anonymous=bool(body.get("anonymous", False)),
            occurred_at=body.get("occurred_at"),
            retention_days=body.get("retention_days"),
            ttl=body.get("ttl"),
        )

    @staticmethod
    def requester_to_actor_context(
        requester: RequesterContext | None,
    ) -> Optional[ResolvedActorContext]:
        if requester is None:
            return None

        normalized_roles = [str(role).strip().lower() for role in (requester.roles or []) if str(role).strip()]
        actor_type = "bot" if any("bot" in role or "agent" in role for role in normalized_roles) else "employee"
        role = requester.job_role or (normalized_roles[0] if normalized_roles else None)
        confidence = float(requester.profile_confidence or 0.0)

        return ResolvedActorContext(
            actor_id=requester.user_id,
            actor_type=actor_type,
            role=role,
            department=None,
            display_name=None,
            mode_applied="full",
            identity_confidence=max(confidence, 0.25),
            mandate_was_active=False,
        )

    async def ingest_memory(
        self,
        *,
        data: MemoryCreate,
        request_id: str | None = None,
        actor_ctx: ResolvedActorContext | None = None,
        requester: RequesterContext | None = None,
        storage: str = "long_term",
        increment_usage: bool = True,
    ) -> CognitiveIngestionResult:
        effective_actor_ctx = actor_ctx or self.requester_to_actor_context(requester)
        memory = await self.memory_service.create_memory(
            data=data,
            embedding=[],
            request_id=request_id,
            actor_ctx=effective_actor_ctx,
        )
        return await self.finalize_created_memory(
            memory=memory,
            content=data.content,
            request_id=request_id,
            storage=storage,
            increment_usage=increment_usage,
        )

    async def finalize_created_memory(
        self,
        *,
        memory: Any,
        content: str,
        request_id: str | None = None,
        storage: str = "long_term",
        increment_usage: bool = True,
    ) -> CognitiveIngestionResult:
        if increment_usage:
            usage = UsageService(self.session, self.org_id)
            await usage.increment(metric="memory_writes", value=1)

        await self.session.commit()
        pipelines = self._enqueue_pipelines(
            memory_id=memory.id,
            content=content,
            trace_id=request_id,
            storage=storage,
        )
        return CognitiveIngestionResult(
            memory=memory,
            storage=storage,
            pipelines_enqueued=pipelines,
        )

    def _enqueue_pipelines(
        self,
        *,
        memory_id: str,
        content: str,
        trace_id: str | None,
        storage: str,
    ) -> list[str]:
        enqueue_embed_and_index(
            memory_id=memory_id,
            content=content,
            org_id=self.org_id,
        )
        enqueue_memory_pipeline(
            org_id=self.org_id,
            memory_id=memory_id,
            initiator_user_id=self.user_id,
            initiator_roles=self.roles_string,
            initiator_clearance_level=self.clearance_level,
            trace_id=trace_id,
            storage=storage,
        )
        enqueue_episode_pipeline(
            org_id=self.org_id,
            memory_id=memory_id,
            initiator_user_id=self.user_id,
            initiator_roles=self.roles_string,
            initiator_clearance_level=self.clearance_level,
            trace_id=trace_id,
            storage=storage,
        )
        enqueue_fact_pipeline(
            org_id=self.org_id,
            memory_id=memory_id,
            initiator_user_id=self.user_id,
            initiator_roles=self.roles_string,
            initiator_clearance_level=self.clearance_level,
            trace_id=trace_id,
            storage=storage,
        )
        return ["embed", "memory_pipeline", "episode_pipeline", "fact_pipeline"]
