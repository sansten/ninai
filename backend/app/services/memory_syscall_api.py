"""Memory syscall API surface.

Capability-scoped operations: read, append, search, upsert, consolidate.
All operations require valid capability token and are audit-logged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid5, NAMESPACE_DNS

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.capability_token import CapabilityToken, MemorySyscallScope
from app.services.audit_service import AuditService


def _normalize_memory_id_to_uuid(memory_id: str) -> str:
    """Convert memory_id string to valid UUID string for audit logging."""
    try:
        # If it's already a valid UUID, return as-is
        UUID(memory_id)
        return memory_id
    except (ValueError, TypeError):
        # For test/synthetic IDs, create deterministic UUID
        # Using NAMESPACE_DNS and the memory_id as name
        synthetic_uuid = uuid5(NAMESPACE_DNS, memory_id)
        return str(synthetic_uuid)


class MemorySyscallAPI:
    """Capability-scoped memory syscall interface."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit = AuditService(db)

    async def read(
        self,
        *,
        memory_id: str,
        token: CapabilityToken,
    ) -> dict[str, Any]:
        """Read memory item. Requires memory.read scope."""
        token.validate(MemorySyscallScope.READ)

        audit_resource_id = _normalize_memory_id_to_uuid(memory_id)

        await self.audit.log_event(
            event_type="memory.syscall.read",
            actor_id=token.actor_user_id,
            organization_id=token.organization_id,
            resource_type="memory",
            resource_id=audit_resource_id,
            success=True,
            details={
                "token_id": token.token_id,
                "session_id": token.session_id,
                "agent_id": token.agent_id,
            },
        )

        # TODO: Implement actual read logic
        return {
            "id": memory_id,
            "content": "placeholder",
            "accessed_at": datetime.utcnow().isoformat(),
        }

    async def append(
        self,
        *,
        memory_id: str,
        content: dict[str, Any],
        token: CapabilityToken,
    ) -> dict[str, Any]:
        """Append to memory. Requires memory.append scope."""
        token.validate(MemorySyscallScope.APPEND)

        audit_resource_id = _normalize_memory_id_to_uuid(memory_id)

        await self.audit.log_event(
            event_type="memory.syscall.append",
            actor_id=token.actor_user_id,
            organization_id=token.organization_id,
            resource_type="memory",
            resource_id=audit_resource_id,
            success=True,
            details={
                "token_id": token.token_id,
                "session_id": token.session_id,
                "agent_id": token.agent_id,
                "content_keys": list(content.keys()),
            },
        )

        # TODO: Implement actual append logic
        return {"id": memory_id, "appended_at": datetime.utcnow().isoformat()}

    async def search(
        self,
        *,
        query: str,
        limit: int = 10,
        token: CapabilityToken,
    ) -> list[dict[str, Any]]:
        """Search memory. Requires memory.search scope."""
        token.validate(MemorySyscallScope.SEARCH)

        await self.audit.log_event(
            event_type="memory.syscall.search",
            actor_id=token.actor_user_id,
            organization_id=token.organization_id,
            resource_type="memory",
            resource_id=token.organization_id,
            success=True,
            details={
                "token_id": token.token_id,
                "session_id": token.session_id,
                "agent_id": token.agent_id,
                "query_length": len(query),
                "limit": limit,
            },
        )

        # TODO: Implement actual search logic
        return [{"id": "match1", "score": 0.95}]

    async def upsert(
        self,
        *,
        memory_id: Optional[str],
        content: dict[str, Any],
        token: CapabilityToken,
    ) -> dict[str, Any]:
        """Upsert memory (insert or update). Requires memory.upsert scope."""
        token.validate(MemorySyscallScope.UPSERT)

        result_id = memory_id or str(uuid5(NAMESPACE_DNS, str(datetime.utcnow())))
        audit_resource_id = _normalize_memory_id_to_uuid(result_id)

        await self.audit.log_event(
            event_type="memory.syscall.upsert",
            actor_id=token.actor_user_id,
            organization_id=token.organization_id,
            resource_type="memory",
            resource_id=audit_resource_id,
            success=True,
            details={
                "token_id": token.token_id,
                "session_id": token.session_id,
                "agent_id": token.agent_id,
                "is_insert": memory_id is None,
            },
        )

        # TODO: Implement actual upsert logic
        return {"id": result_id, "upserted_at": datetime.utcnow().isoformat()}

    async def consolidate(
        self,
        *,
        memory_id: str,
        token: CapabilityToken,
    ) -> dict[str, Any]:
        """Consolidate memory (compress, summarize). Requires memory.consolidate scope."""
        token.validate(MemorySyscallScope.CONSOLIDATE)

        audit_resource_id = _normalize_memory_id_to_uuid(memory_id)

        await self.audit.log_event(
            event_type="memory.syscall.consolidate",
            actor_id=token.actor_user_id,
            organization_id=token.organization_id,
            resource_type="memory",
            resource_id=audit_resource_id,
            success=True,
            details={
                "token_id": token.token_id,
                "session_id": token.session_id,
                "agent_id": token.agent_id,
            },
        )

        # TODO: Implement actual consolidate logic
        return {"id": memory_id, "consolidated_at": datetime.utcnow().isoformat()}

    async def feedback(
        self,
        *,
        memory_id: str,
        feedback_type: str,
        details: dict[str, Any],
        token: CapabilityToken,
    ) -> dict[str, Any]:
        """Record feedback on memory. Requires memory.feedback scope."""
        token.validate(MemorySyscallScope.FEEDBACK)

        audit_resource_id = _normalize_memory_id_to_uuid(memory_id)

        await self.audit.log_event(
            event_type="memory.syscall.feedback",
            actor_id=token.actor_user_id,
            organization_id=token.organization_id,
            resource_type="memory",
            resource_id=audit_resource_id,
            success=True,
            details={
                "token_id": token.token_id,
                "session_id": token.session_id,
                "agent_id": token.agent_id,
                "feedback_type": feedback_type,
            },
        )

        # TODO: Implement actual feedback logic
        return {"id": memory_id, "feedback_recorded_at": datetime.utcnow().isoformat()}
