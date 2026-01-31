"""Agent scheduler service.

Provides a small process table + scheduling logic for agent work.
This is intentionally DB-backed to enable fairness, quotas, and observability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_process import AgentProcess
from app.models.base import utc_now


ALLOWED_TERMINAL_STATUSES = {"succeeded", "failed", "blocked"}
REQUIRED_ENQUEUE_SCOPE = {"scheduler.enqueue"}
REQUIRED_DEQUEUE_SCOPE = {"scheduler.dequeue"}
REQUIRED_UPDATE_SCOPE = {"scheduler.update"}


def _ensure_scope(required: set[str], provided: Iterable[str] | None) -> None:
    provided_set = set(provided or [])
    if not required.issubset(provided_set):
        missing = required - provided_set
        raise PermissionError(f"Missing capability scope(s): {','.join(sorted(missing))}")


class AgentSchedulerService:
    """Manage agent processes with tenant-scoped fairness and quotas."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        max_running_per_org: int = 2,
        auto_commit: bool = True,
    ) -> None:
        self.db = db
        self.max_running_per_org = max_running_per_org
        self.auto_commit = auto_commit

    async def enqueue(
        self,
        *,
        organization_id: str,
        agent_name: str,
        priority: int = 0,
        session_id: str | None = None,
        agent_run_id: str | None = None,
        max_attempts: int = 3,
        quota_tokens: int = 0,
        quota_storage_mb: int = 0,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        scopes: Iterable[str] | None = None,
    ) -> AgentProcess:
        _ensure_scope(REQUIRED_ENQUEUE_SCOPE, scopes)
        proc = AgentProcess(
            organization_id=organization_id,
            session_id=session_id,
            agent_run_id=agent_run_id,
            agent_name=agent_name,
            priority=priority,
            status="queued",
            attempts=0,
            max_attempts=max_attempts,
            quota_tokens=quota_tokens,
            quota_storage_mb=quota_storage_mb,
            trace_id=trace_id,
            process_metadata=metadata or {},
        )
        self.db.add(proc)
        await self._persist()
        await self.db.refresh(proc)
        return proc

    async def _running_count(self, org_id: str) -> int:
        res = await self.db.execute(
            select(func.count())
            .select_from(AgentProcess)
            .where(AgentProcess.organization_id == org_id, AgentProcess.status == "running")
        )
        return int(res.scalar_one())

    async def dequeue_next(self, *, org_id: str, scopes: Iterable[str] | None = None) -> AgentProcess | None:
        """Fetch and mark the next runnable process.
    _ensure_scope(REQUIRED_DEQUEUE_SCOPE, scopes)

        Applies per-tenant running caps and skips processes out of attempts.
        """

        running = await self._running_count(org_id)
        if running >= self.max_running_per_org:
            return None

        stmt = (
            select(AgentProcess)
            .where(
                AgentProcess.organization_id == org_id,
                AgentProcess.status == "queued",
                AgentProcess.attempts < AgentProcess.max_attempts,
            )
            .order_by(AgentProcess.priority.desc(), AgentProcess.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        res = await self.db.execute(stmt)
        proc = res.scalar_one_or_none()
        if proc is None:
            return None

        proc.status = "running"
        proc.attempts += 1
        proc.started_at = utc_now()
        proc.last_error = ""
        await self._persist()
        await self.db.refresh(proc)
        return proc

    async def mark_blocked(self, *, process_id: str, reason: str = "", scopes: Iterable[str] | None = None) -> None:
        await self._mark_terminal(process_id, status="blocked", reason=reason, scopes=scopes)

    async def mark_failed(self, *, process_id: str, reason: str = "", scopes: Iterable[str] | None = None) -> None:
        await self._mark_terminal(process_id, status="failed", reason=reason, scopes=scopes)

    async def mark_succeeded(self, *, process_id: str, scopes: Iterable[str] | None = None) -> None:
        await self._mark_terminal(process_id, status="succeeded", reason="", scopes=scopes)

    async def _mark_terminal(self, process_id: str, *, status: str, reason: str, scopes: Iterable[str] | None) -> None:
        if status not in ALLOWED_TERMINAL_STATUSES:
            raise ValueError(f"Invalid terminal status: {status}")

        _ensure_scope(REQUIRED_UPDATE_SCOPE, scopes)

        res = await self.db.execute(
            select(AgentProcess).where(AgentProcess.id == process_id).with_for_update()
        )
        proc = res.scalar_one_or_none()
        if proc is None:
            return

        proc.status = status
        proc.finished_at = utc_now()
        if reason:
            proc.last_error = reason
        await self._persist()

    async def reset_to_queue(
        self, *, process_id: str, reason: str = "", scopes: Iterable[str] | None = None
    ) -> None:
        """Return a process to the queue (e.g., after backpressure).

        Does not reset attempts; caller controls preemption policy.
        """

        _ensure_scope(REQUIRED_UPDATE_SCOPE, scopes)

        res = await self.db.execute(
            select(AgentProcess).where(AgentProcess.id == process_id).with_for_update()
        )
        proc = res.scalar_one_or_none()
        if proc is None:
            return

        proc.status = "queued"
        proc.started_at = None
        proc.finished_at = None
        if reason:
            proc.last_error = reason
        await self._persist()

    async def _persist(self) -> None:
        if self.auto_commit:
            await self.db.commit()
        else:
            await self.db.flush()
