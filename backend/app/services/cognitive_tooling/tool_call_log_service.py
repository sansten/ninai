"""Tool call logging for Cognitive Loop.

This writes rows to tool_call_logs. Callers must ensure they do not persist
sensitive raw outputs unless allowed by tool sensitivity policy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tool_call_log import ToolCallLog


class ToolCallLogService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        session_id: str,
        iteration_id: str,
        tool_name: str,
        tool_input: dict,
        tool_output_summary: dict,
        status: str,
        denial_reason: str | None = None,
        warnings: list[str] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> ToolCallLog:
        started_at = started_at or datetime.now(timezone.utc)
        finished_at = finished_at or started_at

        safe_output_summary = dict(tool_output_summary or {})
        if warnings:
            # Persist warnings alongside the safe summary.
            # Avoid adding a dedicated DB column/migration for now.
            safe_output_summary["warnings"] = list(warnings)

        row = ToolCallLog(
            session_id=session_id,
            iteration_id=iteration_id,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output_summary=safe_output_summary,
            status=status,
            denial_reason=denial_reason,
            started_at=started_at,
            finished_at=finished_at,
        )
        self.session.add(row)
        await self.session.flush()
        return row
