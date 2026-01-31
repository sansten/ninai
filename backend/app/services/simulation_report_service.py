"""SimulationReport persistence service.

Stores SimulationOutput JSON for auditing and downstream agents.

Assumes tenant context (RLS) is already set on the AsyncSession.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.simulation_report import SimulationReport


class SimulationReportService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        org_id: str,
        report: dict[str, Any],
        session_id: str | None = None,
        memory_id: str | None = None,
    ) -> SimulationReport:
        row = SimulationReport(
            organization_id=org_id,
            session_id=session_id,
            memory_id=memory_id,
            report=report,
        )
        self.session.add(row)
        await self.session.flush()
        return row
