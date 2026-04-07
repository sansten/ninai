"""CognitiveScheduleService — Phase 84.

CRUD + validation for user-defined cognitive schedules.

Cron validation uses a standard 5-field regex (no external deps).
compute_next_run uses a simple approximation sufficient for scheduler
integration; production deployments should swap in croniter.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_schedule import CognitiveSchedule


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SCHEDULES_PER_ORG = 20

_VALID_COGNITIVE_VERBS = frozenset({
    "analyze", "summarize", "plan", "report",
    "monitor", "escalate", "acknowledge",
})

# 5-field cron format: each field is * | */n | n[-n][/n][,n[-n][/n]]*
_CRON_FIELD = (
    r'(?:'
    r'\*(?:/\d+)?'                         # * or */n
    r'|\d+(?:-\d+)?(?:/\d+)?'             # n, n-n, n/m, n-n/m
    r'(?:,\d+(?:-\d+)?(?:/\d+)?)*'        # repeated with commas
    r')'
)
_CRON_PATTERN = re.compile(
    r'^\s*' + r'\s+'.join([_CRON_FIELD] * 5) + r'\s*$'
)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def validate_cron(expr: str) -> None:
    """Raise ValueError if expr is not a valid 5-field cron expression."""
    if not expr or not isinstance(expr, str):
        raise ValueError("cron expression must be a non-empty string")
    if not _CRON_PATTERN.match(expr.strip()):
        raise ValueError(f"invalid cron expression: {expr!r}")


def compute_next_run(expr: str, after: datetime | None = None) -> datetime:
    """Return the next scheduled datetime at or after *after* (UTC).

    This is a lightweight approximation:
    - `* * * * *` → next whole minute
    - `*/n * * * *` → next multiple-of-n minute
    - `m  * * * *` → next occurrence of that minute

    For production accuracy replace with croniter.
    """
    validate_cron(expr)
    base = (after or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    base += timedelta(minutes=1)  # always at least 1 minute in the future

    fields = expr.strip().split()
    minute_field = fields[0]

    if minute_field == "*":
        return base

    if minute_field.startswith("*/"):
        interval = int(minute_field[2:]) or 1
        elapsed = base.minute % interval
        if elapsed == 0:
            return base
        return base + timedelta(minutes=(interval - elapsed))

    # Specific minute value(s) — use first value
    targets = [int(p.split("-")[0].split("/")[0]) for p in minute_field.split(",")]
    for _ in range(60):
        if base.minute in targets:
            return base
        base += timedelta(minutes=1)
    # Fallback: add 1 hour
    return base


def validate_verb(verb: str) -> None:
    """Raise ValueError if verb is not a recognised cognitive verb."""
    if verb not in _VALID_COGNITIVE_VERBS:
        raise ValueError(
            f"invalid cognitive verb {verb!r}; "
            f"must be one of {sorted(_VALID_COGNITIVE_VERBS)}"
        )


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class CognitiveScheduleService:
    """CRUD service for CognitiveSchedule records."""

    def __init__(self, session: AsyncSession, org_id: str) -> None:
        self._session = session
        self._org_id = org_id

    async def _count_for_org(self) -> int:
        result = await self._session.execute(
            select(func.count()).select_from(CognitiveSchedule).where(
                CognitiveSchedule.organization_id == self._org_id
            )
        )
        return result.scalar_one()

    async def create(
        self,
        *,
        cron_expression: str,
        cognitive_verb: str,
        payload: dict | None = None,
        label: str | None = None,
    ) -> CognitiveSchedule:
        """Create a new schedule. Raises ValueError on validation failure or over-limit."""
        validate_cron(cron_expression)
        validate_verb(cognitive_verb)

        count = await self._count_for_org()
        if count >= MAX_SCHEDULES_PER_ORG:
            raise ValueError(
                f"organisation {self._org_id!r} already has {MAX_SCHEDULES_PER_ORG} "
                "schedules (limit reached)"
            )

        schedule = CognitiveSchedule(
            id=str(uuid.uuid4()),
            organization_id=self._org_id,
            cron_expression=cron_expression,
            cognitive_verb=cognitive_verb,
            payload=payload or {},
            label=label,
            is_active=True,
            next_run_at=compute_next_run(cron_expression),
        )
        self._session.add(schedule)
        await self._session.flush()
        return schedule

    async def get(self, schedule_id: str) -> CognitiveSchedule | None:
        result = await self._session.execute(
            select(CognitiveSchedule).where(
                CognitiveSchedule.id == schedule_id,
                CognitiveSchedule.organization_id == self._org_id,
            )
        )
        return result.scalars().first()

    async def list(self, *, include_inactive: bool = False) -> list[CognitiveSchedule]:
        stmt = select(CognitiveSchedule).where(
            CognitiveSchedule.organization_id == self._org_id
        )
        if not include_inactive:
            stmt = stmt.where(CognitiveSchedule.is_active.is_(True))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update(
        self,
        schedule_id: str,
        *,
        cron_expression: str | None = None,
        cognitive_verb: str | None = None,
        payload: dict | None = None,
        label: str | None = None,
        is_active: bool | None = None,
    ) -> CognitiveSchedule | None:
        sched = await self.get(schedule_id)
        if sched is None:
            return None

        if cron_expression is not None:
            validate_cron(cron_expression)
            sched.cron_expression = cron_expression
            sched.next_run_at = compute_next_run(cron_expression)
        if cognitive_verb is not None:
            validate_verb(cognitive_verb)
            sched.cognitive_verb = cognitive_verb
        if payload is not None:
            sched.payload = payload
        if label is not None:
            sched.label = label
        if is_active is not None:
            sched.is_active = is_active

        await self._session.flush()
        return sched

    async def delete(self, schedule_id: str) -> bool:
        sched = await self.get(schedule_id)
        if sched is None:
            return False
        await self._session.delete(sched)
        await self._session.flush()
        return True
