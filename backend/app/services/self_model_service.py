"""SelfModel service.

Computes and caches per-organization calibration signals used by planning and policy.

Design goals:
- RLS-safe: assumes tenant context is set on the AsyncSession.
- Idempotent ingestion: tool/evaluation-derived events reuse source row IDs.
- Fail-open caching: if Redis is unavailable, DB remains source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import RedisClient
from app.models.cognitive_session import CognitiveSession
from app.models.evaluation_report import EvaluationReport
from app.models.self_model import SelfModelEvent, SelfModelProfile
from app.models.tool_call_log import ToolCallLog


_SELF_MODEL_PROFILE_CACHE_PREFIX = "selfmodel:profile"
_SELF_MODEL_PROFILE_CACHE_TTL_SECONDS = 300


@dataclass(frozen=True)
class PlannerSummary:
    unreliable_tools: list[str]
    low_confidence_domains: list[str]
    recommended_evidence_multiplier: int


def _clamp01(val: float) -> float:
    return float(max(0.0, min(1.0, val)))


def ema(*, previous: float | None, observed: float, alpha: float) -> float:
    """Simple exponential moving average bounded to [0,1]."""
    a = float(max(0.0, min(1.0, alpha)))
    obs = _clamp01(float(observed))
    if previous is None:
        return obs
    return _clamp01((1.0 - a) * _clamp01(float(previous)) + a * obs)


def p95(values: Iterable[float]) -> float | None:
    items = sorted(float(v) for v in values if v is not None)
    if not items:
        return None
    # Nearest-rank method
    k = int(max(1, round(0.95 * len(items)))) - 1
    k = max(0, min(len(items) - 1, k))
    return float(items[k])


class SelfModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    def _cache_key(self, *, org_id: str) -> str:
        return f"{_SELF_MODEL_PROFILE_CACHE_PREFIX}:{org_id}"

    async def _cache_get(self, *, org_id: str) -> dict[str, Any] | None:
        try:
            return await RedisClient.get_json(self._cache_key(org_id=org_id))
        except Exception:
            return None

    async def _cache_set(self, *, org_id: str, payload: dict[str, Any]) -> None:
        try:
            await RedisClient.set_json(
                self._cache_key(org_id=org_id),
                payload,
                ttl=_SELF_MODEL_PROFILE_CACHE_TTL_SECONDS,
            )
        except Exception:
            return None

    async def _cache_invalidate(self, *, org_id: str) -> None:
        try:
            await RedisClient.delete(self._cache_key(org_id=org_id))
        except Exception:
            return None

    async def get_profile(self, *, org_id: str) -> SelfModelProfile:
        cached = await self._cache_get(org_id=org_id)
        if cached:
            last_updated_raw = cached.get("last_updated")
            try:
                last_updated = datetime.fromisoformat(str(last_updated_raw))
            except Exception:
                last_updated = datetime.now(timezone.utc)
            return SelfModelProfile(
                organization_id=org_id,
                domain_confidence=cached.get("domain_confidence") or {},
                tool_reliability=cached.get("tool_reliability") or {},
                agent_accuracy=cached.get("agent_accuracy") or {},
                last_updated=last_updated,
            )

        res = await self.session.execute(
            select(SelfModelProfile).where(SelfModelProfile.organization_id == org_id)
        )
        row = res.scalar_one_or_none()
        if row is None:
            row = SelfModelProfile(
                organization_id=org_id,
                domain_confidence={},
                tool_reliability={},
                agent_accuracy={},
            )
            self.session.add(row)
            await self.session.flush()

        await self._cache_set(
            org_id=org_id,
            payload={
                "domain_confidence": row.domain_confidence or {},
                "tool_reliability": row.tool_reliability or {},
                "agent_accuracy": row.agent_accuracy or {},
                "last_updated": (row.last_updated or datetime.now(timezone.utc)).isoformat(),
            },
        )
        return row

    async def get_planner_summary(self, *, org_id: str) -> PlannerSummary:
        prof = await self.get_profile(org_id=org_id)

        tool_reliability = prof.tool_reliability or {}
        unreliable_tools: list[str] = []
        for tool_name, stats in tool_reliability.items():
            if not isinstance(stats, dict):
                continue
            rate = stats.get("success_rate_30d")
            n = stats.get("sample_size_30d")
            try:
                if n is not None and int(n) < 3:
                    continue
                if rate is not None and float(rate) < 0.80:
                    unreliable_tools.append(str(tool_name))
            except Exception:
                continue

        domain_conf = prof.domain_confidence or {}
        low_domains = [d for d, v in domain_conf.items() if isinstance(v, (int, float)) and float(v) < 0.60]

        multiplier = 1
        if low_domains or unreliable_tools:
            multiplier = 2

        return PlannerSummary(
            unreliable_tools=sorted(set(unreliable_tools))[:10],
            low_confidence_domains=sorted(set(low_domains))[:10],
            recommended_evidence_multiplier=int(multiplier),
        )

    async def ingest_from_session(self, *, org_id: str, session_id: str) -> int:
        """Create self_model_events derived from tool_call_logs and evaluation_reports.

        Idempotency: tool/eval events reuse the source row UUID as the SelfModelEvent.id.
        """

        # Tool call derived events.
        tres = await self.session.execute(select(ToolCallLog).where(ToolCallLog.session_id == session_id))
        tool_logs = list(tres.scalars().all())

        values: list[dict[str, Any]] = []
        for t in tool_logs:
            status = str(getattr(t, "status", "") or "")
            if status == "success":
                event_type = "tool_success"
            elif status == "failed":
                event_type = "tool_failure"
            elif status == "denied":
                event_type = "policy_denial"
            else:
                continue

            dur_ms: float | None = None
            started = getattr(t, "started_at", None)
            finished = getattr(t, "finished_at", None)
            if started is not None and finished is not None:
                try:
                    dur_ms = max(0.0, (finished - started).total_seconds() * 1000.0)
                except Exception:
                    dur_ms = None

            values.append(
                {
                    "id": str(getattr(t, "id")),
                    "organization_id": org_id,
                    "event_type": event_type,
                    "tool_name": str(getattr(t, "tool_name", "") or "") or None,
                    "agent_name": None,
                    "session_id": session_id,
                    "memory_id": None,
                    "payload": {
                        "source": "tool_call_log",
                        "tool_call_id": str(getattr(t, "id")),
                        "status": status,
                        "duration_ms": dur_ms,
                        "denial_reason": getattr(t, "denial_reason", None),
                    },
                    # created_at stays server-side default; we keep insertion cheap.
                }
            )

        inserted = 0
        if values:
            stmt = insert(SelfModelEvent).values(values)
            stmt = stmt.on_conflict_do_nothing(index_elements=[SelfModelEvent.id])
            res = await self.session.execute(stmt)
            inserted += int(getattr(res, "rowcount", 0) or 0)

        # Evaluation report derived event.
        eres = await self.session.execute(
            select(EvaluationReport)
            .where(EvaluationReport.session_id == session_id)
            .order_by(EvaluationReport.created_at.desc())
            .limit(1)
        )
        report = eres.scalar_one_or_none()
        if report is not None:
            payload = {
                "source": "evaluation_report",
                "evaluation_report_id": str(getattr(report, "id")),
                "final_decision": str(getattr(report, "final_decision", "")),
            }
            stmt2 = insert(SelfModelEvent).values(
                {
                    "id": str(getattr(report, "id")),
                    "organization_id": org_id,
                    "event_type": "agent_confirmed",
                    "tool_name": None,
                    "agent_name": "evaluation_report",
                    "session_id": session_id,
                    "memory_id": None,
                    "payload": payload,
                }
            )
            stmt2 = stmt2.on_conflict_do_nothing(index_elements=[SelfModelEvent.id])
            res2 = await self.session.execute(stmt2)
            inserted += int(getattr(res2, "rowcount", 0) or 0)

        return inserted

    async def recompute_profile(self, *, org_id: str, alpha: float = 0.20) -> SelfModelProfile:
        """Recompute tool reliability and domain confidence (EMA smoothed)."""

        now = datetime.now(timezone.utc)
        since_7 = now - timedelta(days=7)
        since_30 = now - timedelta(days=30)

        # Load previous profile.
        prev = await self.get_profile(org_id=org_id)
        prev_tool = prev.tool_reliability or {}
        prev_domain = prev.domain_confidence or {}

        # Tool reliability based on ingested events.
        eres = await self.session.execute(
            select(SelfModelEvent)
            .where(
                SelfModelEvent.organization_id == org_id,
                SelfModelEvent.created_at >= since_30,
                SelfModelEvent.event_type.in_(["tool_success", "tool_failure", "policy_denial"]),
            )
        )
        events = list(eres.scalars().all())

        by_tool: dict[str, dict[str, Any]] = {}
        for ev in events:
            tool = str(getattr(ev, "tool_name", "") or "")
            if not tool:
                continue
            created_at = getattr(ev, "created_at", None)
            et = str(getattr(ev, "event_type", "") or "")
            payload = getattr(ev, "payload", {}) or {}

            dur_ms = payload.get("duration_ms")
            try:
                dur_ms_f = float(dur_ms) if dur_ms is not None else None
            except Exception:
                dur_ms_f = None

            entry = by_tool.setdefault(
                tool,
                {
                    "succ_7": 0,
                    "fail_7": 0,
                    "deny_7": 0,
                    "dur_7": [],
                    "succ_30": 0,
                    "fail_30": 0,
                    "deny_30": 0,
                    "dur_30": [],
                },
            )

            in_7 = bool(created_at is not None and created_at >= since_7)

            if et == "tool_success":
                entry["succ_30"] += 1
                if in_7:
                    entry["succ_7"] += 1
            elif et == "tool_failure":
                entry["fail_30"] += 1
                if in_7:
                    entry["fail_7"] += 1
            elif et == "policy_denial":
                entry["deny_30"] += 1
                if in_7:
                    entry["deny_7"] += 1

            if dur_ms_f is not None:
                entry["dur_30"].append(dur_ms_f)
                if in_7:
                    entry["dur_7"].append(dur_ms_f)

        new_tool: dict[str, Any] = {}
        for tool, c in by_tool.items():
            total_30 = int(c["succ_30"] + c["fail_30"] + c["deny_30"])
            total_7 = int(c["succ_7"] + c["fail_7"] + c["deny_7"])

            rate_30 = float(c["succ_30"]) / total_30 if total_30 else None
            rate_7 = float(c["succ_7"]) / total_7 if total_7 else None

            prev_stats = prev_tool.get(tool) if isinstance(prev_tool, dict) else None
            prev_rate = None
            if isinstance(prev_stats, dict):
                prev_rate = prev_stats.get("success_rate_30d")

            smoothed = ema(previous=float(prev_rate) if prev_rate is not None else None, observed=float(rate_30 or 0.0), alpha=alpha)

            new_tool[tool] = {
                "success_rate_7d": _clamp01(rate_7) if rate_7 is not None else None,
                "success_rate_30d": _clamp01(smoothed),
                "sample_size_7d": total_7,
                "sample_size_30d": total_30,
                "p95_ms_7d": p95(c["dur_7"]),
                "p95_ms_30d": p95(c["dur_30"]),
            }

        # Domain confidence based on evaluation reports.
        # Uses context_snapshot['domain'] when present; falls back to 'general'.
        dres = await self.session.execute(
            select(EvaluationReport, CognitiveSession)
            .join(CognitiveSession, CognitiveSession.id == EvaluationReport.session_id)
            .where(
                CognitiveSession.organization_id == org_id,
                EvaluationReport.created_at >= since_30,
            )
        )
        domain_scores: dict[str, list[float]] = {}
        for report, sess in dres.all():
            ctx = getattr(sess, "context_snapshot", {}) or {}
            domain = str(ctx.get("domain") or "general")
            dec = str(getattr(report, "final_decision", "") or "")
            if dec == "pass":
                score = 1.0
            elif dec == "contested":
                score = 0.5
            else:
                score = 0.0
            domain_scores.setdefault(domain, []).append(score)

        new_domain: dict[str, float] = dict(prev_domain) if isinstance(prev_domain, dict) else {}
        for domain, scores in domain_scores.items():
            observed = sum(scores) / max(1, len(scores))
            prev_val = new_domain.get(domain)
            new_domain[domain] = ema(
                previous=float(prev_val) if prev_val is not None else None,
                observed=float(observed),
                alpha=alpha,
            )

        upsert = insert(SelfModelProfile).values(
            {
                "organization_id": org_id,
                "domain_confidence": new_domain,
                "tool_reliability": new_tool,
                "agent_accuracy": prev.agent_accuracy or {},
                "last_updated": now,
            }
        )
        upsert = upsert.on_conflict_do_update(
            index_elements=[SelfModelProfile.organization_id],
            set_={
                "domain_confidence": upsert.excluded.domain_confidence,
                "tool_reliability": upsert.excluded.tool_reliability,
                "agent_accuracy": upsert.excluded.agent_accuracy,
                "last_updated": upsert.excluded.last_updated,
            },
        )
        await self.session.execute(upsert)

        await self._cache_invalidate(org_id=org_id)

        fres = await self.session.execute(
            select(SelfModelProfile)
            .where(SelfModelProfile.organization_id == org_id)
            .execution_options(populate_existing=True)
        )
        row = fres.scalar_one()
        # Defensive: ensure JSON fields reflect the DB upsert results.
        await self.session.refresh(row)
        return row
