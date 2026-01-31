"""Evaluation report generation for Cognitive Loop sessions.

Produces a persisted EvaluationReport row summarizing a finished session.

Design goals:
- RLS-safe: assumes tenant context is set on the AsyncSession.
- Fail-closed: if required data is missing, creates a conservative report.
- Idempotent-ish: returns the latest report if one already exists unless forced.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.cognitive_session import CognitiveSession
from app.models.cognitive_iteration import CognitiveIteration
from app.models.evaluation_report import EvaluationReport
from app.models.tool_call_log import ToolCallLog
from app.models.base import generate_uuid
from app.schemas.cognitive import EvaluationQualityMetrics, EvaluationReportPayload


class EvaluationReportService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest_for_session(self, *, session_id: str) -> EvaluationReport | None:
        res = await self.session.execute(
            select(EvaluationReport)
            .where(EvaluationReport.session_id == session_id)
            .order_by(EvaluationReport.created_at.desc())
            .limit(1)
        )
        return res.scalar_one_or_none()

    async def generate_for_session(self, *, session_id: str, force: bool = False) -> EvaluationReport:
        if not force:
            existing = await self.get_latest_for_session(session_id=session_id)
            if existing is not None:
                return existing

        sres = await self.session.execute(select(CognitiveSession).where(CognitiveSession.id == session_id))
        sess = sres.scalar_one_or_none()
        if sess is None:
            raise ValueError(f"CognitiveSession not found: {session_id}")

        # Gather iterations + tool logs
        ires = await self.session.execute(
            select(CognitiveIteration)
            .where(CognitiveIteration.session_id == session_id)
            .order_by(CognitiveIteration.iteration_num.asc())
        )
        iterations = list(ires.scalars().all())

        tres = await self.session.execute(select(ToolCallLog).where(ToolCallLog.session_id == session_id))
        tool_logs = list(tres.scalars().all())

        iteration_count = max(1, len(iterations))

        confidences: list[float] = []
        evidence_ids: set[str] = set()
        for it in iterations:
            metrics = getattr(it, "metrics", {}) or {}
            try:
                confidences.append(float(metrics.get("confidence", 0.0) or 0.0))
            except Exception:
                confidences.append(0.0)
            for mid in list(metrics.get("evidence_memory_ids", []) or []):
                if mid:
                    evidence_ids.add(str(mid))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        policy_denials = sum(1 for t in tool_logs if str(getattr(t, "status", "")) == "denied")
        tool_failures = sum(1 for t in tool_logs if str(getattr(t, "status", "")) == "failed")

        status = str(getattr(sess, "status", "") or "")
        if status == "succeeded":
            final_decision = "pass"
            reason = "Session succeeded"
        elif status == "failed":
            # If failed, check if the last critic eval was needs_evidence; otherwise default to fail.
            final_decision = "fail"
            reason = "Session failed"
            if iterations:
                last_eval = str(iterations[-1].evaluation or "")
                if last_eval == "needs_evidence":
                    final_decision = "needs_evidence"
                    reason = "Session requires additional evidence"
        elif status == "aborted":
            final_decision = "contested"
            reason = "Session aborted"
        else:
            final_decision = "contested"
            reason = "Session not finalized"

        payload = EvaluationReportPayload(
            final_decision=final_decision,  # type: ignore[arg-type]
            reason=reason,
            goal_id=str(getattr(sess, "goal_id", None)) if getattr(sess, "goal_id", None) else None,
            evidence_memory_ids=sorted(evidence_ids),
            tool_calls=[str(getattr(t, "id")) for t in tool_logs],
            iteration_count=iteration_count,
            quality_metrics=EvaluationQualityMetrics(
                avg_confidence=float(max(0.0, min(1.0, avg_conf))),
                policy_denials=int(policy_denials),
                tool_failures=int(tool_failures),
            ),
        )

        row = EvaluationReport(
            id=generate_uuid(),
            session_id=session_id,
            report=payload.model_dump(),
            final_decision=payload.final_decision,
            created_at=datetime.now(timezone.utc),
        )
        self.session.add(row)
        await self.session.flush()
        return row
