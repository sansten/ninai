from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.core.database import get_tenant_session
from app.services.memory_feedback_service import MemoryFeedbackService


class FeedbackLearningAgent(BaseAgent):
    name = "FeedbackLearningAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        applied = outputs.get("applied")
        if not isinstance(applied, bool):
            raise ValueError("feedback learning outputs.applied must be a bool")
        if applied:
            if not isinstance(outputs.get("applied_count"), int):
                raise ValueError("feedback learning outputs.applied_count must be an int when applied")
            if not isinstance(outputs.get("updates"), list):
                raise ValueError("feedback learning outputs.updates must be a list when applied")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        tenant = context.get("tenant") or {}
        actor = context.get("actor") or {}
        memory_ctx = context.get("memory") or {}

        org_id = tenant.get("org_id")
        user_id = actor.get("user_id") or ""
        storage = memory_ctx.get("storage")

        if not org_id:
            raise ValueError("feedback learning requires tenant.org_id")

        if storage != "long_term":
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=0.0,
                outputs={"applied": False, "applied_count": 0, "updates": [], "reason": "not_long_term"},
                warnings=["Feedback learning currently applies only to long-term memories"],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
            self.validate_outputs(result)
            return result

        async with get_tenant_session(
            user_id=user_id,
            org_id=org_id,
            roles="",
            clearance_level=0,
            justification="feedback_learning_agent",
        ) as session:
            svc = MemoryFeedbackService(session, user_id=user_id, org_id=org_id)
            summary = await svc.apply_pending_feedback(memory_id=memory_id, applied_by=user_id or None)

        applied_count = int(summary.get("applied_count", 0) or 0)
        applied = applied_count > 0
        outputs: dict[str, Any] = {
            "applied": bool(applied),
            "applied_count": applied_count,
            "updates": summary.get("updates", []),
            "confidence": 0.65 if applied else 0.3,
            "rationale": "applied_pending_feedback" if applied else "no_pending_feedback",
        }

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.2)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )

        self.validate_outputs(result)
        return result
