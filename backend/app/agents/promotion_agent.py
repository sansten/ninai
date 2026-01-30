from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.core.database import get_tenant_session
from app.core.celery_app import celery_app
from app.services.memory_promoter import MemoryPromoter
from app.services.short_term_memory import ShortTermMemoryService


class PromotionAgent(BaseAgent):
    name = "PromotionAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("promoted"), bool):
            raise ValueError("promotion outputs.promoted must be a bool")
        if outputs.get("promoted") is True and not isinstance(outputs.get("promoted_memory_id"), str):
            raise ValueError("promotion outputs.promoted_memory_id must be a string when promoted")

    def _should_promote(self, *, content: str, enrichment: dict | None) -> tuple[bool, str]:
        lower = (content or "").lower()

        md = (enrichment or {}).get("metadata") if isinstance(enrichment, dict) else None
        entities = (md or {}).get("entities") if isinstance(md, dict) else {}

        topics = (enrichment or {}).get("topics") if isinstance(enrichment, dict) else None
        topic_list = (topics or {}).get("topics") if isinstance(topics, dict) else []
        normalized_topics = {str(t).strip().lower() for t in (topic_list or []) if str(t).strip()}

        classification = (enrichment or {}).get("classification") if isinstance(enrichment, dict) else None
        importance_score = None
        if isinstance(classification, dict):
            importance_score = classification.get("importance_score")

        # Promote if importance score is high.
        if importance_score is not None and float(importance_score) >= 0.75:
            return True, "classification_importance"

        # Promote if we have strong identifiers (order/ticket/email) and appears to be actionable.
        has_order = isinstance(entities, dict) and bool(entities.get("order_id"))
        has_email = isinstance(entities, dict) and bool(entities.get("email"))

        if has_order and any(k in lower for k in ["refund", "charge", "invoice", "payment", "shipping", "delivery"]):
            return True, "entity_order_actionable"

        if has_email and any(k in lower for k in ["login", "password", "account", "security"]):
            return True, "entity_email_account_security"

        # Topic-based heuristic.
        if normalized_topics.intersection({"billing", "security", "legal", "engineering"}):
            if any(k in lower for k in ["issue", "incident", "escalation", "contract", "invoice", "refund"]):
                return True, "topic_actionable"

        return False, "not_enough_signal"

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        tenant = context.get("tenant") or {}
        actor = context.get("actor") or {}
        memory = context.get("memory") or {}

        org_id = tenant.get("org_id")
        user_id = actor.get("user_id") or ""
        storage = memory.get("storage")
        content = memory.get("content", "")
        enrichment = memory.get("enrichment") if isinstance(memory, dict) else None

        if not org_id:
            raise ValueError("promotion requires tenant.org_id")

        if storage != "short_term":
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=0.0,
                outputs={"promoted": False, "reason": "not_short_term"},
                warnings=["Promotion runs only for short-term memories"],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
            self.validate_outputs(result)
            return result

        should_promote, reason = self._should_promote(content=content, enrichment=enrichment if isinstance(enrichment, dict) else None)

        if not should_promote:
            finished_at = datetime.now(timezone.utc)
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=0.4,
                outputs={"promoted": False, "reason": reason},
                warnings=[],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
            self.validate_outputs(result)
            return result

        # Side-effect: promote STM -> LTM. We run with tenant session vars (RLS).
        async with get_tenant_session(
            user_id=user_id,
            org_id=org_id,
            roles="",
            clearance_level=0,
            justification="promotion_agent",
        ) as session:
            promoter = MemoryPromoter(session=session, user_id=user_id, org_id=org_id)

            # Ensure memory exists in STM for the current actor.
            if user_id:
                stm_service = ShortTermMemoryService(user_id, org_id)
                stm = await stm_service.get(memory_id)
                if stm is None:
                    finished_at = datetime.now(timezone.utc)
                    result = AgentResult(
                        agent_name=self.name,
                        agent_version=self.version,
                        memory_id=memory_id,
                        status="success",
                        confidence=0.0,
                        outputs={"promoted": False, "reason": "stm_not_found"},
                        warnings=["Short-term memory not found for promotion"],
                        errors=[],
                        started_at=started_at,
                        finished_at=finished_at,
                        trace_id=str(trace_id) if trace_id else None,
                    )
                    self.validate_outputs(result)
                    return result

            promoted = await promoter.promote_by_id(stm_id=memory_id, reason=f"agent:{reason}")

        finished_at = datetime.now(timezone.utc)
        if promoted is None:
            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=0.0,
                outputs={"promoted": False, "reason": "promotion_failed"},
                warnings=["Promotion did not create a long-term record"],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )
        else:
            # Trigger point (LOGSEQ_INTEGRATION_SPEC.md): enqueue logseq export for promoted LTM memory.
            broker = celery_app.conf.broker_url
            if broker and not str(broker).startswith("memory://"):
                celery_app.send_task(
                    "app.tasks.memory_pipeline.logseq_export_task",
                    kwargs={
                        "org_id": str(org_id),
                        "memory_id": str(promoted.id),
                        "initiator_user_id": user_id or None,
                        "trace_id": str(trace_id) if trace_id else None,
                        "storage": "long_term",
                    },
                )

            result = AgentResult(
                agent_name=self.name,
                agent_version=self.version,
                memory_id=memory_id,
                status="success",
                confidence=0.75,
                outputs={"promoted": True, "promoted_memory_id": str(promoted.id), "reason": reason},
                warnings=[],
                errors=[],
                started_at=started_at,
                finished_at=finished_at,
                trace_id=str(trace_id) if trace_id else None,
            )

        self.validate_outputs(result)
        return result
