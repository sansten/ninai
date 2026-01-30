from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_run import AgentRun
from app.models.cognitive_session import CognitiveSession
from app.models.evaluation_report import EvaluationReport
from app.models.memory import MemoryMetadata
from app.models.meta_agent import MetaAgentRun, MetaConflictRegistry
from app.services.meta_agent.calibration_service import CalibrationService
from app.services.meta_agent.confidence_aggregator import AggregationInputs, ConfidenceAggregator
from app.services.meta_agent.belief_store_service import BeliefStoreService
from app.services.meta_agent.conflict_resolver import (
    ClassificationCandidate,
    detect_classification_conflict,
    resolve_classification_candidates,
)


class MetaSupervisor:
    def __init__(self):
        self.calibration_service = CalibrationService()
        self.belief_store = BeliefStoreService()

    async def review_memory(self, session: AsyncSession, *, org_id: str, memory_id: str, trace_id: str | None = None) -> MetaAgentRun:
        now = datetime.utcnow()
        run = MetaAgentRun(
            id=str(uuid.uuid4()),
            organization_id=org_id,
            resource_type="memory",
            resource_id=memory_id,
            supervision_type="review",
            status="contested",
            final_confidence=None,
            risk_score=None,
            reasoning_summary=None,
            evidence={},
            created_at=now,
        )
        session.add(run)
        await session.flush()

        try:
            profile = await self.calibration_service.get_profile_for_read(session, org_id=org_id)
            aggregator = ConfidenceAggregator(signal_weights=profile.signal_weights)

            # Collect candidate classifications
            mem_res = await session.execute(
                select(MemoryMetadata).where(MemoryMetadata.organization_id == org_id, MemoryMetadata.id == memory_id)
            )
            memory = mem_res.scalar_one_or_none()
            if memory is None:
                raise ValueError("memory_not_found")

            existing_classification = str(getattr(memory, "classification", "") or "").strip().lower() or None

            raw_candidates: list[str] = []
            confidence_candidates: list[ClassificationCandidate] = []
            if existing_classification:
                raw_candidates.append(existing_classification)
                confidence_candidates.append(ClassificationCandidate(existing_classification, 0.5))

            agent_res = await session.execute(
                select(AgentRun)
                .where(AgentRun.organization_id == org_id, AgentRun.memory_id == memory_id)
                .order_by(AgentRun.started_at.desc())
            )
            for r in agent_res.scalars().all():
                if r.agent_name == "ClassificationAgent" and isinstance(r.outputs, dict):
                    c = (r.outputs or {}).get("classification")
                    if isinstance(c, str) and c.strip():
                        val = c.strip().lower()
                        raw_candidates.append(val)

                        conf = None
                        out_conf = (r.outputs or {}).get("confidence")
                        if isinstance(out_conf, (int, float)):
                            conf = float(out_conf)
                        else:
                            conf = float(getattr(r, "confidence", 0.0) or 0.0)

                        confidence_candidates.append(ClassificationCandidate(val, max(0.0, min(1.0, conf))))
                    break

            conflict = detect_classification_conflict(raw_candidates)
            if conflict.has_conflict:
                conflict_row = MetaConflictRegistry(
                    id=str(uuid.uuid4()),
                    organization_id=org_id,
                    resource_type="memory",
                    resource_id=memory_id,
                    conflict_type="classification",
                    candidates={"classifications": (conflict.details or {}).get("candidates", [])},
                    resolution={},
                    resolved_by=None,
                    status="open",
                    created_at=now,
                )
                session.add(conflict_row)

            resolved = resolve_classification_candidates(
                confidence_candidates,
                confidence_gap_threshold=float(profile.conflict_escalation_threshold or 0.60),
            )

            downgrade_attempt = False
            if existing_classification and existing_classification in {"public", "internal", "confidential", "restricted"}:
                order = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
                downgrade_attempt = order.get(resolved, 1) < order.get(existing_classification, 1)

            # Simple deterministic signals
            agent_conf = 0.5
            if confidence_candidates:
                distinct = {c.value.lower() for c in confidence_candidates}
                agent_conf = 0.7 if len(distinct) == 1 else 0.4

            inputs = AggregationInputs(
                agent_confidence=agent_conf,
                evidence_strength=0.5,
                historical_accuracy=0.5,
                consistency_score=1.0 if not conflict.has_conflict else 0.5,
                contradiction_penalty=1.0 if conflict.has_conflict else 0.0,
            )
            agg = aggregator.aggregate(inputs)

            status_out = "accepted"
            if downgrade_attempt:
                status_out = "escalated"
            elif conflict.has_conflict and float(agg.risk_score) >= float(profile.conflict_escalation_threshold or 0.60):
                status_out = "contested"
            elif existing_classification and resolved != existing_classification:
                status_out = "modified"

            run.status = status_out
            run.final_confidence = agg.overall_confidence
            run.risk_score = agg.risk_score
            run.reasoning_summary = "classification_review"
            run.evidence = {
                "resolved": {"classification": resolved},
                "candidates": raw_candidates,
                "downgrade_attempt": downgrade_attempt,
            }
            await session.flush()

            # Belief revision: store classification belief
            contradiction_ids = []
            if conflict.has_conflict:
                contradiction_ids.append(conflict_row.id)
                conflict_row.resolution = {"resolved": {"classification": resolved}}
                conflict_row.resolved_by = "meta_auto" if status_out != "escalated" else None
                if status_out in {"accepted", "modified"}:
                    conflict_row.status = "resolved"
                    conflict_row.resolved_at = datetime.utcnow()

            await self.belief_store.upsert_belief(
                session,
                org_id=org_id,
                memory_id=memory_id,
                belief_key="classification",
                belief_value={"classification": resolved},
                confidence=float(agg.overall_confidence),
                evidence_memory_ids=[],
                contradiction_ids=contradiction_ids,
            )

            return run

        except Exception as exc:  # fail-closed
            run.status = "escalated"
            run.final_confidence = 0.0
            run.risk_score = 1.0
            run.reasoning_summary = "meta_review_failed"
            run.evidence = {"error": str(exc)}
            await session.flush()
            return run

    async def review_cognitive_session(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        session_id: str,
        trace_id: str | None = None,
    ) -> MetaAgentRun:
        now = datetime.utcnow()
        run = MetaAgentRun(
            id=str(uuid.uuid4()),
            organization_id=org_id,
            resource_type="cognitive_session",
            resource_id=session_id,
            supervision_type="review",
            status="contested",
            final_confidence=None,
            risk_score=None,
            reasoning_summary=None,
            evidence={},
            created_at=now,
        )
        session.add(run)
        await session.flush()

        try:
            # Validate session exists + org match (defense-in-depth beyond RLS)
            sess_res = await session.execute(
                select(CognitiveSession).where(CognitiveSession.id == session_id, CognitiveSession.organization_id == org_id)
            )
            sess = sess_res.scalar_one_or_none()
            if sess is None:
                raise ValueError("session_not_found")

            profile = await self.calibration_service.get_profile_for_read(session, org_id=org_id)
            aggregator = ConfidenceAggregator(signal_weights=profile.signal_weights)

            rep_res = await session.execute(
                select(EvaluationReport).where(EvaluationReport.session_id == session_id).order_by(EvaluationReport.created_at.desc())
            )
            report = rep_res.scalars().first()
            if report is None:
                raise ValueError("missing_evaluation_report")

            decision = str(getattr(report, "final_decision", "") or "").strip().lower()
            agent_conf = 0.8 if decision == "pass" else 0.3
            contradiction = 0.0 if decision == "pass" else 0.4

            inputs = AggregationInputs(
                agent_confidence=agent_conf,
                evidence_strength=0.6,
                historical_accuracy=0.5,
                consistency_score=0.8,
                contradiction_penalty=contradiction,
            )
            agg = aggregator.aggregate(inputs)

            if decision == "fail":
                status_out = "rejected"
            elif decision == "pass" and float(agg.overall_confidence) < float(profile.conflict_escalation_threshold or 0.60):
                status_out = "contested"
            else:
                status_out = "accepted"

            run.status = status_out
            run.final_confidence = agg.overall_confidence
            run.risk_score = agg.risk_score
            run.reasoning_summary = "cognitive_session_review"
            run.evidence = {"evaluation": {"final_decision": decision}}
            await session.flush()
            return run

        except Exception as exc:
            run.status = "escalated"
            run.final_confidence = 0.0
            run.risk_score = 1.0
            run.reasoning_summary = "meta_review_failed"
            run.evidence = {"error": str(exc)}
            await session.flush()
            return run

    async def list_runs(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        limit: int = 50,
    ) -> list[MetaAgentRun]:
        stmt = select(MetaAgentRun).where(MetaAgentRun.organization_id == org_id)
        if resource_type:
            stmt = stmt.where(MetaAgentRun.resource_type == resource_type)
        if resource_id:
            stmt = stmt.where(MetaAgentRun.resource_id == resource_id)
        stmt = stmt.order_by(MetaAgentRun.created_at.desc()).limit(int(limit))
        res = await session.execute(stmt)
        return list(res.scalars().all())

    async def list_conflicts(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        status: str | None = None,
        limit: int = 50,
    ) -> list[MetaConflictRegistry]:
        stmt = select(MetaConflictRegistry).where(MetaConflictRegistry.organization_id == org_id)
        if status:
            stmt = stmt.where(MetaConflictRegistry.status == status)
        stmt = stmt.order_by(MetaConflictRegistry.created_at.desc()).limit(int(limit))
        res = await session.execute(stmt)
        return list(res.scalars().all())
