from __future__ import annotations

from typing import Any

from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.models.agent_run import AgentRun
from app.models.autonomous_goal import AutonomousGoal
from app.models.contradiction import Contradiction
from app.models.knowledge_gap import KnowledgeGap
from app.models.memory_fact import MemoryFact
from app.services.intrinsic_motivation_service import IntrinsicMotivationService


class CognitiveGoalLoopService:
    """Surfaces active goals, knowledge gaps, and lightweight world-state context."""

    WORLD_MODEL_AGENT = "WorldModelAgent"

    def __init__(self, session: AsyncSession, *, org_id: str) -> None:
        self.session = session
        self.org_id = org_id

    async def build_context(
        self,
        *,
        include_suggested_goals: bool = True,
        goal_limit: int = 5,
        gap_limit: int = 8,
        world_change_limit: int = 8,
        entity_limit: int = 8,
    ) -> dict[str, Any]:
        active_goals = await self._load_active_goals(limit=goal_limit)
        knowledge_gaps = await self._load_open_gaps(limit=gap_limit)
        if include_suggested_goals:
            suggested_goals = await self._suggest_goals(knowledge_gaps)
        else:
            suggested_goals = []
        world_state = await self._load_world_state(
            world_change_limit=world_change_limit,
            entity_limit=entity_limit,
        )

        return {
            "active_goals": active_goals,
            "knowledge_gaps": knowledge_gaps,
            "suggested_goals": suggested_goals,
            "world_state": world_state,
            "loop_health": {
                "active_goal_count": len(active_goals),
                "knowledge_gap_count": len(knowledge_gaps),
                "suggested_goal_count": len(suggested_goals),
                "world_change_count": len(world_state.get("recent_changes") or []),
                "highlighted_entity_count": len(world_state.get("highlighted_entities") or []),
            },
        }

    async def _load_active_goals(self, *, limit: int) -> list[dict[str, Any]]:
        stmt = (
            select(AutonomousGoal)
            .where(
                AutonomousGoal.organization_id == self.org_id,
                AutonomousGoal.status.in_(["active", "proposed"]),
            )
            .order_by(
                desc(AutonomousGoal.urgency),
                desc(AutonomousGoal.expected_value),
                desc(AutonomousGoal.created_at),
            )
            .limit(max(1, int(limit or 1)))
        )
        goals = (await self.session.execute(stmt)).scalars().all()
        return [
            {
                "goal_id": str(goal.id),
                "title": goal.title,
                "description": goal.description,
                "initiator": goal.initiator,
                "status": goal.status,
                "urgency": float(goal.urgency or 0.0),
                "expected_value": float(goal.expected_value or 0.0),
                "confidence": float(goal.confidence or 0.0),
                "trigger_memory_ids": list(goal.trigger_memory_ids or []),
                "metadata": dict(goal.meta or {}),
            }
            for goal in goals
        ]

    async def _load_open_gaps(self, *, limit: int) -> list[dict[str, Any]]:
        stmt = (
            select(KnowledgeGap)
            .where(
                or_(
                    KnowledgeGap.organization_id == self.org_id,
                    KnowledgeGap.org_id == self.org_id,
                ),
                KnowledgeGap.resolved_at.is_(None),
            )
            .order_by(
                desc(KnowledgeGap.discovered_at),
                desc(KnowledgeGap.created_at),
            )
            .limit(max(1, int(limit or 1)))
        )
        gaps = (await self.session.execute(stmt)).scalars().all()
        serialized = [self._serialize_gap(gap) for gap in gaps]

        if len(serialized) < limit:
            derived = await self._derive_contradiction_gaps(limit=limit - len(serialized))
            seen = {item["description"] for item in serialized}
            for item in derived:
                if item["description"] in seen:
                    continue
                serialized.append(item)
                seen.add(item["description"])
                if len(serialized) >= limit:
                    break

        return serialized[: max(1, int(limit or 1))]

    async def _derive_contradiction_gaps(self, *, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        fact_a_model = aliased(MemoryFact)
        fact_b_model = aliased(MemoryFact)
        stmt = (
            select(Contradiction, fact_a_model, fact_b_model)
            .join(fact_a_model, fact_a_model.id == Contradiction.fact_a)
            .join(fact_b_model, fact_b_model.id == Contradiction.fact_b)
            .where(
                Contradiction.organization_id == self.org_id,
                Contradiction.resolved_at.is_(None),
            )
            .order_by(desc(Contradiction.created_at))
            .limit(max(1, int(limit or 1)))
        )
        rows = (await self.session.execute(stmt)).all()
        derived: list[dict[str, Any]] = []
        for contradiction, fact_a, fact_b in rows:
            derived.append(
                {
                    "gap_id": f"derived:{contradiction.id}",
                    "gap_type": "contradiction",
                    "domain": "memory_facts",
                    "description": contradiction.reason,
                    "confidence_in_gap": 0.85,
                    "priority": "high",
                    "status": "open",
                    "related_memories": [
                        str(fact_a.source_memory_id),
                        str(fact_b.source_memory_id),
                    ],
                    "suggested_learning_approach": "resolve_conflict",
                    "metadata": {
                        "derived": True,
                        "contradiction_id": str(contradiction.id),
                        "fact_a": str(contradiction.fact_a),
                        "fact_b": str(contradiction.fact_b),
                    },
                }
            )
        return derived

    async def _suggest_goals(self, knowledge_gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not knowledge_gaps:
            return []

        svc = IntrinsicMotivationService(self.session)
        goals = await svc.generate_curiosity_goals(self.org_id, knowledge_gaps[:5])
        suggested: list[dict[str, Any]] = []
        for goal in goals[:5]:
            estimated_value = await svc.estimate_goal_value(goal)
            suggested.append(
                {
                    "goal_id": str(goal.get("id") or ""),
                    "title": goal.get("title"),
                    "description": goal.get("description"),
                    "initiator": goal.get("initiator"),
                    "urgency": float(goal.get("urgency") or 0.0),
                    "confidence": float(goal.get("confidence") or 0.0),
                    "expected_value": float(goal.get("expected_value") or 0.0),
                    "estimated_value": float(estimated_value or 0.0),
                    "trigger_memory_ids": list(goal.get("trigger_memory_ids") or []),
                    "metadata": dict(goal.get("metadata") or {}),
                }
            )
        return suggested

    async def _load_world_state(
        self,
        *,
        world_change_limit: int,
        entity_limit: int,
    ) -> dict[str, Any]:
        stmt = (
            select(AgentRun)
            .where(
                AgentRun.organization_id == self.org_id,
                AgentRun.agent_name == self.WORLD_MODEL_AGENT,
                AgentRun.status == "success",
            )
            .order_by(desc(AgentRun.finished_at))
            .limit(25)
        )
        runs = (await self.session.execute(stmt)).scalars().all()
        if not runs:
            return {
                "recent_changes": [],
                "highlighted_entities": [],
                "last_world_model_run_at": None,
            }

        changes: list[dict[str, Any]] = []
        entities: dict[str, dict[str, Any]] = {}
        last_run_at = None
        for run in runs:
            if last_run_at is None and getattr(run, "finished_at", None) is not None:
                last_run_at = run.finished_at
            outputs = dict(run.outputs or {})
            for change in outputs.get("state_changes") or []:
                if not isinstance(change, dict):
                    continue
                entity = str(change.get("entity") or "").strip()
                change_type = str(change.get("change_type") or "").strip()
                if not entity or not change_type:
                    continue
                changes.append(
                    {
                        "entity": entity,
                        "change_type": change_type,
                        "description": change.get("description") or change.get("reason"),
                        "memory_id": str(run.memory_id),
                    }
                )
            for node in outputs.get("world_nodes") or []:
                if not isinstance(node, dict):
                    continue
                entity = str(node.get("entity") or "").strip()
                if not entity:
                    continue
                existing = entities.get(entity)
                confidence = float(node.get("node_confidence") or 0.0)
                if existing is None or confidence > float(existing.get("node_confidence") or 0.0):
                    entities[entity] = {
                        "entity": entity,
                        "entity_type": node.get("entity_type") or "concept",
                        "domains": list(node.get("domains") or []),
                        "node_confidence": confidence,
                    }

        highlighted_entities = sorted(
            entities.values(),
            key=lambda item: float(item.get("node_confidence") or 0.0),
            reverse=True,
        )[: max(1, int(entity_limit or 1))]

        return {
            "recent_changes": changes[: max(1, int(world_change_limit or 1))],
            "highlighted_entities": highlighted_entities,
            "last_world_model_run_at": last_run_at.isoformat() if last_run_at else None,
        }

    @staticmethod
    def _serialize_gap(gap: KnowledgeGap) -> dict[str, Any]:
        return {
            "gap_id": str(gap.id),
            "gap_type": gap.gap_type,
            "domain": gap.domain,
            "description": gap.description or gap.gap_description,
            "confidence_in_gap": float(gap.confidence_in_gap or 0.0),
            "priority": gap.priority,
            "status": gap.status,
            "related_memories": list(gap.related_memories or []),
            "suggested_learning_approach": gap.suggested_learning_approach,
            "metadata": dict(gap.meta or {}),
        }
