from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.meta_agent import BeliefStore


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class BeliefStoreService:
    async def upsert_belief(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        memory_id: str,
        belief_key: str,
        belief_value: dict,
        confidence: float,
        evidence_memory_ids: list[str] | None = None,
        contradiction_ids: list[str] | None = None,
    ) -> BeliefStore:
        confidence = _clamp01(confidence)
        evidence_memory_ids = evidence_memory_ids or []
        contradiction_ids = contradiction_ids or []

        result = await session.execute(
            select(BeliefStore).where(
                BeliefStore.organization_id == org_id,
                BeliefStore.memory_id == memory_id,
                BeliefStore.belief_key == belief_key,
            )
        )
        existing = result.scalar_one_or_none()

        if existing is None:
            belief = BeliefStore(
                organization_id=org_id,
                memory_id=memory_id,
                belief_key=belief_key,
                belief_value={"current": belief_value, "alternatives": []},
                confidence=confidence,
                evidence_memory_ids=list(evidence_memory_ids),
                contradiction_ids=list(contradiction_ids),
            )
            session.add(belief)
            await session.flush()
            return belief

        # Belief revision: never delete previous belief.
        payload = existing.belief_value or {}
        current = payload.get("current")
        alternatives = list(payload.get("alternatives") or [])

        if float(confidence) > float(existing.confidence or 0.0):
            # Promote new belief; demote old to alternative.
            if current is not None:
                alternatives.append({"value": current, "confidence": float(existing.confidence or 0.0)})
            payload["current"] = belief_value
            payload["alternatives"] = alternatives
            existing.belief_value = payload
            existing.confidence = confidence
        else:
            # Keep current; store as alternative.
            alternatives.append({"value": belief_value, "confidence": float(confidence)})
            payload["alternatives"] = alternatives
            existing.belief_value = payload

        existing.evidence_memory_ids = list(evidence_memory_ids)
        existing.contradiction_ids = list(contradiction_ids)
        await session.flush()
        return existing
