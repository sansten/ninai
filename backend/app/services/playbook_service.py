"""Playbook retrieval service (PR4)."""

from __future__ import annotations

import re

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.playbook import Playbook


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_\-]+", (text or "").lower()))


class PlaybookService:
    def __init__(self, session: AsyncSession, org_id: str):
        self.session = session
        self.org_id = org_id

    async def search_playbooks(self, *, query: str, limit: int = 5) -> list[dict]:
        stmt = (
            select(Playbook)
            .where(Playbook.organization_id == self.org_id)
            .order_by(desc(Playbook.success_rate), desc(Playbook.updated_at))
            .limit(100)
        )
        candidates = list((await self.session.execute(stmt)).scalars().all())
        q_tokens = _tokens(query)

        scored = []
        for pb in candidates:
            sig = pb.problem_signature or {}
            sig_tokens = set(sig.get("tokens", []))
            title_tokens = _tokens(pb.title)
            overlap = len(q_tokens.intersection(sig_tokens.union(title_tokens)))
            score = overlap + float(pb.success_rate or 0.0)
            scored.append((score, pb))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [pb for score, pb in scored[:limit] if score > 0]

        return [
            {
                "id": pb.id,
                "title": pb.title,
                "scope_type": str(pb.scope_type),
                "scope_id": pb.scope_id,
                "problem_signature": pb.problem_signature,
                "steps": pb.steps,
                "constraints": pb.constraints,
                "success_rate": pb.success_rate,
                "evidence": pb.evidence,
                "confidence": min(1.0, max(0.0, float(pb.success_rate or 0.0))),
            }
            for pb in top
        ]
