"""Cognitive forget service (Feature 24.11).

Implements selective unlearning flow for a subject within an organization:
- marks associated memories for cascade deletion (soft delete marker + metadata)
- invalidates linked causal edges
- recomputes freshness/credibility annotations for affected memories
- emits a knowledge.erased event
- returns an erasure certificate for compliance
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.credibility_agent import (
    classify_source_tier,
    compute_credibility_score,
)
from app.agents.memory_decay_agent import (
    compute_base_freshness,
    compute_decay_rate,
    domain_half_life,
)
from app.models.causal_edge import CausalEdge
from app.models.memory import MemoryMetadata
from app.models.user import User
from app.services.webhook_service import WebhookService


@dataclass
class ForgetResult:
    certificate_id: str
    organization_id: str
    subject: str
    reason: str
    domains: list[str]
    erased_memory_count: int
    invalidated_causal_edges: int
    recomputed_memory_count: int
    knowledge_erased_event_emitted: bool
    generated_at: str


class CognitiveForgetService:
    """Feature 24.11 business logic for POST /cognitive/forget."""

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def _normalize_domains(domains: list[str] | None) -> list[str]:
        cleaned = [str(d).strip().lower() for d in (domains or []) if str(d).strip()]
        # stable dedupe
        seen: set[str] = set()
        out: list[str] = []
        for d in cleaned:
            if d in seen:
                continue
            seen.add(d)
            out.append(d)
        return out

    @staticmethod
    def _memory_matches_domains(memory: MemoryMetadata, domains: list[str]) -> bool:
        if not domains:
            return True

        memory_domain = str(getattr(memory, "business_domain", "") or "").strip().lower()
        if memory_domain and memory_domain in domains:
            return True

        tags = [str(t).strip().lower() for t in (getattr(memory, "tags", None) or []) if str(t).strip()]
        if any(tag in domains for tag in tags):
            return True

        extra = dict(getattr(memory, "extra_metadata", None) or {})
        extra_domain = str(extra.get("domain") or "").strip().lower()
        return bool(extra_domain and extra_domain in domains)

    async def _resolve_subject_user_ids(self, *, subject: str, organization_id: str) -> list[str]:
        s = str(subject or "").strip().lower()
        if not s:
            return []

        stmt = select(User.id).where(User.email.ilike(s))
        ids = [str(v) for v in (await self.db.execute(stmt)).scalars().all()]

        # Support passing a direct user_id as subject.
        if not ids:
            direct = select(User.id).where(User.id == s)
            ids = [str(v) for v in (await self.db.execute(direct)).scalars().all()]

        return ids

    async def _find_associated_memories(
        self,
        *,
        organization_id: str,
        subject: str,
        subject_user_ids: list[str],
        domains: list[str],
    ) -> list[MemoryMetadata]:
        stmt = select(MemoryMetadata).where(
            MemoryMetadata.organization_id == organization_id,
            MemoryMetadata.is_active.is_(True),
        )
        rows = list((await self.db.execute(stmt)).scalars().all())

        s_lower = str(subject or "").strip().lower()
        matched: list[MemoryMetadata] = []
        for mem in rows:
            owner_match = str(getattr(mem, "owner_id", "")) in set(subject_user_ids)
            preview = str(getattr(mem, "content_preview", "") or "").lower()
            extra = dict(getattr(mem, "extra_metadata", None) or {})
            content_match = bool(s_lower and (s_lower in preview or s_lower in str(extra).lower()))

            if not (owner_match or content_match):
                continue
            if not self._memory_matches_domains(mem, domains):
                continue
            matched.append(mem)

        return matched

    async def _mark_memories_for_cascade_deletion(
        self,
        *,
        memories: list[MemoryMetadata],
        subject: str,
        reason: str,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for mem in memories:
            meta = dict(mem.extra_metadata or {})
            meta["forget_marker"] = {
                "subject": subject,
                "reason": reason,
                "marked_at": now,
                "cascade_delete": True,
            }
            mem.extra_metadata = meta
            mem.is_active = False
            count += 1
        await self.db.flush()
        return count

    async def _invalidate_linked_causal_edges(
        self,
        *,
        organization_id: str,
        memory_ids: list[str],
    ) -> int:
        if not memory_ids:
            return 0

        ids = set(memory_ids)
        stmt = select(CausalEdge).where(CausalEdge.organization_id == organization_id)
        edges = list((await self.db.execute(stmt)).scalars().all())

        touched = 0
        now = datetime.now(timezone.utc)
        for edge in edges:
            evidence_ids = {str(v) for v in (edge.evidence_memory_ids or [])}
            linked = (
                str(edge.cause_entity_id) in ids
                or str(edge.effect_entity_id) in ids
                or bool(evidence_ids & ids)
            )
            if not linked:
                continue

            edge.invalidation_count = int(edge.invalidation_count or 0) + 1
            edge.strength = max(0.0, float(edge.strength or 0.0) - 0.5)
            edge.last_validated_at = now
            touched += 1

        await self.db.flush()
        return touched

    async def _recompute_affected_scores(self, *, memories: list[MemoryMetadata]) -> int:
        now = datetime.now(timezone.utc)
        count = 0

        for mem in memories:
            meta = dict(mem.extra_metadata or {})
            content = str(mem.content_preview or "")
            source_type = str(mem.source_type or "")
            author_role = str(meta.get("author_role") or "")

            source_tier = classify_source_tier(
                source_type=source_type,
                author_role=author_role,
                content=content,
            )
            credibility_score = compute_credibility_score(
                source_tier=source_tier,
                citation_depth=0,
                corroboration_count=0,
                high_severity_conflict_count=0,
            )

            domain = str(getattr(mem, "business_domain", "") or meta.get("domain") or "general")
            half_life = domain_half_life(domain)
            decay_rate = compute_decay_rate(half_life=half_life)
            age_days = max((now - (mem.created_at or now)).total_seconds() / 86400.0, 0.0)
            freshness_score = compute_base_freshness(age_days=age_days, decay_rate=decay_rate)

            meta["recomputed_after_forget"] = {
                "credibility_score": credibility_score,
                "source_tier": source_tier,
                "freshness_score": freshness_score,
                "decay_rate": decay_rate,
                "age_days": round(age_days, 2),
                "at": now.isoformat(),
            }
            mem.extra_metadata = meta
            count += 1

        await self.db.flush()
        return count

    async def _emit_knowledge_erased_event(
        self,
        *,
        organization_id: str,
        certificate: ForgetResult,
        memory_ids: list[str],
    ) -> bool:
        try:
            svc = WebhookService(self.db)
            await svc.emit_event(
                organization_id=organization_id,
                event_type="knowledge.erased",
                payload={
                    "certificate_id": certificate.certificate_id,
                    "subject": certificate.subject,
                    "reason": certificate.reason,
                    "domains": certificate.domains,
                    "erased_memory_count": certificate.erased_memory_count,
                    "invalidated_causal_edges": certificate.invalidated_causal_edges,
                    "memory_ids": memory_ids,
                    "generated_at": certificate.generated_at,
                },
            )
            return True
        except Exception:
            return False

    async def forget(
        self,
        *,
        organization_id: str,
        subject: str,
        domains: list[str] | None,
        reason: str,
        requested_by_user_id: str,
    ) -> ForgetResult:
        """Run selective unlearning for a subject and return certificate."""
        del requested_by_user_id  # kept for audit extension compatibility

        norm_domains = self._normalize_domains(domains)
        subject_user_ids = await self._resolve_subject_user_ids(
            subject=subject,
            organization_id=organization_id,
        )
        memories = await self._find_associated_memories(
            organization_id=organization_id,
            subject=subject,
            subject_user_ids=subject_user_ids,
            domains=norm_domains,
        )
        memory_ids = [str(m.id) for m in memories]

        erased_count = await self._mark_memories_for_cascade_deletion(
            memories=memories,
            subject=subject,
            reason=reason,
        )
        invalidated_edges = await self._invalidate_linked_causal_edges(
            organization_id=organization_id,
            memory_ids=memory_ids,
        )
        recomputed = await self._recompute_affected_scores(memories=memories)

        certificate = ForgetResult(
            certificate_id=str(uuid.uuid4()),
            organization_id=organization_id,
            subject=subject,
            reason=reason,
            domains=norm_domains,
            erased_memory_count=erased_count,
            invalidated_causal_edges=invalidated_edges,
            recomputed_memory_count=recomputed,
            knowledge_erased_event_emitted=False,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        emitted = await self._emit_knowledge_erased_event(
            organization_id=organization_id,
            certificate=certificate,
            memory_ids=memory_ids,
        )
        certificate.knowledge_erased_event_emitted = emitted

        await self.db.commit()
        return certificate
