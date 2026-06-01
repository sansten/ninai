from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory import MemoryMetadata
from app.models.memory_state_snapshot import MemoryStateSnapshot
from app.models.memory_state_stream import MemoryStateStream


class StateSpaceService:
    """Incremental symbolic state snapshots for deterministic reads."""

    MAX_FACTS_PER_SCOPE = 24
    MAX_RECENT_MEMORIES = 10
    MAX_FACTS_PER_CANDIDATE = 4

    def __init__(self, session: AsyncSession, *, org_id: str) -> None:
        self.session = session
        self.org_id = org_id

    async def apply_memory_write_state(
        self,
        *,
        memory: MemoryMetadata,
        fact_rows: list[dict[str, Any]],
        resolved_entities: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        resolved_entities = list(resolved_entities or [])
        scopes = self._scopes_for_memory(memory=memory, fact_rows=fact_rows, resolved_entities=resolved_entities)
        updated_scopes = 0
        stream_events = 0

        for scope_type, scope_key in scopes:
            scoped_facts = self._facts_for_scope(fact_rows=fact_rows, scope_type=scope_type, scope_key=scope_key)
            stream = MemoryStateStream(
                organization_id=self.org_id,
                source_memory_id=str(memory.id),
                scope_type=scope_type,
                scope_key=scope_key,
                event_type="memory_write",
                event_summary=self._event_summary(memory=memory, facts=scoped_facts),
                event_features=self._event_features(memory=memory, facts=scoped_facts),
                fact_delta=scoped_facts,
            )
            self.session.add(stream)
            stream_events += 1

            snapshot = await self._get_or_create_snapshot(scope_type=scope_type, scope_key=scope_key)
            symbolic_state = self._merge_symbolic_state(
                existing=dict(snapshot.symbolic_state or {}),
                memory=memory,
                scoped_facts=scoped_facts,
                scope_type=scope_type,
                scope_key=scope_key,
                resolved_entities=resolved_entities,
            )
            snapshot.symbolic_state = symbolic_state
            snapshot.summary_text = self._summary_text(symbolic_state)
            snapshot.salience_score = self._salience(symbolic_state)
            snapshot.staleness_score = 0.0
            snapshot.last_memory_id = str(memory.id)
            snapshot.last_event_at = memory.occurred_at or getattr(memory, "updated_at", None) or datetime.now(timezone.utc)
            snapshot.state_version = int(snapshot.state_version or 0) + 1
            updated_scopes += 1

        return {
            "updated_scopes": updated_scopes,
            "stream_events": stream_events,
        }

    async def lookup_candidates(
        self,
        *,
        query: str,
        intelligence: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        entities = self._query_entities(query, intelligence)
        tokens = self._query_tokens(query, limit=10)
        if not entities and not tokens:
            return []

        disjuncts = []
        for entity in entities[:4]:
            normalized = self._normalize_scope_key(entity)
            if normalized:
                disjuncts.append(
                    and_(
                        MemoryStateSnapshot.scope_type == "entity",
                        MemoryStateSnapshot.scope_key == normalized,
                    )
                )
        for token in tokens[:3]:
            disjuncts.append(MemoryStateSnapshot.summary_text.ilike(f"%{token}%"))
        if not disjuncts:
            return []

        stmt = (
            select(MemoryStateSnapshot)
            .where(
                MemoryStateSnapshot.organization_id == self.org_id,
                or_(*disjuncts),
            )
            .order_by(MemoryStateSnapshot.updated_at.desc())
            .limit(max(4, int(limit or 1)) * 3)
        )
        rows = list((await self.session.execute(stmt)).scalars().all())
        if not rows:
            return []

        candidates: list[tuple[float, dict[str, Any]]] = []
        for snapshot in rows:
            for candidate in self._snapshot_candidates(snapshot=snapshot, query=query, intelligence=intelligence):
                candidates.append((self._safe_float(candidate.get("score")), candidate))

        candidates.sort(key=lambda item: item[0], reverse=True)
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for _, candidate in candidates:
            candidate_id = str(candidate.get("id") or "").strip()
            if not candidate_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            deduped.append(candidate)
            if len(deduped) >= max(1, int(limit or 1)):
                break
        return deduped

    async def _get_or_create_snapshot(self, *, scope_type: str, scope_key: str) -> MemoryStateSnapshot:
        stmt = select(MemoryStateSnapshot).where(
            MemoryStateSnapshot.organization_id == self.org_id,
            MemoryStateSnapshot.scope_type == scope_type,
            MemoryStateSnapshot.scope_key == scope_key,
        ).limit(1)
        existing = (await self.session.execute(stmt)).scalar_one_or_none()
        if existing is not None:
            return existing
        snapshot = MemoryStateSnapshot(
            organization_id=self.org_id,
            scope_type=scope_type,
            scope_key=scope_key,
            state_version=0,
            symbolic_state={},
            summary_text="",
            salience_score=0.0,
            staleness_score=0.0,
        )
        self.session.add(snapshot)
        await self.session.flush()
        return snapshot

    def _scopes_for_memory(
        self,
        *,
        memory: MemoryMetadata,
        fact_rows: list[dict[str, Any]],
        resolved_entities: list[dict[str, Any]],
    ) -> list[tuple[str, str]]:
        scopes: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _add(scope_type: str, scope_key: str | None) -> None:
            normalized_key = self._normalize_scope_key(scope_key)
            item = (str(scope_type or "").strip().lower(), normalized_key)
            if not item[0] or not item[1] or item in seen:
                return
            seen.add(item)
            scopes.append(item)

        _add("org", self.org_id)
        _add("user", memory.owner_id)

        extra = dict(memory.extra_metadata or {})
        _add("conversation", extra.get("gateway_context_id") or memory.source_id)

        for fact in fact_rows:
            _add("entity", fact.get("subject"))
            obj = str(fact.get("object") or "").strip()
            if self._looks_named_entity(obj):
                _add("entity", obj)

        for entity in resolved_entities:
            canonical = str(entity.get("canonical") or "").strip()
            subject = str(entity.get("subject") or "").strip()
            _add("entity", canonical or subject)

        return scopes

    def _facts_for_scope(
        self,
        *,
        fact_rows: list[dict[str, Any]],
        scope_type: str,
        scope_key: str,
    ) -> list[dict[str, Any]]:
        if scope_type not in {"entity", "conversation", "user", "org"}:
            return []
        if scope_type == "entity":
            return [
                dict(fact)
                for fact in fact_rows
                if self._normalize_scope_key(fact.get("subject")) == scope_key
                or self._normalize_scope_key(fact.get("object")) == scope_key
            ]
        return [dict(fact) for fact in fact_rows]

    def _event_summary(self, *, memory: MemoryMetadata, facts: list[dict[str, Any]]) -> str:
        if facts:
            top = facts[0]
            return f"{top.get('subject')} {top.get('predicate')} {top.get('object')}"
        return str(memory.content_preview or "")[:180]

    def _event_features(self, *, memory: MemoryMetadata, facts: list[dict[str, Any]]) -> dict[str, Any]:
        extra = dict(memory.extra_metadata or {})
        return {
            "memory_id": str(memory.id),
            "memory_type": memory.memory_type,
            "source_type": memory.source_type,
            "scope": memory.scope,
            "scope_id": memory.scope_id,
            "occurred_at": (memory.occurred_at.isoformat() if memory.occurred_at else extra.get("event_time")),
            "tag_count": len(memory.tags or []),
            "fact_count": len(facts),
        }

    def _merge_symbolic_state(
        self,
        *,
        existing: dict[str, Any],
        memory: MemoryMetadata,
        scoped_facts: list[dict[str, Any]],
        scope_type: str,
        scope_key: str,
        resolved_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state = dict(existing or {})
        facts = list(state.get("facts") or [])
        recent_memories = list(state.get("recent_memories") or [])
        aliases = list(state.get("aliases") or [])

        memory_entry = {
            "memory_id": str(memory.id),
            "title": memory.title,
            "content_preview": str(memory.content_preview or "")[:240],
            "occurred_at": memory.occurred_at.isoformat() if memory.occurred_at else (memory.extra_metadata or {}).get("event_time"),
            "source_type": memory.source_type,
            "tags": list(memory.tags or [])[:8],
        }
        recent_memories = [memory_entry] + [
            item for item in recent_memories
            if str(item.get("memory_id") or "") != str(memory.id)
        ]
        recent_memories = recent_memories[: self.MAX_RECENT_MEMORIES]

        for fact in scoped_facts:
            normalized = {
                "subject": str(fact.get("subject") or "").strip(),
                "predicate": str(fact.get("predicate") or "").strip().lower(),
                "object": str(fact.get("object") or "").strip(),
                "confidence": self._safe_float(fact.get("confidence"), 0.85),
                "status": str(fact.get("status") or "active"),
                "source_memory_id": str(memory.id),
                "valid_from": memory.occurred_at.isoformat() if memory.occurred_at else (memory.extra_metadata or {}).get("event_time"),
            }
            facts = [
                item for item in facts
                if not (
                    self._normalize_scope_key(item.get("subject")) == self._normalize_scope_key(normalized.get("subject"))
                    and str(item.get("predicate") or "").strip().lower() == normalized["predicate"]
                    and str(item.get("object") or "").strip().lower() == normalized["object"].lower()
                )
            ]
            facts.insert(0, normalized)

        if scope_type == "entity":
            for entity in resolved_entities:
                for candidate in (entity.get("canonical"), entity.get("subject")):
                    normalized = str(candidate or "").strip()
                    if not normalized:
                        continue
                    if self._normalize_scope_key(normalized) != scope_key:
                        continue
                    if normalized not in aliases:
                        aliases.append(normalized)

        deduped_facts: list[dict[str, Any]] = []
        seen_fact_keys: set[tuple[str, str, str]] = set()
        for fact in facts:
            key = (
                self._normalize_scope_key(fact.get("subject")),
                str(fact.get("predicate") or "").strip().lower(),
                str(fact.get("object") or "").strip().lower(),
            )
            if key in seen_fact_keys:
                continue
            seen_fact_keys.add(key)
            deduped_facts.append(fact)
            if len(deduped_facts) >= self.MAX_FACTS_PER_SCOPE:
                break

        state.update(
            {
                "scope_type": scope_type,
                "scope_key": scope_key,
                "facts": deduped_facts,
                "recent_memories": recent_memories,
                "aliases": aliases[:8],
                "updated_from_memory_id": str(memory.id),
            }
        )
        return state

    def _summary_text(self, state: dict[str, Any]) -> str:
        parts: list[str] = []
        for fact in list(state.get("facts") or [])[:8]:
            parts.append(
                " ".join(
                    part for part in [
                        str(fact.get("subject") or "").strip(),
                        str(fact.get("predicate") or "").strip(),
                        str(fact.get("object") or "").strip(),
                    ]
                    if part
                )
            )
        for memory in list(state.get("recent_memories") or [])[:3]:
            preview = str(memory.get("content_preview") or "").strip()
            if preview:
                parts.append(preview)
        return " | ".join(part for part in parts if part)[:4000]

    def _salience(self, state: dict[str, Any]) -> float:
        fact_count = len(list(state.get("facts") or []))
        recent_count = len(list(state.get("recent_memories") or []))
        return round(min(1.0, (fact_count * 0.08) + (recent_count * 0.03)), 4)

    def _snapshot_candidates(
        self,
        *,
        snapshot: MemoryStateSnapshot,
        query: str,
        intelligence: dict[str, Any],
    ) -> list[dict[str, Any]]:
        state = dict(snapshot.symbolic_state or {})
        facts = list(state.get("facts") or [])
        if not facts:
            return []

        profile = self._question_profile(query, intelligence)
        scored_facts: list[tuple[float, dict[str, Any]]] = []
        for fact in facts:
            score = self._fact_score(fact=fact, profile=profile)
            if score <= 0:
                continue
            scored_facts.append((score, fact))
        scored_facts.sort(key=lambda item: item[0], reverse=True)

        candidates: list[dict[str, Any]] = []
        for index, (score, fact) in enumerate(scored_facts[: self.MAX_FACTS_PER_CANDIDATE]):
            supporting_facts = [
                item for _, item in scored_facts[: self.MAX_FACTS_PER_CANDIDATE]
                if self._normalize_scope_key(item.get("subject")) == self._normalize_scope_key(fact.get("subject"))
            ]
            preview = f"[State] {fact.get('subject')} {fact.get('predicate')}: {fact.get('object')}"
            candidate_id = f"state::{snapshot.scope_type}::{snapshot.scope_key}::{index}"
            candidates.append(
                {
                    "id": candidate_id,
                    "memory_id": candidate_id,
                    "title": f"State {snapshot.scope_type}:{snapshot.scope_key}",
                    "content_preview": preview,
                    "content": preview,
                    "score": min(0.995, score),
                    "classification": "internal",
                    "source_type": "state_space",
                    "tags": ["state_space", str(fact.get("predicate") or "").strip().lower()],
                    "entities": {},
                    "extra_metadata": {
                        "state_space": {
                            "scope_type": snapshot.scope_type,
                            "scope_key": snapshot.scope_key,
                            "state_version": int(snapshot.state_version or 0),
                            "summary_text": snapshot.summary_text,
                        },
                        "fact_support": dict(fact),
                        "fact_supporting_facts": supporting_facts[: self.MAX_FACTS_PER_CANDIDATE],
                    },
                }
            )
        return candidates

    @classmethod
    def _query_entities(cls, query: str, intelligence: dict[str, Any]) -> list[str]:
        entities: list[str] = []
        seen: set[str] = set()
        for item in list(intelligence.get("extracted_entities") or []):
            value = str(item).strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                entities.append(value)
        for match in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+){0,2}\b", str(query or "")):
            value = match.group(0).strip()
            if value and value.lower() not in seen:
                seen.add(value.lower())
                entities.append(value)
        return entities[:4]

    @classmethod
    def _query_profile(cls, query: str, intelligence: dict[str, Any]) -> dict[str, Any]:
        return cls._question_profile(query, intelligence)

    @classmethod
    def _question_profile(cls, query: str, intelligence: dict[str, Any]) -> dict[str, Any]:
        entities = cls._query_entities(query, intelligence)
        return {
            "entities": [str(item).lower() for item in entities],
            "tokens": cls._query_tokens(query, limit=10),
            "subject": str(entities[0]).lower() if entities else "",
        }

    @classmethod
    def _fact_score(cls, *, fact: dict[str, Any], profile: dict[str, Any]) -> float:
        subject = str(fact.get("subject") or "").strip().lower()
        predicate = str(fact.get("predicate") or "").strip().lower()
        obj = str(fact.get("object") or "").strip().lower()
        score = cls._safe_float(fact.get("confidence"), 0.0)
        if profile.get("subject") and profile["subject"] == subject:
            score += 1.2
        for entity in profile.get("entities") or []:
            if entity == subject:
                score += 1.0
            elif entity in obj:
                score += 0.55
        for token in profile.get("tokens") or []:
            if token == predicate:
                score += 0.7
            elif token in predicate:
                score += 0.35
            elif token in obj or token in subject:
                score += 0.2
        return score

    @staticmethod
    def _query_tokens(query: str, *, limit: int) -> list[str]:
        tokens = re.findall(r"\b[a-z0-9_]{3,}\b", str(query or "").lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= limit:
                break
        return deduped

    @staticmethod
    def _normalize_scope_key(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _looks_named_entity(value: str) -> bool:
        return bool(re.search(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+){0,2}\b", str(value or "")))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
