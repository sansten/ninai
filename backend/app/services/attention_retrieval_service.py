"""Attention-weighted retrieval service (Phase 58)."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any


class AttentionRetrievalService:
    GOAL_WEIGHT = 0.4
    INCIDENT_WEIGHT = 0.3
    RECENCY_WEIGHT = 0.2
    BASE_RELEVANCE_WEIGHT = 0.1

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(score):
            return None
        return max(0.0, min(1.0, score))

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\b[a-z0-9_]+\b", (text or "").lower()))

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        if not set_a and not set_b:
            return 1.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    @staticmethod
    def _days_old(created_at: datetime, now: datetime) -> int:
        created = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
        current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        delta_days = (current - created).days
        return max(0, delta_days)

    def score(
        self,
        *,
        memory: dict,
        active_goals: list[dict],
        active_incidents: list[dict],
        query_tokens: frozenset[str],
        now: datetime,
    ) -> float:
        tags = {str(t).strip().lower() for t in (memory.get("tags") or []) if str(t).strip()}
        mem_text = str(memory.get("content") or memory.get("content_preview") or "")
        mem_tokens = self._tokenize(mem_text)

        goal_score = 0.0
        for goal in active_goals or []:
            goal_tags = {str(t).strip().lower() for t in (goal.get("tags") or []) if str(t).strip()}
            goal_score = max(goal_score, self._jaccard(tags, goal_tags))

        incident_score = 0.0
        for incident in active_incidents or []:
            incident_tokens = self._tokenize(str(incident.get("content") or ""))
            if not incident_tokens and incident.get("tags"):
                incident_tokens = {str(t).strip().lower() for t in (incident.get("tags") or []) if str(t).strip()}
            incident_score = max(incident_score, self._jaccard(mem_tokens, incident_tokens))

        created_at = memory.get("created_at")
        if isinstance(created_at, str):
            text = created_at.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                created_at = datetime.fromisoformat(text)
            except ValueError:
                created_at = now
        if not isinstance(created_at, datetime):
            created_at = now

        days_old = self._days_old(created_at, now)
        recency_score = math.exp(-days_old / 30)

        relevance_score = self._jaccard(mem_tokens, set(query_tokens or frozenset()))
        retrieval_score = self._coerce_score(
            memory.get("_retrieval_score", memory.get("score"))
        )
        if retrieval_score is not None:
            relevance_score = max(relevance_score, retrieval_score)

        total = (
            self.GOAL_WEIGHT * goal_score
            + self.INCIDENT_WEIGHT * incident_score
            + self.RECENCY_WEIGHT * recency_score
            + self.BASE_RELEVANCE_WEIGHT * relevance_score
        )
        return max(0.0, min(1.0, round(total, 6)))

    def rank(
        self,
        *,
        memories: list[dict],
        active_goals: list[dict],
        active_incidents: list[dict],
        query_tokens: frozenset[str],
        now: datetime,
        limit: int = 10,
    ) -> list[dict]:
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for idx, memory in enumerate(memories or []):
            s = self.score(
                memory=memory,
                active_goals=active_goals,
                active_incidents=active_incidents,
                query_tokens=query_tokens,
                now=now,
            )
            m = dict(memory)
            m["_attention_score"] = s
            scored.append((idx, s, m))

        scored.sort(key=lambda item: (-item[1], item[0]))
        safe_limit = max(0, int(limit))
        return [item[2] for item in scored[:safe_limit]]
