"""Memory importance ranking service (Phase 71)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


class MemoryImportanceRanker:
    REFERENCE_WEIGHT = 0.30
    GOAL_LINK_WEIGHT = 0.25
    RECENCY_WEIGHT = 0.20
    CREDIBILITY_WEIGHT = 0.15
    ACTIVATION_WEIGHT = 0.10

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _parse_dt(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)

    def score(self, *, memory: dict, now: datetime) -> float:
        now_utc = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        created_at = self._parse_dt(memory.get("created_at"))

        days_old = max(0.0, (now_utc - created_at).total_seconds() / 86400.0)
        recency = math.exp(-days_old / 30.0)

        reference_count = max(0, int(memory.get("reference_count", 0) or 0))
        goal_link_count = max(0, int(memory.get("goal_link_count", 0) or 0))

        ref_score = min(1.0, reference_count / 10.0)
        goal_score = min(1.0, goal_link_count / 5.0)

        credibility = self._clamp(self._float_or_default(memory.get("credibility_score"), 0.7))
        activation = self._clamp(self._float_or_default(memory.get("activation"), 0.5))

        total = (
            self.REFERENCE_WEIGHT * ref_score
            + self.GOAL_LINK_WEIGHT * goal_score
            + self.RECENCY_WEIGHT * recency
            + self.CREDIBILITY_WEIGHT * credibility
            + self.ACTIVATION_WEIGHT * activation
        )
        return round(self._clamp(total), 4)

    def rank(self, *, memories: list[dict], now: datetime, limit: int = 10) -> list[dict]:
        lim = max(0, int(limit if limit is not None else 10))
        scored: list[dict] = []

        for memory in memories or []:
            item = dict(memory)
            item["_importance_score"] = self.score(memory=item, now=now)
            scored.append(item)

        scored.sort(key=lambda m: float(m.get("_importance_score", 0.0)), reverse=True)
        return scored[:lim]

    def importance_tier(self, score: float) -> str:
        value = self._clamp(score)
        if value >= 0.8:
            return "critical"
        if value >= 0.6:
            return "important"
        if value >= 0.4:
            return "normal"
        return "archivable"
