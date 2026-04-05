"""Proactive intelligence push service (Feature 24.10).

Scans active sessions and subscribed webhooks, evaluates relevance of recent
org events to current session goals/context, and emits proactive push events
when relevance exceeds a configurable threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.cognitive_session import CognitiveSession
from app.models.event import Event
from app.models.webhook import WebhookSubscription
from app.services.webhook_service import WebhookService


_WORD_MIN = 3
_DEFAULT_PUSH_THRESHOLD = 0.6


@dataclass
class ProactivePushCandidate:
    event_id: str
    event_type: str
    relevance: float
    matched_session_ids: list[str]


def _tokenize(text: str) -> set[str]:
    tokens = []
    for raw in (text or "").lower().replace("_", " ").split():
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if len(cleaned) >= _WORD_MIN:
            tokens.append(cleaned)
    return set(tokens)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class ProactivePushService:
    """Compute and emit proactive webhook pushes for one tenant context."""

    def __init__(self, db: AsyncSession):
        self.db = db

    @staticmethod
    def get_push_threshold(org_settings: dict[str, Any] | None = None) -> float:
        """Resolve push threshold from org settings then global config."""
        org_settings = org_settings or {}
        if "push_threshold" in org_settings:
            val = _safe_float(org_settings.get("push_threshold"), _DEFAULT_PUSH_THRESHOLD)
        else:
            val = _safe_float(getattr(settings, "PROACTIVE_PUSH_THRESHOLD", _DEFAULT_PUSH_THRESHOLD), _DEFAULT_PUSH_THRESHOLD)
        return max(0.0, min(1.0, val))

    @staticmethod
    def compute_relevance(*, event: Event | dict[str, Any], goal: str, context_snapshot: dict[str, Any]) -> float:
        """Compute lexical relevance of an event against goal + context.

        Score components:
        - 70% token overlap between event text and goal/context text
        - 30% event-type intent bonus for operational/high-urgency events
        """
        if isinstance(event, Event):
            event_type = str(event.event_type or "")
            payload = dict(event.payload or {})
        elif isinstance(event, dict):
            event_type = str(event.get("event_type") or "")
            payload = dict(event.get("payload") or {})
        else:
            event_type = str(getattr(event, "event_type", "") or "")
            payload = dict(getattr(event, "payload", {}) or {})

        event_text = " ".join([
            event_type,
            str(payload.get("title") or ""),
            str(payload.get("summary") or ""),
            str(payload.get("content") or ""),
            str(payload.get("description") or ""),
        ])

        goal_text = " ".join([
            str(goal or ""),
            str(context_snapshot.get("prior_decision") or ""),
            str(context_snapshot.get("focus") or ""),
            str(context_snapshot.get("goal") or ""),
        ])

        event_tokens = _tokenize(event_text)
        goal_tokens = _tokenize(goal_text)

        if not event_tokens or not goal_tokens:
            lexical = 0.0
        else:
            lexical = len(event_tokens & goal_tokens) / len(event_tokens | goal_tokens)

        lowered_type = event_type.lower()
        if any(t in lowered_type for t in ("critical", "incident", "breach", "outage")):
            intent_bonus = 1.0
        elif any(t in lowered_type for t in ("memory", "knowledge", "review", "alert")):
            intent_bonus = 0.6
        else:
            intent_bonus = 0.2

        score = 0.70 * lexical + 0.30 * intent_bonus
        return round(max(0.0, min(1.0, score)), 4)

    async def scan_active_sessions(self, *, lookback_minutes: int = 180) -> list[CognitiveSession]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(lookback_minutes)))
        result = await self.db.execute(
            select(CognitiveSession).where(
                CognitiveSession.status == "running",
                CognitiveSession.updated_at >= cutoff,
            )
        )
        return list(result.scalars().all())

    async def scan_subscribed_webhooks(self) -> list[WebhookSubscription]:
        result = await self.db.execute(
            select(WebhookSubscription).where(WebhookSubscription.is_active.is_(True))
        )
        return list(result.scalars().all())

    async def scan_recent_events(self, *, lookback_minutes: int = 30, limit: int = 200) -> list[Event]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(lookback_minutes)))
        result = await self.db.execute(
            select(Event)
            .where(Event.created_at >= cutoff)
            .order_by(Event.created_at.desc())
            .limit(max(1, int(limit)))
        )
        return list(result.scalars().all())

    async def build_candidates(
        self,
        *,
        sessions: list[CognitiveSession],
        events: list[Event],
        threshold: float,
    ) -> list[ProactivePushCandidate]:
        candidates: list[ProactivePushCandidate] = []
        for ev in events:
            best = 0.0
            matched: list[str] = []
            for sess in sessions:
                score = self.compute_relevance(
                    event=ev,
                    goal=str(sess.goal or ""),
                    context_snapshot=dict(sess.context_snapshot or {}),
                )
                if score >= threshold:
                    matched.append(str(sess.id))
                if score > best:
                    best = score
            if matched:
                candidates.append(
                    ProactivePushCandidate(
                        event_id=str(ev.id),
                        event_type=str(ev.event_type),
                        relevance=round(best, 4),
                        matched_session_ids=matched,
                    )
                )
        candidates.sort(key=lambda c: c.relevance, reverse=True)
        return candidates

    async def emit_proactive_pushes(
        self,
        *,
        organization_id: str,
        candidates: list[ProactivePushCandidate],
        max_pushes: int = 25,
    ) -> int:
        """Emit proactive push events via outbox for subscribed webhooks."""
        count = 0
        webhook_service = WebhookService(self.db)
        for candidate in candidates[: max(1, int(max_pushes))]:
            await webhook_service.emit_event(
                organization_id=organization_id,
                event_type="cognitive.proactive_push",
                payload={
                    "source_event_id": candidate.event_id,
                    "source_event_type": candidate.event_type,
                    "relevance": candidate.relevance,
                    "matched_session_ids": candidate.matched_session_ids,
                    "emitted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            count += 1
        return count

    async def run_cycle(
        self,
        *,
        organization_id: str,
        push_threshold: float,
        session_lookback_minutes: int = 180,
        event_lookback_minutes: int = 30,
        max_pushes: int = 25,
    ) -> dict[str, Any]:
        """Full proactive push cycle for one org.

        Requires tenant context to be set on the DB session.
        """
        sessions = await self.scan_active_sessions(lookback_minutes=session_lookback_minutes)
        subs = await self.scan_subscribed_webhooks()
        if not sessions or not subs:
            return {
                "organization_id": organization_id,
                "active_sessions": len(sessions),
                "active_webhooks": len(subs),
                "recent_events": 0,
                "candidates": 0,
                "pushed": 0,
                "push_threshold": push_threshold,
            }

        events = await self.scan_recent_events(lookback_minutes=event_lookback_minutes)
        candidates = await self.build_candidates(
            sessions=sessions,
            events=events,
            threshold=push_threshold,
        )

        # Emit as org-scoped outbox events for existing webhook pipeline.
        emitted = 0
        webhook_service = WebhookService(self.db)
        for candidate in candidates[: max(1, int(max_pushes))]:
            await webhook_service.emit_event(
                organization_id=organization_id,
                event_type="cognitive.proactive_push",
                payload={
                    "source_event_id": candidate.event_id,
                    "source_event_type": candidate.event_type,
                    "relevance": candidate.relevance,
                    "matched_session_ids": candidate.matched_session_ids,
                    "push_threshold": push_threshold,
                    "emitted_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            emitted += 1

        return {
            "organization_id": organization_id,
            "active_sessions": len(sessions),
            "active_webhooks": len(subs),
            "recent_events": len(events),
            "candidates": len(candidates),
            "pushed": emitted,
            "push_threshold": push_threshold,
        }
