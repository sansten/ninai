"""Persona profile service (Phase 52)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.persona_profile import PersonaProfile


_EMA_ALPHA = 0.2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_ema_for_level(level: str) -> float:
    norm = str(level or "intermediate").lower()
    if norm == "novice":
        return 0.2
    if norm == "expert":
        return 0.8
    return 0.5


def _level_from_ema(value: float) -> str:
    if value < 0.35:
        return "novice"
    if value < 0.7:
        return "intermediate"
    return "expert"


def _verbosity_from_signal(signal: dict[str, Any]) -> str:
    requested_detail = bool(signal.get("requested_detail", False))
    query_length = int(signal.get("query_length") or 0)
    if requested_detail:
        return "detailed"
    if query_length <= 24:
        return "brief"
    return "normal"


def _signal_to_expertise_target(signal: dict[str, Any]) -> float:
    query_length = int(signal.get("query_length") or 0)
    used_jargon = bool(signal.get("used_jargon", False))
    requested_detail = bool(signal.get("requested_detail", False))

    score = 0.1
    if query_length >= 120:
        score += 0.2
    if used_jargon:
        score += 0.4
    if requested_detail:
        score += 0.3
    if query_length >= 220:
        score += 0.1
    return max(0.0, min(1.0, score))


class PersonaProfileService:
    async def get_or_create(self, *, db: AsyncSession, user_id: str, org_id: str) -> PersonaProfile:
        res = await db.execute(
            select(PersonaProfile).where(
                PersonaProfile.user_id == user_id,
                PersonaProfile.org_id == org_id,
            )
        )
        profile = res.scalar_one_or_none()
        if profile is not None:
            return profile

        profile = PersonaProfile(
            user_id=user_id,
            org_id=org_id,
            expertise_level="intermediate",
            preferred_verbosity="normal",
            domain_vocabulary={"acronyms": [], "preferred_terms": {}, "_expertise_ema": 0.5},
            interaction_count=0,
            last_updated=_utc_now(),
        )
        db.add(profile)
        await db.flush()
        return profile

    async def update_from_interaction(
        self,
        *,
        db: AsyncSession,
        user_id: str,
        org_id: str,
        signal: dict[str, Any],
    ) -> PersonaProfile:
        profile = await self.get_or_create(db=db, user_id=user_id, org_id=org_id)

        vocab = dict(profile.domain_vocabulary or {})
        prev_ema = float(vocab.get("_expertise_ema", _default_ema_for_level(profile.expertise_level)))
        target = _signal_to_expertise_target(signal)
        new_ema = (1.0 - _EMA_ALPHA) * prev_ema + _EMA_ALPHA * target

        profile.expertise_level = _level_from_ema(new_ema)
        profile.preferred_verbosity = _verbosity_from_signal(signal)
        profile.interaction_count = int(profile.interaction_count or 0) + 1

        if bool(signal.get("used_jargon", False)):
            acronyms = list(vocab.get("acronyms") or [])
            if "domain-jargon" not in acronyms:
                acronyms.append("domain-jargon")
            vocab["acronyms"] = acronyms

        vocab.setdefault("preferred_terms", {})
        vocab["_expertise_ema"] = round(new_ema, 4)
        profile.domain_vocabulary = vocab
        profile.last_updated = _utc_now()

        await db.flush()
        return profile

    async def get_style_hints(self, *, db: AsyncSession, user_id: str, org_id: str) -> dict[str, Any]:
        profile = await self.get_or_create(db=db, user_id=user_id, org_id=org_id)
        level = str(profile.expertise_level or "intermediate").lower()
        verbosity = str(profile.preferred_verbosity or "normal").lower()
        vocab = dict(profile.domain_vocabulary or {})

        if level == "novice":
            tone = "supportive"
        elif level == "expert":
            tone = "technical"
        else:
            tone = "neutral"

        vocab_hints = list(vocab.get("acronyms") or [])
        preferred_terms = vocab.get("preferred_terms") or {}
        vocab_hints.extend([str(k) for k in preferred_terms.keys()])

        return {
            "tone": tone,
            "verbosity": verbosity,
            "vocabulary_hints": vocab_hints[:20],
        }