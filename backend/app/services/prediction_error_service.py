"""PredictionErrorService — Phase 85.

Implements the prediction-error learning signal:

  anticipate()  — before the LLM call, form an expectation from question type +
                  retrieval quality.
  measure()     — after the LLM answer, quantify how surprising it was.
  record()      — persist high-divergence events for priority consolidation.

This turns surprise into a learning signal: the system preferentially
consolidates memories related to what it got wrong, mirroring how biological
memory encoding is strengthened by prediction error (Schultz / Friston PE
theory).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVERGENCE_THRESHOLD = 0.55   # events above this score are recorded
_MIN_SAMPLES_TO_TRUST = 3     # minimum chunk count to form a reliable expectation

_DATE_RE = re.compile(
    r"\b(when|what (year|month|date|day)|how long|how many (years?|months?|days?|weeks?))\b",
    re.IGNORECASE,
)
_BOOL_RE = re.compile(
    r"\b(did|does|is|are|was|were|has|have|had|can|could|will|would)\b.{0,40}\?",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(
    r"\b(who|whose|which (person|company|team|group|role))\b",
    re.IGNORECASE,
)

_REFUSAL_FRAGMENTS = frozenset({
    "not mentioned", "not provided", "not available", "no information",
    "cannot determine", "does not contain", "not in the context",
    "not specified", "not stated", "not found", "i don't know",
    "i do not know", "no mention",
})


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Anticipation:
    """Pre-inference expectation formed from question + retrieval quality."""
    category: str           # "entity" | "date" | "boolean" | "narrative"
    confidence: float       # retrieval-coverage proxy, [0, 1]


@dataclass
class PredictionErrorResult:
    divergence_score: float
    actual_category: str
    expected_category: str
    expected_confidence: float
    is_surprising: bool     # divergence_score >= DIVERGENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class PredictionErrorService:
    """Stateless service — all state lives in the DB or is passed as arguments."""

    # ------------------------------------------------------------------
    # Pre-inference
    # ------------------------------------------------------------------

    def anticipate(self, question: str, chunks: list[dict]) -> Anticipation:
        """Form an expectation from question type and retrieval coverage."""
        category = self._classify_expected_category(question)

        q_words = set(question.lower().split())
        stopwords = frozenset({"what", "when", "who", "where", "how", "the", "a", "an",
                                "is", "are", "was", "were", "did", "do", "does"})
        q_content = q_words - stopwords

        ctx_words: set[str] = set()
        for c in chunks[:8]:
            text = (c.get("payload") or {}).get("text") or c.get("text") or ""
            ctx_words.update(text.lower().split())

        if not q_content:
            confidence = 0.30
        else:
            coverage = len(q_content & ctx_words) / len(q_content)
            # Penalty when too few chunks: less evidence = less trustworthy anticipation
            chunk_factor = min(len(chunks) / _MIN_SAMPLES_TO_TRUST, 1.0)
            confidence = min(0.20 + coverage * 0.65 * chunk_factor, 1.0)

        return Anticipation(category=category, confidence=confidence)

    # ------------------------------------------------------------------
    # Post-inference
    # ------------------------------------------------------------------

    def measure(
        self,
        anticipation: Anticipation,
        answer: str,
        answer_confidence: float,
    ) -> PredictionErrorResult:
        """Compute how surprising the answer was relative to the anticipation."""
        ans = (answer or "").strip()
        actual_category = self._classify_actual_category(ans)
        is_refusal = self._is_refusal(ans)

        divergence = 0.0

        # 1. Category mismatch penalty
        if actual_category != anticipation.category and not is_refusal:
            divergence += 0.25

        # 2. Refusal when high coverage was expected
        if is_refusal and anticipation.confidence > 0.55:
            divergence += 0.45

        # 3. Confidence collapse: expected high, got low
        conf_drop = max(0.0, anticipation.confidence - answer_confidence)
        divergence += conf_drop * 0.35

        # 4. Length surprise: expected brief factual but got long narrative
        if anticipation.category in ("entity", "date", "boolean"):
            word_count = len(ans.split())
            if word_count > 40:
                divergence += 0.10
        elif anticipation.category == "narrative" and len(ans.split()) < 5 and not is_refusal:
            divergence += 0.10

        divergence = min(divergence, 1.0)

        return PredictionErrorResult(
            divergence_score=round(divergence, 4),
            actual_category=actual_category,
            expected_category=anticipation.category,
            expected_confidence=anticipation.confidence,
            is_surprising=divergence >= DIVERGENCE_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def record(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        question: str,
        chunks: list[dict],
        result: PredictionErrorResult,
        strategy: str | None = None,
        answer_snippet: str | None = None,
    ) -> None:
        """Persist a high-divergence prediction error event."""
        from app.models.prediction_error_log import PredictionErrorLog

        chunk_ids = [str(c.get("id") or "") for c in chunks[:20] if c.get("id")]

        log = PredictionErrorLog(
            organization_id=org_id,
            query_hash=hashlib.sha256(question.encode()).hexdigest()[:16],
            query_snippet=question[:300],
            expected_category=result.expected_category,
            expected_confidence=result.expected_confidence,
            actual_category=result.actual_category,
            actual_answer_snippet=(answer_snippet or "")[:300],
            divergence_score=result.divergence_score,
            strategy=strategy,
            chunk_ids=chunk_ids,
            consolidated=False,
        )
        session.add(log)
        await session.flush()

    async def load_unconsolidated(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[Any]:
        """Load high-divergence events that haven't been consolidated yet."""
        from app.models.prediction_error_log import PredictionErrorLog

        since = since or (datetime.now(timezone.utc) - timedelta(days=7))
        stmt = (
            select(PredictionErrorLog)
            .where(
                PredictionErrorLog.organization_id == org_id,
                PredictionErrorLog.consolidated.is_(False),
                PredictionErrorLog.created_at >= since,
                PredictionErrorLog.divergence_score >= DIVERGENCE_THRESHOLD,
            )
            .order_by(PredictionErrorLog.divergence_score.desc())
            .limit(limit)
        )
        rows = await session.execute(stmt)
        return list(rows.scalars().all())

    async def mark_consolidated(
        self,
        session: AsyncSession,
        *,
        log_ids: list[str],
    ) -> None:
        """Mark events as processed so the nightly pass doesn't re-process them."""
        from app.models.prediction_error_log import PredictionErrorLog

        if not log_ids:
            return
        await session.execute(
            update(PredictionErrorLog)
            .where(PredictionErrorLog.id.in_(log_ids))
            .values(consolidated=True)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_expected_category(question: str) -> str:
        if _DATE_RE.search(question):
            return "date"
        if _ENTITY_RE.search(question):
            return "entity"
        if _BOOL_RE.search(question):
            return "boolean"
        return "narrative"

    @staticmethod
    def _classify_actual_category(answer: str) -> str:
        ans = answer.lower().strip()
        if not ans:
            return "empty"
        # Short answers with date-like patterns
        if re.search(r"\b\d{4}\b|\b(january|february|march|april|may|june|july|august|"
                     r"september|october|november|december)\b", ans):
            return "date"
        # Very short answers that look like names/entities
        words = ans.split()
        if len(words) <= 4 and re.search(r"[A-Z]", answer):
            return "entity"
        if len(words) <= 6 and ans in ("yes", "no", "true", "false"):
            return "boolean"
        if len(words) <= 6:
            return "entity"
        return "narrative"

    @staticmethod
    def _is_refusal(text: str) -> bool:
        low = text.lower().strip()
        return not low or any(f in low for f in _REFUSAL_FRAGMENTS)
