"""Feedback learning config persistence.

Materializes FeedbackLearningAgent outputs into a per-org config row.

Why this exists:
- The agent implementation guide expects v1 feedback learning to update
  per-org calibration (thresholds/stopwords/heuristic weights).
- The current FeedbackLearningAgent primarily applies pending feedback
  to MemoryMetadata, but we still persist a small calibration delta for
  observability and future tuning.

Tenant safety:
- Callers must pass organization_id.
- Table is protected by Postgres RLS.

Idempotency:
- Single row per org via unique index on organization_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.org_feedback_learning_config import OrgFeedbackLearningConfig


_STOPWORD_RE = re.compile(r"[^a-z0-9_\-]+")


def normalize_stopword(word: str) -> str:
    s = (word or "").strip().lower()
    s = _STOPWORD_RE.sub("", s)
    return s


def normalize_stopwords(words: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for w in words or []:
        if not isinstance(w, str):
            continue
        n = normalize_stopword(w)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    out.sort()
    return out


@dataclass(frozen=True)
class ExtractedCalibration:
    updated_thresholds: dict[str, Any]
    new_stopwords: list[str]
    heuristic_weights: dict[str, Any]
    calibration_delta: dict[str, Any]
    should_update: bool


def _as_dict(val: Any) -> dict[str, Any]:
    return dict(val) if isinstance(val, dict) else {}


def _as_list_str(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for x in val:
        if x is None:
            continue
        out.append(str(x))
    return out


def extract_calibration(outputs: dict[str, Any] | None) -> ExtractedCalibration:
    """Extract calibration-related fields from agent outputs.

    Supports the implementation guide schema (updated_thresholds/new_stopwords/
    calibration_delta) while also accepting current v1 outputs (applied,
    applied_count, updates).
    """

    data = outputs if isinstance(outputs, dict) else {}

    updated_thresholds = _as_dict(data.get("updated_thresholds"))
    new_stopwords = normalize_stopwords(_as_list_str(data.get("new_stopwords")))
    heuristic_weights = _as_dict(data.get("heuristic_weights"))

    delta = _as_dict(data.get("calibration_delta"))

    applied = data.get("applied") is True
    applied_count = 0
    try:
        applied_count = int(data.get("applied_count", 0) or 0)
    except Exception:
        applied_count = 0

    updates = data.get("updates")
    counts: dict[str, int] = {}
    if isinstance(updates, list):
        for u in updates:
            if not isinstance(u, dict):
                continue
            t = u.get("type")
            if not isinstance(t, str) or not t:
                continue
            counts[t] = int(counts.get(t, 0)) + 1

    if applied:
        # Compact observability delta (doesn't change thresholds unless provided).
        delta.setdefault("applied_count", applied_count)
        if counts:
            delta.setdefault("feedback_update_counts", counts)
        rationale = data.get("rationale")
        if isinstance(rationale, str) and rationale:
            delta.setdefault("rationale", rationale)

    should_update = bool(updated_thresholds or new_stopwords or heuristic_weights or (applied and (applied_count > 0)))

    return ExtractedCalibration(
        updated_thresholds=updated_thresholds,
        new_stopwords=new_stopwords,
        heuristic_weights=heuristic_weights,
        calibration_delta=delta,
        should_update=should_update,
    )


class FeedbackLearningConfigService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def apply_from_agent_outputs(
        self,
        *,
        organization_id: str,
        source_memory_id: str,
        outputs: dict[str, Any] | None,
        updated_by_user_id: Optional[str],
        agent_version: Optional[str],
        trace_id: Optional[str],
    ) -> dict[str, Any]:
        extracted = extract_calibration(outputs)
        if not extracted.should_update:
            return {"config_updated": False, "delta": {}, "stopwords_added": 0, "thresholds_updated": 0, "weights_updated": 0}

        existing_stmt = select(OrgFeedbackLearningConfig).where(
            OrgFeedbackLearningConfig.organization_id == organization_id,
        )
        existing_res = await self.session.execute(existing_stmt)
        existing = existing_res.scalar_one_or_none()

        thresholds: dict[str, Any] = dict(getattr(existing, "updated_thresholds", {}) or {})
        weights: dict[str, Any] = dict(getattr(existing, "heuristic_weights", {}) or {})
        stopwords: list[str] = list(getattr(existing, "stopwords", []) or [])
        delta: dict[str, Any] = dict(getattr(existing, "calibration_delta", {}) or {})
        stopwords_before = set(normalize_stopwords(stopwords))

        thresholds.update(extracted.updated_thresholds)
        weights.update(extracted.heuristic_weights)
        stopwords = normalize_stopwords(stopwords + extracted.new_stopwords)

        # Merge delta shallowly (new keys win).
        delta.update(extracted.calibration_delta)

        stopwords_after = set(stopwords)
        stopwords_added = len(stopwords_after - stopwords_before)

        stmt = (
            insert(OrgFeedbackLearningConfig)
            .values(
                {
                    "id": str(uuid4()),
                    "organization_id": organization_id,
                    "updated_thresholds": thresholds,
                    "stopwords": stopwords,
                    "heuristic_weights": weights,
                    "calibration_delta": delta,
                    "updated_by_user_id": updated_by_user_id,
                    "last_source_memory_id": source_memory_id,
                    "last_agent_version": agent_version,
                    "last_trace_id": trace_id,
                }
            )
            .on_conflict_do_update(
                index_elements=["organization_id"],
                set_={
                    "updated_thresholds": thresholds,
                    "stopwords": stopwords,
                    "heuristic_weights": weights,
                    "calibration_delta": delta,
                    "updated_by_user_id": updated_by_user_id,
                    "last_source_memory_id": source_memory_id,
                    "last_agent_version": agent_version,
                    "last_trace_id": trace_id,
                    "updated_at": func.now(),
                },
            )
        )

        await self.session.execute(stmt)
        await self.session.flush()

        return {
            "config_updated": True,
            "delta": extracted.calibration_delta,
            "stopwords_added": stopwords_added,
            "thresholds_updated": len(extracted.updated_thresholds),
            "weights_updated": len(extracted.heuristic_weights),
        }
