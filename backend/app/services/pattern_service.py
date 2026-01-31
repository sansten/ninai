"""Pattern persistence.

Materializes PatternDetectionAgent outputs into Postgres tables:
- memory_patterns
- memory_pattern_evidence

Idempotent via upserts and unique constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.memory_pattern import MemoryPattern
from app.models.memory_pattern_evidence import MemoryPatternEvidence


_NORM_RE = re.compile(r"[^a-z0-9_]+")


def normalize_pattern_key(val: str) -> str:
    s = (val or "").strip().lower()
    s = s.replace(" ", "_")
    s = _NORM_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass(frozen=True)
class ExtractedPattern:
    pattern_key: str
    pattern_type: str
    confidence: float
    evidence: list[str]


def extract_patterns(outputs: dict[str, Any] | None) -> list[ExtractedPattern]:
    if not isinstance(outputs, dict):
        return []
    patterns = outputs.get("patterns")
    if not isinstance(patterns, list):
        return []

    out: list[ExtractedPattern] = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        pk = normalize_pattern_key(str(p.get("pattern") or ""))
        if not pk:
            continue
        pt = normalize_pattern_key(str(p.get("type") or "unknown")) or "unknown"
        try:
            conf = float(p.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        ev_raw = p.get("evidence")
        ev: list[str] = []
        if isinstance(ev_raw, list):
            for e in ev_raw:
                if e is None:
                    continue
                s = str(e).strip()
                if s:
                    ev.append(s)

        out.append(ExtractedPattern(pattern_key=pk, pattern_type=pt, confidence=conf, evidence=ev[:12]))

    # Deduplicate by pattern_key
    seen: set[str] = set()
    uniq: list[ExtractedPattern] = []
    for p in out:
        if p.pattern_key in seen:
            continue
        seen.add(p.pattern_key)
        uniq.append(p)
    return uniq


class PatternService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_patterns_for_memory(
        self,
        *,
        organization_id: str,
        memory_id: str,
        scope: str,
        scope_id: str | None,
        outputs: dict[str, Any] | None,
        created_by: str = "agent",
    ) -> dict[str, Any]:
        extracted = extract_patterns(outputs)
        if not extracted:
            return {"patterns_upserted": 0, "evidence_upserted": 0}

        scope_key = f"{scope}:{scope_id or ''}"
        patterns_upserted = 0
        evidence_upserted = 0

        for p in extracted:
            pattern_stmt = (
                insert(MemoryPattern)
                .values(
                    {
                        "id": str(uuid4()),
                        "organization_id": organization_id,
                        "scope": scope,
                        "scope_id": scope_id,
                        "scope_key": scope_key,
                        "pattern_key": p.pattern_key,
                        "pattern_type": p.pattern_type,
                        "confidence": float(p.confidence),
                        "details": {},
                        "created_by": created_by,
                    }
                )
                .on_conflict_do_update(
                    index_elements=["organization_id", "scope_key", "pattern_key"],
                    set_={
                        "pattern_type": p.pattern_type,
                        "confidence": float(p.confidence),
                        "updated_at": func.now(),
                    },
                )
                .returning(MemoryPattern.id)
            )

            pr = await self.session.execute(pattern_stmt)
            pattern_id = pr.scalar_one()
            patterns_upserted += 1

            evidence_stmt = (
                insert(MemoryPatternEvidence)
                .values(
                    {
                        "id": str(uuid4()),
                        "organization_id": organization_id,
                        "memory_id": memory_id,
                        "pattern_id": str(pattern_id),
                        "confidence": float(p.confidence),
                        "evidence": p.evidence,
                        "created_by": created_by,
                    }
                )
                .on_conflict_do_update(
                    index_elements=["organization_id", "memory_id", "pattern_id"],
                    set_={
                        "confidence": float(p.confidence),
                        "evidence": p.evidence,
                        "updated_at": func.now(),
                    },
                )
            )
            await self.session.execute(evidence_stmt)
            evidence_upserted += 1

        await self.session.flush()
        return {"patterns_upserted": patterns_upserted, "evidence_upserted": evidence_upserted}
