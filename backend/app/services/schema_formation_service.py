"""SchemaFormationService — Phase 87.

Induces abstract schema frames from recurring episodic memory patterns.

Pipeline (runs nightly):
  1. extract_event_fingerprint(memory)  — identify event type from verb+noun
  2. cluster_by_fingerprint(memories)   — group similar events
  3. extract_slots(cluster)             — find recurring named entities / roles
  4. upsert_schema(org_id, cluster)     — create or update the SchemaFrame

Frame-completion inference (on-demand):
  complete_frame(schema, partial_event) — fill missing slots from slot_distributions
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_INSTANCES = 3           # minimum cluster size to induce a schema
_MIN_SLOT_FILL_RATE = 0.50   # slot must appear in >= 50% of instances

# Verb clusters that mark common enterprise event types
_EVENT_VERBS: dict[str, list[str]] = {
    "acquisition":  ["acquired", "bought", "purchased", "merged", "takeover"],
    "project_kickoff": ["launched", "started", "initiated", "kicked off", "began"],
    "decision":     ["decided", "approved", "rejected", "voted", "resolved"],
    "meeting":      ["met", "discussed", "presented", "reviewed", "convened"],
    "milestone":    ["completed", "delivered", "shipped", "finished", "achieved"],
    "incident":     ["failed", "broke", "crashed", "errored", "incident"],
    "hiring":       ["hired", "joined", "onboarded", "promoted", "resigned"],
}

_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20})*)\b")
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?[BMK]?|\b\d+(?:\.\d+)?\s*(?:billion|million|thousand)\b",
                        re.IGNORECASE)
_ROLE_RE = re.compile(
    r"\b(CEO|CTO|CFO|VP|director|manager|lead|engineer|founder|president|chair)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlotDefinition:
    name: str
    slot_type: str   # "entity" | "date" | "money" | "role" | "text"
    required: bool
    fill_rate: float


@dataclass
class SchemaCandidate:
    event_type: str
    trigger_signature: str
    slots: list[SlotDefinition]
    slot_distributions: dict[str, Counter]
    instance_ids: list[str]
    avg_confidence: float


@dataclass
class FrameCompletion:
    filled_slots: dict[str, str]
    inferred_slots: dict[str, str]   # slots filled from distribution priors
    completeness: float               # fraction of required slots filled


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SchemaFormationService:

    # ------------------------------------------------------------------
    # Event fingerprinting
    # ------------------------------------------------------------------

    def extract_event_type(self, text: str) -> str | None:
        """Classify text into an event type using verb-cluster matching."""
        low = text.lower()
        for event_type, verbs in _EVENT_VERBS.items():
            if any(v in low for v in verbs):
                return event_type
        return None

    def fingerprint(self, event_type: str, entities: list[str]) -> str:
        key = event_type + "|" + ",".join(sorted(set(e.lower() for e in entities[:5])))
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Slot extraction from a text
    # ------------------------------------------------------------------

    def extract_slots_from_text(self, text: str) -> dict[str, list[str]]:
        """Extract typed slot values from a memory text."""
        return {
            "entity": _PROPER_NOUN_RE.findall(text),
            "date":   _DATE_RE.findall(text),
            "money":  _MONEY_RE.findall(text),
            "role":   _ROLE_RE.findall(text),
        }

    # ------------------------------------------------------------------
    # Schema induction from a cluster of memory texts
    # ------------------------------------------------------------------

    def induce_schema(
        self,
        event_type: str,
        memories: list[dict],
    ) -> SchemaCandidate | None:
        """
        Induce a SchemaFrame candidate from a cluster of similar memories.

        memories: list of dicts with keys "id" (str), "text" (str), "confidence" (float)
        """
        if len(memories) < _MIN_INSTANCES:
            return None

        # Accumulate slot observations across all instances
        slot_obs: dict[str, Counter] = defaultdict(Counter)
        confidences: list[float] = []
        instance_ids: list[str] = []

        for mem in memories:
            text = mem.get("text") or ""
            instance_ids.append(str(mem.get("id") or ""))
            confidences.append(float(mem.get("confidence") or 0.5))
            slots = self.extract_slots_from_text(text)
            for slot_type, values in slots.items():
                for v in values:
                    slot_obs[slot_type][v.strip()] += 1

        n = len(memories)

        # Build slot definitions: only slots present in >= min fill rate
        slot_defs: list[SlotDefinition] = []
        slot_dists: dict[str, Counter] = {}
        for slot_type, counter in slot_obs.items():
            total_fills = sum(counter.values())
            fill_rate = total_fills / (n * max(1, len(counter)))
            if fill_rate < _MIN_SLOT_FILL_RATE and total_fills < _MIN_INSTANCES:
                continue
            slot_defs.append(SlotDefinition(
                name=slot_type,
                slot_type=slot_type,
                required=fill_rate >= 0.80,
                fill_rate=round(fill_rate, 3),
            ))
            slot_dists[slot_type] = counter

        if not slot_defs:
            return None

        # Fingerprint is stable across cluster members
        all_entities = list(slot_obs.get("entity", {}).keys())[:5]
        sig = self.fingerprint(event_type, all_entities)

        return SchemaCandidate(
            event_type=event_type,
            trigger_signature=sig,
            slots=slot_defs,
            slot_distributions={k: dict(v) for k, v in slot_dists.items()},
            instance_ids=instance_ids,
            avg_confidence=round(sum(confidences) / len(confidences), 3),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def upsert_schema(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        candidate: SchemaCandidate,
    ) -> Any:
        """Insert or update a SchemaFrame row from a candidate."""
        from app.models.schema_frame import SchemaFrame

        stmt = select(SchemaFrame).where(
            SchemaFrame.organization_id == org_id,
            SchemaFrame.trigger_signature == candidate.trigger_signature,
        )
        row = (await session.execute(stmt)).scalar_one_or_none()

        if row is None:
            row = SchemaFrame(
                organization_id=org_id,
                name=candidate.event_type.replace("_", " ").title(),
                trigger_signature=candidate.trigger_signature,
                slots=[s.__dict__ for s in candidate.slots],
                slot_distributions=candidate.slot_distributions,
                instance_count=len(candidate.instance_ids),
                avg_confidence=candidate.avg_confidence,
                instance_memory_ids=candidate.instance_ids[:100],
            )
            session.add(row)
        else:
            # Merge: bump instance count, re-weight distributions
            row.instance_count += len(candidate.instance_ids)
            row.avg_confidence = round(
                (row.avg_confidence + candidate.avg_confidence) / 2, 3
            )
            # Merge slot distributions
            for slot_type, vals in candidate.slot_distributions.items():
                if slot_type not in row.slot_distributions:
                    row.slot_distributions[slot_type] = {}
                for val, cnt in vals.items():
                    row.slot_distributions[slot_type][val] = (
                        row.slot_distributions[slot_type].get(val, 0) + cnt
                    )
            # Extend known instance IDs (capped)
            existing = set(row.instance_memory_ids)
            new_ids = [i for i in candidate.instance_ids if i not in existing]
            row.instance_memory_ids = (row.instance_memory_ids + new_ids)[:200]

        await session.flush()
        return row

    async def list_schemas(
        self,
        session: AsyncSession,
        *,
        org_id: str,
        limit: int = 50,
    ) -> list[Any]:
        from app.models.schema_frame import SchemaFrame

        stmt = (
            select(SchemaFrame)
            .where(SchemaFrame.organization_id == org_id)
            .order_by(SchemaFrame.instance_count.desc())
            .limit(limit)
        )
        return list((await session.execute(stmt)).scalars().all())

    # ------------------------------------------------------------------
    # Frame-completion inference
    # ------------------------------------------------------------------

    def complete_frame(
        self,
        schema_row: Any,
        partial_slots: dict[str, str],
    ) -> FrameCompletion:
        """
        Fill missing required slots from the schema's learned distributions.
        Returns filled (from partial) and inferred (from prior) slots.
        """
        slot_defs: list[dict] = schema_row.slots or []
        distributions: dict[str, dict] = schema_row.slot_distributions or {}

        filled: dict[str, str] = dict(partial_slots)
        inferred: dict[str, str] = {}

        for slot in slot_defs:
            slot_name = slot.get("name") or ""
            if slot_name in filled:
                continue
            # Fill from most-common value in distribution
            dist = distributions.get(slot_name, {})
            if dist:
                best = max(dist, key=dist.__getitem__)
                inferred[slot_name] = best

        required = [s for s in slot_defs if s.get("required")]
        if required:
            covered = sum(
                1 for s in required
                if s.get("name") in filled or s.get("name") in inferred
            )
            completeness = covered / len(required)
        else:
            completeness = 1.0

        return FrameCompletion(
            filled_slots=filled,
            inferred_slots=inferred,
            completeness=round(completeness, 3),
        )
