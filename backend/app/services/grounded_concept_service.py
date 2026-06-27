"""GroundedConceptService — Phase 91.

Attaches scalar groundings to key concepts in the knowledge graph so that
relational reasoning ("was A before B?", "is X larger than typical?") becomes
arithmetic rather than linguistic guessing.

Grounding types:
  temporal  — datetime/duration attached to an event
  scalar    — numeric value with unit (price, count, distance, percentage)
  ordinal   — rank in a sequence (1st, 2nd, last)
  boolean   — binary fact (approved/rejected, present/absent)

Usage:
  extract(text) -> list[GroundedConcept]
  compare(a, b) -> ComparisonResult   (temporal or scalar only)
  enrich_chunk(chunk) -> dict          (adds "groundings" key to payload)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # ISO dates
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    # Month DD, YYYY
    (re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
        re.IGNORECASE,
    ), "mdy"),
    # DD Month YYYY
    (re.compile(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})\b",
        re.IGNORECASE,
    ), "dmy"),
    # YYYY only
    (re.compile(r"\b((?:19|20)\d{2})\b"), "year"),
    # Q1/Q2... YYYY
    (re.compile(r"\b(Q[1-4])\s+((?:19|20)\d{2})\b", re.IGNORECASE), "quarter"),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_SCALAR_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d{2})?)\s*([BMK]?)\b"  # money
    r"|(\d+(?:\.\d+)?)\s*%"                      # percentage
    r"|(\d[\d,]*)\s+(users?|employees?|members?|items?|units?|km|miles?|years?|months?|days?)",
    re.IGNORECASE,
)

_ORDINAL_RE = re.compile(
    r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|"
    r"last|final|penultimate|top|bottom)\b",
    re.IGNORECASE,
)

_BOOL_AFFIRMATIVE = frozenset({
    "approved", "accepted", "confirmed", "granted", "present", "true",
    "passed", "successful", "completed", "active", "enabled",
})
_BOOL_NEGATIVE = frozenset({
    "rejected", "denied", "failed", "absent", "false", "blocked",
    "inactive", "disabled", "cancelled", "revoked",
})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GroundedConcept:
    raw_text: str           # original span from the text
    grounding_type: str     # "temporal" | "scalar" | "ordinal" | "boolean"
    normalized_value: Any   # datetime | float | int | str | bool
    unit: str | None        # e.g. "$", "%", "users", "km"
    confidence: float       # extraction confidence [0, 1]


@dataclass
class ComparisonResult:
    concept_a: GroundedConcept
    concept_b: GroundedConcept
    relation: str           # "before" | "after" | "equal" | "greater" | "less" | "incomparable"
    difference: Any | None  # timedelta or numeric difference


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class GroundedConceptService:

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[GroundedConcept]:
        """Extract all groundable concepts from text."""
        results: list[GroundedConcept] = []
        results.extend(self._extract_temporal(text))
        results.extend(self._extract_scalar(text))
        results.extend(self._extract_ordinal(text))
        results.extend(self._extract_boolean(text))
        return results

    def enrich_chunk(self, chunk: dict) -> dict:
        """Add a 'groundings' list to the chunk's payload. Non-destructive."""
        import copy
        enriched = copy.deepcopy(chunk)
        payload = enriched.setdefault("payload", {})
        text = payload.get("text") or chunk.get("text") or ""
        groundings = self.extract(text)
        payload["groundings"] = [
            {
                "raw": g.raw_text,
                "type": g.grounding_type,
                "value": str(g.normalized_value),
                "unit": g.unit,
                "confidence": g.confidence,
            }
            for g in groundings
        ]
        return enriched

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        a: GroundedConcept,
        b: GroundedConcept,
    ) -> ComparisonResult:
        """Compare two grounded concepts. Only meaningful for same grounding_type."""
        if a.grounding_type != b.grounding_type:
            return ComparisonResult(a, b, "incomparable", None)

        val_a, val_b = a.normalized_value, b.normalized_value

        if a.grounding_type == "temporal":
            try:
                if val_a < val_b:
                    rel, diff = "before", val_b - val_a
                elif val_a > val_b:
                    rel, diff = "after", val_a - val_b
                else:
                    rel, diff = "equal", None
                return ComparisonResult(a, b, rel, diff)
            except TypeError:
                return ComparisonResult(a, b, "incomparable", None)

        if a.grounding_type == "scalar":
            try:
                if val_a < val_b:
                    rel, diff = "less", val_b - val_a
                elif val_a > val_b:
                    rel, diff = "greater", val_a - val_b
                else:
                    rel, diff = "equal", 0.0
                return ComparisonResult(a, b, rel, diff)
            except TypeError:
                return ComparisonResult(a, b, "incomparable", None)

        return ComparisonResult(a, b, "incomparable", None)

    # ------------------------------------------------------------------
    # Private extractors
    # ------------------------------------------------------------------

    def _extract_temporal(self, text: str) -> list[GroundedConcept]:
        results: list[GroundedConcept] = []
        for pattern, fmt in _DATE_PATTERNS:
            for m in pattern.finditer(text):
                dt = self._parse_date(m, fmt)
                if dt is not None:
                    results.append(GroundedConcept(
                        raw_text=m.group(0),
                        grounding_type="temporal",
                        normalized_value=dt,
                        unit=None,
                        confidence=0.90 if fmt in ("iso", "mdy", "dmy") else 0.70,
                    ))
        # De-duplicate by span position
        seen: set[int] = set()
        unique = []
        for g in results:
            pos = text.find(g.raw_text)
            if pos not in seen:
                seen.add(pos)
                unique.append(g)
        return unique

    def _extract_scalar(self, text: str) -> list[GroundedConcept]:
        results: list[GroundedConcept] = []
        for m in _SCALAR_RE.finditer(text):
            raw = m.group(0)
            if m.group(1):  # money: $NNN[BMK]
                num_str = m.group(1).replace(",", "")
                mult_char = (m.group(2) or "").upper()
                mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(mult_char, 1.0)
                value = float(num_str) * mult
                unit = "$"
            elif m.group(3):  # percentage
                value = float(m.group(3))
                unit = "%"
            else:  # count with unit
                value = float(m.group(4).replace(",", ""))
                unit = m.group(5).lower().rstrip("s")
            results.append(GroundedConcept(
                raw_text=raw,
                grounding_type="scalar",
                normalized_value=value,
                unit=unit,
                confidence=0.85,
            ))
        return results

    def _extract_ordinal(self, text: str) -> list[GroundedConcept]:
        _ORD_MAP = {
            "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4, "fifth": 5, "5th": 5,
        }
        results: list[GroundedConcept] = []
        for m in _ORDINAL_RE.finditer(text):
            word = m.group(0).lower()
            rank = _ORD_MAP.get(word)
            results.append(GroundedConcept(
                raw_text=m.group(0),
                grounding_type="ordinal",
                normalized_value=rank if rank else word,
                unit=None,
                confidence=0.80,
            ))
        return results

    def _extract_boolean(self, text: str) -> list[GroundedConcept]:
        low = text.lower()
        results: list[GroundedConcept] = []
        for word in low.split():
            word_clean = re.sub(r"[^\w]", "", word)
            if word_clean in _BOOL_AFFIRMATIVE:
                results.append(GroundedConcept(
                    raw_text=word,
                    grounding_type="boolean",
                    normalized_value=True,
                    unit=None,
                    confidence=0.75,
                ))
            elif word_clean in _BOOL_NEGATIVE:
                results.append(GroundedConcept(
                    raw_text=word,
                    grounding_type="boolean",
                    normalized_value=False,
                    unit=None,
                    confidence=0.75,
                ))
        return results[:5]  # cap to avoid noise

    @staticmethod
    def _parse_date(m: re.Match, fmt: str) -> date | None:
        try:
            if fmt == "iso":
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if fmt == "mdy":
                month = _MONTH_MAP.get(m.group(1).lower())
                if month:
                    return date(int(m.group(3)), month, int(m.group(2)))
            if fmt == "dmy":
                month = _MONTH_MAP.get(m.group(2).lower())
                if month:
                    return date(int(m.group(3)), month, int(m.group(1)))
            if fmt == "year":
                return date(int(m.group(1)), 1, 1)
            if fmt == "quarter":
                q = int(m.group(1)[1])
                year = int(m.group(2))
                return date(year, (q - 1) * 3 + 1, 1)
        except (ValueError, IndexError):
            return None
        return None
