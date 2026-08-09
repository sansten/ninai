"""Entity Resolution Agent — Phase 7.

Resolves extracted entities against the enterprise ontology to surface canonical
names and cross-silo links (same concept appearing across departments).

Also handles domain-agnostic entity types:
  - Temporal entities: holiday names, date format variants → canonical ISO date
  - Person entities: proper names not in ontology → stored as extracted persons
  - These ensure mem.entities is populated for non-enterprise domains (e.g. LoCoMo)
    so graph expansion can still fire in search_memories().

Depends on Phase 6 (SemanticNormalizationAgent) via memory enrichment
semantic_relationships field.

Outputs power downstream phases:
  - Phase 4 (World Model Graph): resolved canonical nodes
  - Phase 8 (Context Amplifier): cross-silo fan-out on expert reads
  - Phase 9 (Silo Propagation): routes signals via domain membership
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.llm_breaker import create_llm_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Built-in ontology — canonical enterprise entities + aliases
# (tenant-specific ontology injected at runtime from OntologyEntry DB rows)
# ---------------------------------------------------------------------------

_BUILTIN_ONTOLOGY: list[dict[str, Any]] = [
    {
        "canonical": "customer",
        "entity_type": "role",
        "aliases": ["client", "account", "end-user", "end user", "buyer", "subscriber"],
        "domains": ["sales", "finance", "support"],
    },
    {
        "canonical": "revenue",
        "entity_type": "metric",
        "aliases": ["arr", "mrr", "income", "bookings", "sales revenue"],
        "domains": ["sales", "finance"],
    },
    {
        "canonical": "project",
        "entity_type": "project",
        "aliases": ["initiative", "program", "effort", "workstream", "epic"],
        "domains": ["engineering", "product", "operations"],
    },
    {
        "canonical": "incident",
        "entity_type": "event",
        "aliases": ["outage", "issue", "problem", "failure", "defect"],
        "domains": ["engineering", "support", "operations"],
    },
    {
        "canonical": "contract",
        "entity_type": "document",
        "aliases": ["agreement", "nda", "sow", "statement of work", "msa"],
        "domains": ["sales", "legal", "finance"],
    },
    {
        "canonical": "employee",
        "entity_type": "role",
        "aliases": ["staff", "team member", "headcount", "hire", "worker", "colleague"],
        "domains": ["hr", "finance", "operations"],
    },
    {
        "canonical": "vendor",
        "entity_type": "organization",
        "aliases": ["supplier", "partner", "third-party", "third party", "service provider"],
        "domains": ["operations", "finance", "legal"],
    },
    {
        "canonical": "budget",
        "entity_type": "metric",
        "aliases": ["cost", "spend", "expense", "allocation", "forecast", "capex", "opex"],
        "domains": ["finance", "operations", "engineering"],
    },
    {
        "canonical": "deployment",
        "entity_type": "event",
        "aliases": ["release", "rollout", "launch", "ship", "publish"],
        "domains": ["engineering", "product", "operations"],
    },
    {
        "canonical": "sprint",
        "entity_type": "time_period",
        "aliases": ["iteration", "cycle", "milestone", "delivery"],
        "domains": ["product", "engineering"],
    },
]


def _build_alias_index(
    ontology: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a flat alias→entry lookup from an ontology list."""
    index: dict[str, dict[str, Any]] = {}
    for entry in ontology:
        index[entry["canonical"].lower()] = entry
        for alias in entry.get("aliases", []):
            key = alias.lower().strip()
            if key and key not in index:
                index[key] = entry
    return index


# Pre-built index for the built-in ontology
_BUILTIN_INDEX: dict[str, dict[str, Any]] = _build_alias_index(_BUILTIN_ONTOLOGY)


# ---------------------------------------------------------------------------
# Temporal entity resolution — holidays and date variants
# Allows "Independence Day" and "July 4" to resolve to the same canonical node.
# ---------------------------------------------------------------------------

_TEMPORAL_ONTOLOGY: list[dict[str, Any]] = [
    {
        "canonical": "independence_day",
        "entity_type": "holiday",
        "canonical_date": "07-04",
        "aliases": [
            "independence day", "july 4", "july 4th", "4th of july",
            "july fourth", "fourth of july",
        ],
        "domains": ["general"],
    },
    {
        "canonical": "christmas",
        "entity_type": "holiday",
        "canonical_date": "12-25",
        "aliases": ["christmas day", "xmas", "christmas"],
        "domains": ["general"],
    },
    {
        "canonical": "thanksgiving",
        "entity_type": "holiday",
        "canonical_date": "11-T4",  # 4th Thursday
        "aliases": ["thanksgiving day", "thanksgiving"],
        "domains": ["general"],
    },
    {
        "canonical": "new_year",
        "entity_type": "holiday",
        "canonical_date": "01-01",
        "aliases": [
            "new year", "new year's", "new years", "new years day",
            "new year's day", "new year day",
        ],
        "domains": ["general"],
    },
    {
        "canonical": "halloween",
        "entity_type": "holiday",
        "canonical_date": "10-31",
        "aliases": ["halloween", "all hallows eve"],
        "domains": ["general"],
    },
    {
        "canonical": "valentines_day",
        "entity_type": "holiday",
        "canonical_date": "02-14",
        "aliases": ["valentines day", "valentine's day", "valentines"],
        "domains": ["general"],
    },
    {
        "canonical": "mothers_day",
        "entity_type": "holiday",
        "canonical_date": "05-2S",  # 2nd Sunday May
        "aliases": ["mothers day", "mother's day"],
        "domains": ["general"],
    },
    {
        "canonical": "fathers_day",
        "entity_type": "holiday",
        "canonical_date": "06-3S",  # 3rd Sunday June
        "aliases": ["fathers day", "father's day"],
        "domains": ["general"],
    },
    {
        "canonical": "labor_day",
        "entity_type": "holiday",
        "canonical_date": "09-1M",  # 1st Monday Sep
        "aliases": ["labor day", "labour day"],
        "domains": ["general"],
    },
    {
        "canonical": "memorial_day",
        "entity_type": "holiday",
        "canonical_date": "05-LM",  # Last Monday May
        "aliases": ["memorial day"],
        "domains": ["general"],
    },
    {
        "canonical": "easter",
        "entity_type": "holiday",
        "canonical_date": "variable",
        "aliases": ["easter", "easter sunday"],
        "domains": ["general"],
    },
]

_TEMPORAL_INDEX: dict[str, dict[str, Any]] = {}
for _entry in _TEMPORAL_ONTOLOGY:
    _key = _entry["canonical"].lower()
    _TEMPORAL_INDEX[_key] = _entry
    for _alias in _entry.get("aliases", []):
        _ak = _alias.lower().strip()
        if _ak and _ak not in _TEMPORAL_INDEX:
            _TEMPORAL_INDEX[_ak] = _entry


# Regex patterns for common date formats → ISO normalization
_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # MM-DD-YYYY or MM/DD/YYYY
    (re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"), "{2}-{0:02d}-{1:02d}"),
    # Month DD, YYYY or Month YYYY
    (re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        re.IGNORECASE,
    ), "month_name"),
    # DD Month YYYY
    (re.compile(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{4})\b",
        re.IGNORECASE,
    ), "day_month_name"),
]

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_TURN_PREFIX_RE = re.compile(r"^\[([^\]]+)\](?:\s+\[([^\]]+)\])?\s*")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_turn_context(content: str) -> tuple[str | None, str | None, str]:
    """Parse LoCoMo-style prefixes like `[2023-05-08] [Caroline] text`."""
    text = content or ""
    m = _TURN_PREFIX_RE.match(text)
    if not m:
        return None, None, text

    first = str(m.group(1) or "").strip()
    second = str(m.group(2) or "").strip()
    body = text[m.end():].strip()

    if _ISO_DATE_RE.match(first):
        return first, second or None, body
    if second and _ISO_DATE_RE.match(second):
        return second, first or None, body
    return None, first or None, body


def _normalize_date_string(text: str) -> str | None:
    """Try to parse a date string and return ISO YYYY-MM-DD, or None."""
    t = text.strip()
    # MM-DD-YYYY / MM/DD/YYYY
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", t)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    # Month DD YYYY
    m = re.match(
        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$", t, re.IGNORECASE,
    )
    if m:
        mo = _MONTH_MAP.get(m.group(1)[:3].lower(), 0)
        d, y = int(m.group(2)), int(m.group(3))
        if mo and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    # DD Month YYYY
    m = re.match(
        r"^(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{4})$", t, re.IGNORECASE,
    )
    if m:
        d = int(m.group(1))
        mo = _MONTH_MAP.get(m.group(2)[:3].lower(), 0)
        y = int(m.group(3))
        if mo and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
    return None


def _extract_date_entities(text: str) -> list[dict[str, Any]]:
    """Extract and normalize all date expressions in text as canonical entities."""
    results: list[dict[str, Any]] = []
    seen_canonicals: set[str] = set()
    anchor_date, _, _ = _parse_turn_context(text)

    if anchor_date and anchor_date not in seen_canonicals:
        seen_canonicals.add(anchor_date)
        results.append({
            "original": anchor_date,
            "canonical": anchor_date,
            "entity_type": "date",
            "domains": ["general"],
            "match_type": "anchor_date",
            "confidence": 1.0,
        })

    # Multi-word date patterns
    for pattern, _ in _DATE_PATTERNS:
        for m in pattern.finditer(text):
            iso = _normalize_date_string(m.group(0))
            if iso and iso not in seen_canonicals:
                seen_canonicals.add(iso)
                results.append({
                    "original": m.group(0),
                    "canonical": iso,
                    "entity_type": "date",
                    "domains": ["general"],
                    "match_type": "normalized",
                    "confidence": 0.95,
                })

    return results


def _extract_relative_date_entities(text: str) -> list[dict[str, Any]]:
    """Normalize basic relative dates against the turn's anchor date."""
    anchor_date, _, body = _parse_turn_context(text)
    if not anchor_date:
        return []

    try:
        anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")
    except ValueError:
        return []

    lower = body.lower()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _emit(original: str, canonical: str, *, match_type: str) -> None:
        if canonical in seen:
            return
        seen.add(canonical)
        results.append({
            "original": original,
            "canonical": canonical,
            "entity_type": "date",
            "domains": ["general"],
            "match_type": match_type,
            "confidence": 0.9,
        })

    for phrase, delta_days in (("yesterday", -1), ("today", 0), ("tomorrow", 1)):
        if re.search(rf"\b{phrase}\b", lower):
            canonical = (anchor_dt + timedelta(days=delta_days)).strftime("%Y-%m-%d")
            _emit(phrase, canonical, match_type="relative_date")

    for modifier, direction in (("last", -1), ("next", 1)):
        for weekday_name, weekday_idx in _WEEKDAY_INDEX.items():
            if not re.search(rf"\b{modifier}\s+{weekday_name}\b", lower):
                continue
            delta = weekday_idx - anchor_dt.weekday()
            if direction < 0:
                delta = delta - 7 if delta >= 0 else delta
            else:
                delta = delta + 7 if delta <= 0 else delta
            canonical = (anchor_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
            _emit(f"{modifier} {weekday_name}", canonical, match_type="relative_weekday")

    return results


def _resolve_temporal(name: str) -> dict[str, Any] | None:
    """Resolve a name against the temporal ontology (holidays, date variants)."""
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in _TEMPORAL_INDEX:
        entry = _TEMPORAL_INDEX[key]
        match_type = "exact" if key == entry["canonical"].lower() else "alias"
        return {
            "original": name,
            "canonical": entry["canonical"],
            "entity_type": entry["entity_type"],
            "canonical_date": entry.get("canonical_date"),
            "domains": entry.get("domains", []),
            "match_type": match_type,
            "confidence": 1.0 if match_type == "exact" else 0.9,
        }
    return None


# ---------------------------------------------------------------------------
# Personal attribute extraction — conversational corpora (LoCoMo-style)
# Turns with a [Speaker] prefix carry first-person facts that would otherwise
# be invisible to the enterprise ontology (relationship_status, origin, etc.).
# Extracted attributes are stored as personal_attribute entities so that:
#   (a) mem.entities is populated for graph expansion, and
#   (b) _apply_entity_resolution writes a [Fact] derived memory for direct
#       semantic retrieval without vocabulary gap.
# ---------------------------------------------------------------------------

def _extract_personal_attributes(content: str) -> list[dict[str, Any]]:
    """Extract personal attribute facts from a [Speaker]-prefixed turn.

    Returns entity dicts with entity_type='personal_attribute'.
    Only fires when the content begins with a [Speaker] token.
    """
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    _, speaker, text = _parse_turn_context(content)
    if not speaker:
        return results

    subject = speaker.strip()
    subj_lc = subject.lower()

    def _emit(attribute: str, value: str) -> None:
        canonical = f"{subj_lc}_{attribute}"
        if canonical not in seen:
            seen.add(canonical)
            results.append({
                "entity_type": "personal_attribute",
                "canonical": canonical,
                "subject": subject,
                "attribute": attribute,
                "value": value,
                "domains": ["personal"],
                "match_type": "extracted",
                "confidence": 0.8,
            })

    # Origin: "my home country, Sweden" / "my home country of Sweden"
    m2 = re.search(r"\bmy home country[,\s]+(?:of\s+)?([A-Z][a-zA-Z]+)", text)
    if m2:
        _emit("origin", m2.group(1))

    # Relationship status: "single parent/mom/dad"
    if re.search(r"\bsingle\s+(?:parent|mom|dad|father|mother)\b", text, re.I):
        _emit("relationship_status", "single")

    # Research topic: "Researching adoption agencies"
    m3 = re.search(r"\b[Rr]esearching\s+([^.–—\n]{3,60}?)(?:[.!?,]|$)", text)
    if m3:
        _emit("research_topic", m3.group(1).strip().rstrip(".,"))

    # Book read: quoted title near reading verb (straight double quotes; apostrophes allowed inside)
    m4 = re.search(
        r'\b(?:loved?|finished?|read|reading|re-read)\b[^"]{0,30}'
        r'"([^"\n]{3,80})"',
        text, re.I,
    )
    if m4:
        _emit("book_read", m4.group(1).strip())

    # Painted subject: "I painted that lake sunrise" — stop at time/clause words
    m5 = re.search(
        r"\bpainted\s+(?:the\s+|that\s+|a\s+)?(.+?)"
        r"(?:\s+(?:last|when|while|before|after|ago|in|at|on|and|but)\b|[.!?,]|$)",
        text, re.I,
    )
    if m5:
        val = m5.group(1).strip().rstrip(".,")
        if val:
            _emit("painted", val)

    # Destress activity: "running farther to de-stress" / "de-stress by running"
    m6 = re.search(
        r"\b(\w+ing)\s+(?:further|farther|more)?\s*(?:to\s+)?(?:de-stress|destress|unwind|relax)\b",
        text, re.I,
    )
    if not m6:
        m6 = re.search(
            r"\b(?:de-stress|destress|unwind|relax)\s+(?:by|with|through)\s+(\w+(?:ing)?)\b",
            text, re.I,
        )
    if m6:
        _emit("destress_activity", m6.group(1).lower())

    # Identity: "my trans experience" / "my transgender journey"
    if re.search(
        r"\bmy\s+(?:trans|transgender)\s+(?:experience|journey|transition|identity|story)\b",
        text, re.I,
    ):
        _emit("identity", "transgender")

    # Career interest: counseling / mental health
    m7 = re.search(
        r"\b(?:studying|training|working|going into|interested in|pursuing)\s+"
        r"((?:mental health|counseling|counselling|therapy|social work)[^.!?\n]{0,40})",
        text, re.I,
    )
    if m7:
        _emit("career_interest", m7.group(1).strip().rstrip(".,"))

    return results


_PERSON_SKIP = frozenset({
    # conjunctions / prepositions / articles
    "the", "and", "but", "for", "nor", "yet", "with", "from", "that", "this",
    # pronouns — must not be stored as person entities
    "she", "her", "hers", "him", "his", "they", "them", "their", "theirs",
    "you", "your", "yours", "our", "ours", "its", "who", "whom", "whose",
    "there", "then", "than", "when", "what", "which", "how", "why",
    # auxiliaries
    "can", "could", "should", "would", "will", "has", "have", "had",
    "was", "were", "are", "been", "being", "not", "also", "very", "just",
    "said", "says", "one", "two", "three", "four", "five",
    # day / month names (would otherwise extract "Monday", "June" as persons)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    # title prefixes
    "mr", "mrs", "ms", "dr", "prof", "sir",
})


def _looks_like_person_name(token: str) -> bool:
    """Heuristic: single capitalized word that is not a known non-person term."""
    t = token.strip()
    if len(t) < 3 or not t[0].isupper():
        return False
    if t.isupper():
        return False
    return t.lower() not in _PERSON_SKIP


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_entity(
    name: str,
    extra_ontology: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Resolve a single entity name against built-in + temporal + extra ontology.

    Match priority:
    1. Temporal ontology (holidays, date aliases) — domain-agnostic
    2. Exact canonical match against enterprise ontology  → confidence 1.0
    3. Alias match                                        → confidence 0.85
    4. Prefix/substring fuzzy fallback                    → confidence 0.65

    Returns a resolution dict or None if unresolved.
    """
    key = _normalize(name)
    if not key:
        return None

    # Temporal resolution first — holidays and date variants are universal
    temporal = _resolve_temporal(name)
    if temporal:
        return temporal

    # Merge lookup: extra ontology aliases take precedence over builtin
    lookup = dict(_BUILTIN_INDEX)
    if extra_ontology:
        extra_index = _build_alias_index(extra_ontology)
        lookup.update(extra_index)

    # Exact / alias match
    if key in lookup:
        entry = lookup[key]
        match_type = "exact" if key == entry["canonical"].lower() else "alias"
        confidence = 1.0 if match_type == "exact" else 0.85
        return {
            "original": name,
            "canonical": entry["canonical"],
            "entity_type": entry["entity_type"],
            "domains": entry.get("domains", []),
            "match_type": match_type,
            "confidence": confidence,
        }

    # Prefix / substring fuzzy match (cheap, no external dependency)
    for alias_key, entry in lookup.items():
        if len(alias_key) >= 3 and (alias_key.startswith(key) or key.startswith(alias_key)):
            return {
                "original": name,
                "canonical": entry["canonical"],
                "entity_type": entry["entity_type"],
                "domains": entry.get("domains", []),
                "match_type": "prefix",
                "confidence": 0.65,
            }

    return None


def extract_entities_from_content(text: str) -> list[str]:
    """Heuristic: extract capitalized noun phrases as candidate entities.

    Targets proper-noun-like tokens (e.g. "Acme Corp", "Q1 Budget").
    Caps at 20 to avoid noise.
    """
    if not text:
        return []
    tokens = re.findall(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b", text)
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result[:20]


def build_cross_silo_links(
    resolved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Surface entities whose ontology spans 2+ business domains (cross-silo).

    These are the structural links between silos — e.g. "customer" appears
    in sales, finance, and support. Returned once per canonical name.
    """
    links: list[dict[str, Any]] = []
    seen_canonicals: set[str] = set()
    for entity in resolved:
        canonical = entity["canonical"]
        domains = entity.get("domains", [])
        if len(domains) >= 2 and canonical not in seen_canonicals:
            seen_canonicals.add(canonical)
            links.append(
                {
                    "canonical": canonical,
                    "entity_type": entity["entity_type"],
                    "domains": domains,
                    "silo_count": len(domains),
                }
            )
    return links


class EntityResolutionAgent(BaseAgent):
    """Phase 7: Resolve entities against the enterprise ontology.

    Inputs  (via context["memory"]["enrichment"]):
      - semantic_relationships: list[{type, concept}] — from Phase 6

    Outputs:
      - resolved_entities:   list[{original, canonical, entity_type, domains,
                                   match_type, confidence}]
      - unresolved_entities: list[str]
      - cross_silo_links:    list[{canonical, entity_type, domains, silo_count}]
      - structured_facts:    list[{subject, predicate, object, confidence}]
      - confidence:          float
      - rationale:           "heuristic" | "llm"
    """

    name = "EntityResolutionAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SemanticNormalizationAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        if not isinstance(outputs.get("resolved_entities", []), list):
            raise ValueError("resolved_entities must be a list")
        if not isinstance(outputs.get("cross_silo_links", []), list):
            raise ValueError("cross_silo_links must be a list")
        if not isinstance(outputs.get("unresolved_entities", []), list):
            raise ValueError("unresolved_entities must be a list")
        if "structured_facts" in outputs and not isinstance(outputs.get("structured_facts"), list):
            raise ValueError("structured_facts must be a list")
        for entity in outputs.get("resolved_entities", []):
            if not isinstance(entity, dict):
                raise ValueError("each resolved_entity must be a dict")
            if "canonical" not in entity or "entity_type" not in entity:
                raise ValueError("each resolved_entity must have canonical and entity_type")

    def _heuristic(
        self,
        *,
        content: str,
        relationships: list[dict[str, str]],
        extra_ontology: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Collect candidate entity names from Phase 6 relationships + raw content
        candidate_names: list[str] = []
        seen_names: set[str] = set()

        for rel in relationships:
            concept = rel.get("concept", "")
            if concept and _normalize(concept) not in seen_names:
                candidate_names.append(concept)
                seen_names.add(_normalize(concept))

        for entity in extract_entities_from_content(content):
            if _normalize(entity) not in seen_names:
                candidate_names.append(entity)
                seen_names.add(_normalize(entity))

        resolved: list[dict[str, Any]] = []
        unresolved: list[str] = []

        for name in candidate_names[:30]:
            resolution = resolve_entity(name, extra_ontology)
            if resolution:
                resolved.append(resolution)
            else:
                unresolved.append(name)

        # Date entities — normalized from any format present in content.
        # These are always stored regardless of ontology coverage (e.g. "06-14-2026"
        # and "Jun 14 2026" both resolve to "2026-06-14" as the same graph node).
        seen_canonicals = {r["canonical"] for r in resolved}
        for date_ent in _extract_date_entities(content):
            if date_ent["canonical"] not in seen_canonicals:
                resolved.append(date_ent)
                seen_canonicals.add(date_ent["canonical"])
        for date_ent in _extract_relative_date_entities(content):
            if date_ent["canonical"] not in seen_canonicals:
                resolved.append(date_ent)
                seen_canonicals.add(date_ent["canonical"])

        # Speaker attribution: if content begins with [SpeakerName], emit a speaker entity
        # at confidence 1.0 so entity-indexed retrieval strongly anchors this memory to the
        # named speaker — critical for adversarial QA where attribution determines correctness.
        _, speaker, _ = _parse_turn_context(content.strip())
        if speaker:
            _spk_name = speaker
            _spk_canonical = _spk_name.lower()
            if _spk_canonical not in seen_canonicals:
                resolved.append({
                    "original": _spk_name,
                    "canonical": _spk_canonical,
                    "entity_type": "speaker",
                    "domains": ["conversational"],
                    "match_type": "speaker_turn",
                    "confidence": 1.0,
                })
                seen_canonicals.add(_spk_canonical)

        # Pass-through: extracted proper names that didn't match any ontology are stored
        # as person entities so the KG can still build edges between memories that
        # mention the same person (e.g. "Caroline" links memory-10 and memory-50).
        for name in unresolved:
            if _looks_like_person_name(name):
                canonical = name.lower()
                if canonical not in seen_canonicals:
                    resolved.append({
                        "original": name,
                        "canonical": canonical,
                        "entity_type": "person",
                        "domains": ["general"],
                        "match_type": "extracted",
                        "confidence": 0.5,
                    })
                    seen_canonicals.add(canonical)

        cross_silo = build_cross_silo_links(resolved)

        total = len(candidate_names)
        if total == 0:
            confidence = 0.5
        else:
            confidence = min(0.9, round(0.4 + 0.5 * (len(resolved) / total), 3))

        return {
            "resolved_entities": resolved,
            "unresolved_entities": unresolved,
            "cross_silo_links": cross_silo,
            "structured_facts": [],
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        content = (context.get("memory") or {}).get("content", "")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        # Phase 6 semantic_relationships fed as enrichment
        relationships: list[dict[str, str]] = enrichment.get("semantic_relationships", [])

        strategy = getattr(settings, "ENTITY_RESOLUTION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(content=content, relationships=relationships)
        else:
            prompt = (
                "You are a memory extraction engine for a cognitive AI system.\n"
                "Output valid JSON only. Do not hallucinate. Only extract what is explicitly stated.\n\n"
                "Extract from the content below:\n\n"
                "1. ENTERPRISE ENTITIES — resolve against canonical business concepts:\n"
                "   entity_type options: role, metric, project, event, document, organization, "
                "time_period, holiday, date, person\n"
                "   For each: {original, canonical, entity_type, domains, match_type, confidence}\n\n"
                "2. PERSONAL ATTRIBUTE FACTS — when the content begins with [SpeakerName], "
                "extract facts the speaker reveals about themselves.\n"
                "   Use entity_type='personal_attribute' and canonical='{speaker_lower}_{attribute}'.\n"
                "   Format: {entity_type, canonical, subject, attribute, value, "
                "domains: ['personal'], match_type: 'extracted', confidence: 0.8}\n"
                "   Extract these attribute types only when clearly stated (never infer):\n"
                "     - origin: country/city the speaker moved from\n"
                "     - relationship_status: single, married, divorced, etc.\n"
                "     - research_topic: what they are currently researching\n"
                "     - book_read: book title mentioned with a reading verb (exact title)\n"
                "     - painted: subject/scene the speaker painted\n"
                "     - destress_activity: how they destress, unwind, or relax\n"
                "     - identity: gender identity or cultural identity if explicitly stated\n"
                "     - career_interest: profession or field they are studying or pursuing\n"
                "     - event: significant event the speaker describes (brief label, e.g. 'ran charity race')\n"
                "     - location_visited: place the speaker visited or went to\n"
                "     - activity: regular activity the speaker does (e.g. 'pottery', 'running')\n\n"
                "3. STRUCTURED FACTS — extract explicit, reusable facts from any domain as short triples.\n"
                "   Format each fact as: {subject, predicate, object, confidence}\n"
                "   Rules:\n"
                "     - only extract facts explicitly supported by the content\n"
                "     - prefer normalized predicate names like relationship_status, depends_on, moved_from,\n"
                "       works_for, located_in, attended, planned_for, identity, owns, likes\n"
                "     - keep object concise and copy exact names when possible\n"
                "     - do not invent a fact if the subject is unclear\n\n"
                "4. unresolved_entities: list of entity name strings that could not be resolved\n"
                "5. cross_silo_links: list of {canonical, entity_type, domains, silo_count} "
                "for entities spanning multiple business departments\n"
                "6. confidence: float 0.0 to 1.0\n"
                "7. rationale: 'llm'\n\n"
                f"CONTENT:\n{content}\n\n"
                "Return JSON with exactly these keys: resolved_entities, unresolved_entities, "
                "cross_silo_links, structured_facts, confidence, rationale"
            )
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_llm_model("agents")),
                timeout_seconds=float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2)),
            )
            try:
                resp = await client.complete_json(
                    prompt=prompt,
                    schema_hint={},
                    tool_event_sink=context.get("tool_event_sink"),
                )
            except Exception:
                resp = None
            if (
                isinstance(resp, dict)
                and isinstance(resp.get("resolved_entities"), list)
                and isinstance(resp.get("unresolved_entities"), list)
                and isinstance(resp.get("cross_silo_links"), list)
                and isinstance(resp.get("structured_facts", []), list)
            ):
                # Merge in deterministic heuristic results (date normalization, enterprise
                # ontology lookup) that are cheap, certain, and don't need LLM.
                heuristic = self._heuristic(content=content, relationships=relationships)
                llm_canonicals = {
                    e["canonical"] for e in resp["resolved_entities"]
                    if isinstance(e, dict) and e.get("canonical")
                }
                for ent in heuristic["resolved_entities"]:
                    if isinstance(ent, dict) and ent.get("canonical") not in llm_canonicals:
                        resp["resolved_entities"].append(ent)
                if not resp.get("cross_silo_links"):
                    resp["cross_silo_links"] = heuristic["cross_silo_links"]
                resp.setdefault("structured_facts", [])
                outputs = resp
            else:
                outputs = self._heuristic(content=content, relationships=relationships)
        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.5)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
