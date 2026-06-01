"""V2 heuristic entity extraction helpers.

Ports the highest-signal parts of the V1 entity extraction path into a small
utility that V2 can use during write-back without depending on the full V1
agent pipeline.

Focus areas:
  - speaker anchoring from ``[Speaker]`` prefixes
  - temporal normalization, including relative dates anchored by ``[YYYY-MM-DD]``
  - rich personal attributes: hobbies, jobs, favorites, pets, health, skills, location
  - temporal events: (person, activity, date) triples for date-indexed retrieval
  - lightweight proper-name extraction for graph linking
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

_ANCHOR_DATE_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]\s*")
_SPEAKER_RE = re.compile(r"^\[([^\]]+)\]\s*")

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_PERSON_SKIP = frozenset({
    "the", "and", "but", "for", "nor", "yet", "with", "from", "that", "this",
    "she", "her", "hers", "him", "his", "they", "them", "their", "theirs",
    "you", "your", "yours", "our", "ours", "its", "who", "whom", "whose",
    "there", "then", "than", "when", "what", "which", "how", "why",
    "can", "could", "should", "would", "will", "has", "have", "had",
    "was", "were", "are", "been", "being", "not", "also", "very", "just",
    "said", "says", "one", "two", "three", "four", "five",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "mr", "mrs", "ms", "dr", "prof", "sir",
})


def _snake(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    return value or "unknown"


def _to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _format_date(dt: date) -> str:
    return dt.strftime("%Y-%m-%d")


def parse_temporal_context(text: str) -> dict[str, str]:
    """Extract anchor_date/speaker prefixes from a benchmark-style utterance."""
    out: dict[str, str] = {}
    s = str(text or "").strip()

    m = _ANCHOR_DATE_RE.match(s)
    if m:
        out["anchor_date"] = m.group(1)
        s = s[m.end():].lstrip()

    m = _SPEAKER_RE.match(s)
    if m:
        out["speaker"] = m.group(1).strip()

    return out


def _strip_prefixes(text: str) -> tuple[str, str | None, str | None]:
    s = str(text or "").strip()
    ctx = parse_temporal_context(s)

    m = _ANCHOR_DATE_RE.match(s)
    if m:
        s = s[m.end():].lstrip()
    m = _SPEAKER_RE.match(s)
    if m:
        s = s[m.end():].lstrip()

    return s, ctx.get("anchor_date"), ctx.get("speaker")


def _normalize_date_string(text: str) -> str | None:
    t = text.strip()

    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", t)
    if m:
        mo, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= day <= 31:
            return f"{year}-{mo:02d}-{day:02d}"

    m = re.match(
        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$",
        t,
        re.IGNORECASE,
    )
    if m:
        mo = _MONTH_MAP.get(m.group(1)[:3].lower(), 0)
        day, year = int(m.group(2)), int(m.group(3))
        if mo and 1 <= day <= 31:
            return f"{year}-{mo:02d}-{day:02d}"

    m = re.match(
        r"^(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{4})$",
        t,
        re.IGNORECASE,
    )
    if m:
        day = int(m.group(1))
        mo = _MONTH_MAP.get(m.group(2)[:3].lower(), 0)
        year = int(m.group(3))
        if mo and 1 <= day <= 31:
            return f"{year}-{mo:02d}-{day:02d}"

    # Month + year only (e.g. "March 2022", "February 2022")
    m = re.match(
        r"^(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(\d{4})$",
        t,
        re.IGNORECASE,
    )
    if m:
        mo = _MONTH_MAP.get(m.group(1)[:3].lower(), 0)
        year = int(m.group(2))
        if mo:
            return f"{year}-{mo:02d}-01"

    return None


def _weekday_relative(anchor: date, weekday_name: str, direction: str) -> str:
    target = _WEEKDAY_MAP[weekday_name]
    current = anchor.weekday()
    if direction == "last":
        delta = (current - target) % 7 or 7
        return _format_date(anchor - timedelta(days=delta))
    delta = (target - current) % 7 or 7
    return _format_date(anchor + timedelta(days=delta))


def _extract_temporal_entities(text: str, anchor_date: str | None) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_text = str(text or "")

    def _emit(iso_date: str, original: str, confidence: float = 0.95) -> None:
        if iso_date in seen:
            return
        seen.add(iso_date)
        entities.append({
            "id": f"date_{iso_date.replace('-', '_')}",
            "name": iso_date,
            "type": "date",
            "canonical_date": iso_date,
            "content": f"date {iso_date}",
            "confidence": confidence,
            "source_text": original,
        })

    for pattern in (
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{4}\b",
        # Month + year only
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{4}\b",
    ):
        for match in re.finditer(pattern, raw_text, re.IGNORECASE):
            iso_date = _normalize_date_string(match.group(0))
            if iso_date:
                _emit(iso_date, match.group(0))

    anchor = _to_date(anchor_date)
    if not anchor:
        return entities

    rel_terms = {
        "today": anchor,
        "yesterday": anchor - timedelta(days=1),
        "tomorrow": anchor + timedelta(days=1),
    }
    low = raw_text.lower()
    for term, dt in rel_terms.items():
        if re.search(rf"\b{re.escape(term)}\b", low):
            _emit(_format_date(dt), term, confidence=0.9)

    for direction in ("last", "next"):
        for weekday_name in _WEEKDAY_MAP:
            phrase = f"{direction} {weekday_name}"
            if re.search(rf"\b{phrase}\b", low):
                _emit(_weekday_relative(anchor, weekday_name, direction), phrase, confidence=0.9)

    return entities


def _extract_personal_attributes(text: str, speaker: str | None) -> list[dict[str, Any]]:
    if not speaker:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    subject = speaker.strip()
    subject_lc = _snake(subject)

    def _emit(attribute: str, value: str) -> None:
        if not value or len(value) < 1:
            return
        canonical = f"{subject_lc}_{attribute}_{_snake(value[:30])}"
        if canonical in seen:
            return
        seen.add(canonical)
        pretty = attribute.replace("_", " ")
        results.append({
            "id": f"{subject_lc}_{attribute}",
            "name": f"{subject} {pretty}: {value}",
            "type": "personal_attribute",
            "subject": subject,
            "attribute": attribute,
            "value": value,
            "content": f"{subject} {pretty}: {value}",
            "confidence": 0.8,
        })

    # --- Origin / nationality ---
    m = re.search(r"\bmy home country[,\s]+(?:of\s+)?([A-Z][a-zA-Z]+)", text)
    if m:
        _emit("origin", m.group(1))
    m = re.search(r"\b(?:I(?:'m| am) from|I live in|I(?:'ve| have) lived in)\s+([A-Z][a-zA-Z ]{2,30}?)(?:[,.]|$)", text)
    if m:
        _emit("location", m.group(1).strip())
    m = re.search(r"\bI(?:'ve| have) moved to\s+([A-Z][a-zA-Z ]{2,30}?)(?:[,.]|$)", text)
    if m:
        _emit("location", m.group(1).strip())

    # --- Relationship status ---
    if re.search(r"\bsingle\s+(?:parent|mom|dad|father|mother)\b", text, re.I):
        _emit("relationship_status", "single")
    m = re.search(r"\b(?:my\s+)?(?:husband|wife|partner|spouse|boyfriend|girlfriend)\b", text, re.I)
    if m:
        _emit("relationship_status", "in relationship")
    if re.search(r"\b(?:I(?:'m| am) married|got married|we(?:'re| are) married)\b", text, re.I):
        _emit("relationship_status", "married")

    # --- Jobs / occupation ---
    m = re.search(
        r"\bI(?:'m| am)(?: currently| now)? (?:a|an)\s+([a-zA-Z][\w\s]{2,40}?)"
        r"(?:\s+(?:at|for|with|by|in)\b|[,.]|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 2:
            _emit("job", val)
    m = re.search(
        r"\b(?:work(?:ing|s|ed)?(?: as| for)?|job(?:\s+is)?)\s+(?:a|an)?\s*([a-zA-Z][\w\s]{2,40}?)"
        r"(?:[,.]|$|\s+(?:at|for|with|in)\b)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 2:
            _emit("job", val)
    m = re.search(r"\bmy (?:job|career|profession|occupation) (?:is|as)\s+([a-zA-Z][\w\s]{2,40}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("job", m.group(1).strip().rstrip(".,"))

    # --- Hobbies / activities ---
    m = re.search(
        r"\b(?:I(?:'ve| have) (?:been )?)?(?:enjoy|love|like|adore|am into|passion for|hobby is|hobbies are)\s+"
        r"(?:to\s+)?([a-zA-Z][\w\s]{2,50}?)(?:[,.]|$)",
        text, re.I,
    )
    if m:
        _emit("hobby", m.group(1).strip().rstrip(".,"))
    m = re.search(r"\bmy (?:hobby|hobbies|passion|interest) (?:is|are|include)\s+([a-zA-Z][\w\s,]{2,80}?)(?:[.]|$)", text, re.I)
    if m:
        _emit("hobby", m.group(1).strip().rstrip(".,"))
    # "I got into X" / "I started X" as hobby introduction
    m = re.search(
        r"\b(?:I got into|I started|I began|I took up|I picked up)\s+([a-zA-Z][\w\s]{2,40}?)"
        r"(?:\s+(?:because|thanks to|after|through|on|when|and)|[,.]|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 3:
            _emit("hobby", val)
    # "X introduced me to Y" / "friend suggested / advised X"
    m = re.search(
        r"\b(?:(?:a\s+)?friend(?:'s?\s+(?:advice|suggestion|recommendation))?|my\s+friend)\s+"
        r"(?:introduced me to|suggested|recommended|advised\s+(?:me\s+(?:to\s+try\s+)?)?)\s+"
        r"([a-zA-Z][\w\s]{2,40}?)(?:[,.]|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val:
            _emit("hobby_introduced_by_friend", val)

    # --- Sports / martial arts / physical activities ---
    sports = [
        "kickboxing", "taekwondo", "karate", "judo", "boxing", "wrestling",
        "basketball", "football", "soccer", "baseball", "tennis", "volleyball",
        "swimming", "cycling", "running", "hiking", "bowling", "golf", "chess",
        "dancing", "yoga", "gymnastics", "skating", "skiing", "surfing",
    ]
    for sport in sports:
        if re.search(rf"\b(?:played?|doing?|practiced?|trained?|done|taken up|started)\s+{sport}\b", text, re.I):
            _emit("sport_activity", sport)
        elif re.search(rf"\b{sport}\b", text, re.I):
            _emit("sport_activity", sport)

    # --- Favorite things ---
    m = re.search(
        r"\bmy favou?rite\s+([a-zA-Z][\w\s]{0,30}?)\s+(?:is|are|was|were)\s+"
        r"[\"']?([a-zA-Z0-9][\w\s,'\"-]{1,80}?)[\"']?(?:[,.]|$)",
        text, re.I,
    )
    if m:
        category = _snake(m.group(1).strip())
        value = m.group(2).strip().rstrip(".,")
        _emit(f"favorite_{category}", value)
    m = re.search(
        r'\b(?:I (?:love|adore|prefer|like best))\s+["\']([^"\']{2,80})["\']',
        text, re.I,
    )
    if m:
        _emit("favorite", m.group(1).strip())

    # --- Movies / books / music specifically (high-signal for QA) ---
    m = re.search(
        r'\b(?:loved?|finished?|read|reading|re-read|watched?|seeing|saw)\b[^"]{0,30}"([^"\n]{3,80})"',
        text, re.I,
    )
    if m:
        val = m.group(1).strip()
        if re.search(r"\bread|reading\b", text[:m.start()], re.I):
            _emit("book_read", val)
        else:
            _emit("watched", val)
    # "X by Author" book mentions (unquoted)
    m = re.search(
        r'\b(?:read|finished|reading|enjoyed?|loved?|recommend)\s+([A-Z][a-zA-Z\s]{1,40}?)\s+by\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b',
        text, re.I,
    )
    if m:
        _emit("book_read", m.group(1).strip())

    m = re.search(
        r"\bmy favou?rite (?:movie|film|book|band|artist|singer|song|album|show|series|trilogy|franchise|saga)\s+(?:is|are|was)\s+"
        r'["\']?([A-Z][^"\'.\n]{2,60})["\']?(?:[,.]|$)',
        text, re.I,
    )
    if m:
        _emit("favorite_media", m.group(1).strip())
    # Also match "X trilogy/series is my favorite"
    m = re.search(
        r'\b(?:The\s+)?([A-Z][a-zA-Z\s]{2,40}?)(?:\s+(?:trilogy|franchise|saga|series))?\s+'
        r'(?:is|was|are)\s+(?:my|our)\s+(?:favou?rite|all-time\s+favou?rite)\b',
        text, re.I,
    )
    if m:
        _emit("favorite_media", m.group(1).strip())

    # --- Sports position / team role ---
    _POSITIONS = (
        "point guard|shooting guard|power forward|small forward|center|midfielder|"
        "goalkeeper|striker|defender|quarterback|wide receiver|running back|"
        "pitcher|catcher|shortstop|first baseman|outfielder|setter|libero|"
        "captain|coach|manager|goalkeeper|winger|fullback|halfback"
    )
    m = re.search(
        rf"\b(?:play(?:ing|s)?|signed?|position(?:\s+is)?|role(?:\s+is)?)\s+(?:as\s+)?(?:a\s+|the\s+)?({_POSITIONS})\b",
        text, re.I,
    )
    if m:
        _emit("sports_position", m.group(1).lower())
    # "signed/joined [team] as a [position]" — position comes after team name
    m = re.search(
        rf"\b(?:signed?|joined?|contracted?)\s+.{{0,60}}\bas\s+(?:a\s+|the\s+)?({_POSITIONS})\b",
        text, re.I,
    )
    if m:
        _emit("sports_position", m.group(1).lower())
    m = re.search(
        rf"\b(?:a\s+|the\s+)?({_POSITIONS})\s+(?:for|on|at|in)\b",
        text, re.I,
    )
    if m:
        _emit("sports_position", m.group(1).lower())

    # --- Sports team signed with ---
    # Capture only capitalized title-case words as team name, stop at lowercase word
    m = re.search(
        r"\b(?:signed\s+(?:with|for|to)|joined|playing\s+for|contracted\s+(?:with|to))\s+"
        r"(?:[Tt]he\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b",
        text,  # no re.I so [A-Z] stays uppercase-only (proper nouns)
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 3:
            _emit("sports_team", val)

    # --- Challenges / difficulties experienced ---
    m = re.search(
        r"\b(?:challenge(?:d|s)?|struggle(?:d|s)?|difficult(?:y|ies)|hard(?:est)? part|"
        r"obstacle|setback|problem|trouble)\s+(?:is|was|with|of|in|that)?\s*"
        r"([a-zA-Z][\w\s,]{3,60}?)(?:[,.]|\s+(?:but|and|so|however|which)\b|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 4:
            _emit("challenge", val)

    # --- Donations / charity ---
    m = re.search(
        r"\b(?:donated?|gave away|gave to|gifted?)\s+(?:my\s+|a\s+|an\s+|the\s+)?([a-zA-Z][\w\s]{1,40}?)"
        r"(?:\s+(?:to|at|for)\b|[,.]|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 2 and val.lower() not in ("it", "them", "this", "some", "anything"):
            _emit("donation", val)

    # --- Pet adoption ---
    _PET_WORDS = r"dog|puppy|pup|cat|kitten|snake|rabbit|hamster|bird|fish|turtle|horse|pig"
    m = re.search(
        rf"\b(?:adopted?|rescued?|got)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:little\s+)?({_PET_WORDS})\b",
        text, re.I,
    )
    if m:
        pet_type = m.group(1).lower()
        if pet_type in ("puppy", "pup"):
            pet_type = "dog"
        elif pet_type == "kitten":
            pet_type = "cat"
        _emit("has_pet", pet_type)

    # --- Pets / animals ---
    pet_types = ["dog", "cat", "snake", "rabbit", "hamster", "bird", "fish", "turtle", "horse", "pig"]
    for pet in pet_types:
        m = re.search(
            rf"\b(?:my|our|a|an)\s+{pet}(?:s)?\s+(?:named?|called?)\s+([A-Z][a-zA-Z]+)",
            text, re.I,
        )
        if m:
            _emit(f"pet_{pet}", m.group(1))
        m = re.search(
            rf"\b([A-Z][a-zA-Z]+)\s+(?:is|was|my)\s+(?:my\s+)?(?:pet\s+)?{pet}\b",
            text, re.I,
        )
        if m and m.group(1).lower() not in _PERSON_SKIP:
            _emit(f"pet_{pet}", m.group(1))
        # "I have a dog/snake" without name
        if re.search(rf"\bI(?:'ve| have) (?:a|an|two|three|four|five|\d+)\s+{pet}s?\b", text, re.I):
            _emit(f"has_pet", pet)
        # Multiple named pets: "my snakes are named Susie and Seraphim"
        _multi_pat = re.compile(
            r"(?i)\bmy\s+" + pet + r"s?\s+"
            r"(?:are\s+(?:named?|called?)?\s*|(?:named?|called?)\s+)"
            r"(.{3,80})(?:[.!?]|$)"
        )
        m_multi = _multi_pat.search(text)
        if m_multi:
            _COMMON_WORDS = {"Are", "Named", "Called", "The", "And", "My", "Our"}
            names = [n for n in re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', m_multi.group(1))
                     if n not in _COMMON_WORDS]
            if names:
                _emit(f"pet_{pet}_names", ", ".join(names))

    # --- Skills / programming languages ---
    m = re.search(
        r"\b(?:I(?:'ve| have| know| use| work with| specialize in)?)\s+"
        r"(?:been (?:using|working with))?\s*([A-Z][a-zA-Z+#]{1,20}(?:\s+and\s+[A-Z][a-zA-Z+#]{1,20})*)"
        r"\s+(?:programming|language|framework|developer|development)",
        text, re.I,
    )
    if m:
        _emit("programming_skill", m.group(1).strip())
    # Direct language mentions
    langs = ["Python", "C++", "Java", "JavaScript", "TypeScript", "Go", "Rust", "Ruby", "Swift", "Kotlin", "C#", "PHP", "R", "Scala"]
    found_langs = []
    for lang in langs:
        escaped = re.escape(lang)
        # For languages ending in non-word chars (C++, C#), use lookahead instead of \b at end
        if lang[-1] in ("+", "#"):
            pat = rf"\b{escaped}(?=\s|$|[,.])"
        else:
            pat = rf"\b{escaped}\b"
        if re.search(pat, text, re.I):
            found_langs.append(lang)
    if found_langs:
        _emit("programming_languages", ", ".join(found_langs))

    # --- Vehicles / cars ---
    _CAR_BRANDS = (
        r"Prius|Tesla|Honda|Toyota|Ford|Chevy|Chevrolet|BMW|Audi|Mercedes|Ferrari|Lamborghini|"
        r"Civic|Camry|Corolla|Accord|Mazda|Nissan|Hyundai|Kia|Subaru|Volvo|Porsche|Jeep|Dodge"
    )
    m = re.search(
        r"\b(?:bought?|got|purchased?|picked up|driving|drives?|leased?|own(?:ing)?)\s+"
        r"(?:a\s+|an?\s+|my\s+|the\s+)?(?:new\s+)?"
        r"((?:[A-Z][a-zA-Z]+\s+)?(?:" + _CAR_BRANDS + r")(?:\s+\d{3,4}(?:\s+[A-Z]+)?)?)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) < 50:
            _emit("vehicle", val)

    # --- Allergies / health conditions ---
    m = re.search(r"\b(?:I(?:'m| am) allergic to|I have an? allergy to)\s+([a-zA-Z][\w\s]{1,40}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("allergy", m.group(1).strip().rstrip(".,"))
    m = re.search(r"\b(?:I(?:'ve| have) (?:been diagnosed with|suffer from|have))\s+([a-zA-Z][\w\s]{2,40}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("health_condition", m.group(1).strip().rstrip(".,"))

    # --- Studying / field of study ---
    m = re.search(r"\b(?:studying|majoring in|enrolled in)\s+([a-zA-Z][\w\s]{2,50}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("field_of_study", m.group(1).strip().rstrip(".,"))

    # --- Paintings / artwork ---
    m = re.search(
        r"\bpainted\s+(?:the\s+|that\s+|a\s+)?(.+?)"
        r"(?:\s+(?:last|when|while|before|after|ago|in|at|on|and|but)\b|[.!?,]|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val:
            _emit("painted", val)

    # --- Hair color / appearance ---
    m = re.search(r"\b(?:dyed?|colou?red?|changed?|got|painted)\s+(?:my|her|his)?\s*hair\s+(?:to\s+)?([a-zA-Z]+)\b", text, re.I)
    if m:
        _emit("hair_color", m.group(1).lower())
    m = re.search(r"\bmy hair (?:is|was|became|turned)\s+([a-zA-Z]+)\b", text, re.I)
    if m:
        _emit("hair_color", m.group(1).lower())
    # "chose/picked/went with X for my hair"
    m = re.search(r"\b(?:chose?|picked?|went with|selected?|decided on)\s+([a-zA-Z]+)\s+(?:for|as|to)\s+(?:my|his|her)\s+hair\b", text, re.I)
    if m:
        _emit("hair_color", m.group(1).lower())
    # "hair is now X" / "now have X hair"
    m = re.search(r"\bhair (?:is now|became|turned?|dyed)\s+([a-zA-Z]+)\b", text, re.I)
    if m:
        _emit("hair_color", m.group(1).lower())
    m = re.search(r"\bnow\s+have\s+([a-zA-Z]+)\s+hair\b", text, re.I)
    if m:
        _emit("hair_color", m.group(1).lower())
    # "X with purple hair" — common in image captions
    m = re.search(
        r"\b(?:man|woman|person|guy|girl|photo|photography|image|selfie|pic)\b[^.]*?\bwith\s+"
        r"(blonde?|blond|brunette|auburn|red|brown|black|white|grey|gray|silver|golden|purple|"
        r"blue|green|pink|orange|yellow|platinum|copper|chestnut|strawberry|raven|violet|magenta)\s+hair\b",
        text, re.I,
    )
    if m:
        _emit("hair_color", m.group(1).lower())

    # --- Instrument / music ---
    instruments = ["guitar", "piano", "drums", "violin", "cello", "bass", "flute", "trumpet", "saxophone", "ukulele", "banjo", "keyboard", "harmonica"]
    for instr in instruments:
        if re.search(rf"\b(?:play(?:ing|ed|s)?|learned?|practiced?|picked up|started|play)\s+(?:the\s+)?{instr}\b", text, re.I):
            _emit("instrument", instr)
        elif re.search(rf"\b{instr}\b", text, re.I):
            _emit("instrument", instr)

    # --- Movies / shows watched ---
    m = re.search(r'\b(?:watched?|saw|seen|viewing|binge[d-]?)\s+(?:the\s+)?["\']?([A-Z][^"\'.\n]{2,60})["\']?(?:\s+(?:last|this|on|at|in|and|but|with)|\b|$)', text, re.I)
    if m:
        val = m.group(1).strip().rstrip(".,")
        if len(val.split()) <= 8:
            _emit("watched", val)

    # --- Books read ---
    m = re.search(r'\b(?:read|reading|finished|re-read)\s+(?:a\s+book\s+called\s+)?["\']([^"\'.\n]{2,60})["\']', text, re.I)
    if m:
        _emit("book_read", m.group(1).strip())
    m = re.search(r'\bbook(?:\s+called)?\s+["\']([^"\'.\n]{2,60})["\']', text, re.I)
    if m:
        _emit("book_read", m.group(1).strip())

    # --- Games / gaming ---
    m = re.search(r"\b(?:play(?:ing|ed|s)?|played)\s+([A-Z][a-zA-Z\s:]{2,40}?)(?:\s+(?:game|online|with|for|at)|\b|[.,]|$)", text, re.I)
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and 2 < len(val) < 40 and val.lower() not in {"basketball", "football", "soccer", "tennis", "chess", "golf", "bowling"}:
            _emit("game_played", val)

    # --- Food / dietary preference ---
    m = re.search(r"\b(?:I(?:'m| am) (?:a\s+)?(?:vegan|vegetarian|pescatarian|gluten.free|lactose.intolerant))\b", text, re.I)
    if m:
        _emit("diet", m.group(0).replace("I'm", "").replace("I am", "").strip())
    m = re.search(r"\bmy (?:favourite|favorite) food\s+(?:is|are|was)\s+([a-zA-Z][\w\s]{1,40}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("favorite_food", m.group(1).strip().rstrip(".,"))
    # Meat preference: "I prefer chicken over beef", "I love chicken more than"
    _MEATS = r"chicken|beef|pork|lamb|fish|salmon|tuna|shrimp|turkey|duck|venison|tofu"
    m = re.search(
        rf"\b(?:prefer|love|enjoy|like|favour?ite(?:\s+meat)?(?:\s+is)?)\s+(?:eating\s+)?({_MEATS})\b",
        text, re.I,
    )
    if m:
        _emit("favorite_meat", m.group(1).lower())
    # "I eat chicken most often" / "I mostly eat chicken"
    m = re.search(
        rf"\b(?:eat|eating|consume)\s+(?:mostly\s+|mainly\s+|primarily\s+)?({_MEATS})\s+(?:most|mainly|primarily|often|usually|regularly)\b",
        text, re.I,
    )
    if m:
        _emit("favorite_meat", m.group(1).lower())

    # --- Travel / places visited ---
    m = re.search(r"\b(?:visited?|traveled? to|been to|went to|trip to)\s+([A-Z][a-zA-Z ]{2,30}?)(?:[,.]|$|\s+(?:and|in|on|last|this|for))", text, re.I)
    if m:
        _emit("visited_place", m.group(1).strip())

    # --- Awards / achievements ---
    m = re.search(r"\b(?:won|received?|earned?|got|awarded?)\s+(?:a\s+|the\s+|an\s+)?([a-zA-Z][\w\s]{2,50}?(?:award|prize|medal|scholarship|grant|trophy))\b", text, re.I)
    if m:
        _emit("award", m.group(1).strip())

    # --- Relationship events ---
    m = re.search(r"\b(?:got engaged|proposed|engaged to)\b", text, re.I)
    if m:
        _emit("relationship_event", "engaged")
    m = re.search(r"\b(?:got divorced|filed for divorce|separated from)\b", text, re.I)
    if m:
        _emit("relationship_event", "divorced")
    m = re.search(r"\b(?:had a baby|gave birth|expecting|pregnant|new baby)\b", text, re.I)
    if m:
        _emit("relationship_event", "had baby")

    # --- Destress activities ---
    m = re.search(
        r"\b(\w+ing)\s+(?:further|farther|more|longer)?\s*(?:to\s+)?"
        r"(?:de-stress|destress|unwind|relax|clear my mind)\b",
        text, re.I,
    )
    if not m:
        m = re.search(
            r"\b(?:de-stress|destress|unwind|relax)\s+(?:by|with|through)\s+(\w+(?:ing)?)\b",
            text, re.I,
        )
    if m:
        _emit("destress_activity", m.group(1).lower())

    # --- Identity ---
    if re.search(
        r"\bmy\s+(?:trans|transgender)\s+(?:experience|journey|transition|identity|story)\b",
        text, re.I,
    ):
        _emit("identity", "transgender")
    if re.search(r"\b(?:I(?:'m| am) (?:trans|transgender|non-binary|nonbinary|queer|gay|lesbian|bisexual))\b", text, re.I):
        m2 = re.search(r"\b(?:trans|transgender|non-binary|nonbinary|queer|gay|lesbian|bisexual)\b", text, re.I)
        if m2:
            _emit("identity", m2.group(0).lower())

    # --- Career interests ---
    m = re.search(
        r"\b(?:studying|training|working|going into|interested in|pursuing|career in)\s+"
        r"((?:mental health|counseling|counselling|therapy|social work)[^.!?\n]{0,40})",
        text, re.I,
    )
    if m:
        _emit("career_interest", m.group(1).strip().rstrip(".,"))

    # --- Creative / art style preferences ---
    m = re.search(
        r"\b(?:my (?:favou?rite|preferred?|go-to) (?:style|genre|type|kind|medium)\s+(?:of\s+)?(?:painting|art|music|writing|photography|dance|film|cinema|literature)(?:\s+is)?|"
        r"I (?:prefer|love|enjoy|like)\s+(?:contemporary|abstract|classical|modern|impressionist|realist|surrealist|minimalist|expressionist|romantic|baroque|folk|jazz|blues|hip.hop|country|electronic|indie|pop|rock|metal|classical)\s+(?:painting|art|music|style|genre)?)\b",
        text, re.I,
    )
    if m:
        _emit("creative_preference", m.group(0).strip())
    # Simpler: "X style of painting", "X art", etc.
    m = re.search(
        r"\b(contemporary|abstract|classical|modern|impressionist|realist|surrealist|minimalist|expressionist|romantic|baroque)\s+"
        r"(?:is my (?:favou?rite|preferred)|style|painting|art)\b",
        text, re.I,
    )
    if m:
        _emit("art_style", m.group(1).lower())
    m = re.search(
        r"\bmy (?:favou?rite|preferred)\s+(?:style\s+(?:of\s+)?)?(?:painting|art|drawing)\b[^.]{0,30}\s+"
        r"(contemporary|abstract|classical|modern|impressionist|realist|surrealist|minimalist|expressionist|romantic)\b",
        text, re.I,
    )
    if m:
        _emit("art_style", m.group(1).lower())

    # --- Work / focus area (politics, research, main focus) ---
    m = re.search(
        r"\b(?:my (?:main |primary |chief )?(?:focus|work|research|project|area|field)(?:\s+is|\s+involves?|\s+covers?)?|"
        r"(?:I(?:'m| am) (?:focused|working|researching)\s+(?:on|in)))\s+(?:(?:on|in|about|towards?)\s+)?"
        r"([a-zA-Z][\w\s,]{3,60}?)(?:[,.]|\s+(?:and|but|which|that|in order)\b|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 3:
            _emit("focus_area", val)

    # --- Plans / intentions (broader than just goals) ---
    m = re.search(
        r"\b(?:I(?:'m| am) planning to|planning to|I plan to|I intend to|I(?:'m| am) going to|my plan is to|I(?:'m| am) looking (?:into|at))\s+"
        r"([a-zA-Z][\w\s,]{3,60}?)(?:[,.]|\s+(?:this|next|in|over|by|before|after)\b|$)",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val and len(val) > 3:
            _emit("plan", val)

    # --- Pet names ---
    _PET_TYPES = "snakes?|dogs?|cats?|pets?|birds?|fish|rabbits?|hamsters?|turtles?|lizards?|parrots?|ferrets?|puppies|puppy|kittens?|pups?"
    m = re.search(
        rf"\bmy\s+(?:first|second|third|little|other|[0-9]+(?:st|nd|rd|th)?)?\s*(?:{_PET_TYPES})\s+([A-Z][a-zA-Z]+)\b",
        text, re.I,
    )
    if m:
        _COMMON_PET_WORDS = {"are", "named", "called", "is", "was", "the", "and", "my", "a", "an"}
        if m.group(1).lower() not in _COMMON_PET_WORDS:
            _emit("pet_name", m.group(1))
    # "This is [Name]" pattern (in pet context)
    if re.search(rf"\b(?:{_PET_TYPES})\b", text, re.I):
        m = re.search(r"\bThis is ([A-Z][a-zA-Z]+)\.?\s*$", text)
        if m:
            _emit("pet_name", m.group(1))

    # --- Research topics ---
    m = re.search(r"\bresearch(?:ing|ed)?\s+([a-zA-Z][a-zA-Z\s]{1,50}?)(?:\s+(?:and\b|it\b|has\b|is\b|to\b|so\b|that\b)|[,;.!?]|$)", text, re.I)
    if m:
        raw_val = m.group(1).strip().rstrip(".,!?;-")
        # Keep first 3 meaningful words only
        _STOP_WORDS = {"a", "an", "it", "is", "to", "the", "and", "or", "but", "in", "on", "at", "for", "of", "this", "these", "those", "my", "me", "we", "our"}
        words = [w.rstrip(".,!?;-") for w in raw_val.split()]
        content_words = [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 1]
        val = " ".join(content_words[:3])
        if val and 2 < len(val) < 60 and not val.lower().startswith(("a lot", "lot", "more", "further", "networking")):
            _emit("research_topic", val)

    # --- Collections ---
    m = re.search(r"\b(?:I (?:collect|have a collection of))\s+([a-zA-Z][\w\s]{2,50}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("collection", m.group(1).strip().rstrip(".,"))

    # --- Goals ---
    m = re.search(r"\b(?:my goal is to|I (?:want|plan|aim) to)\s+([a-zA-Z][\w\s,]{3,80}?)(?:[,.]|$)", text, re.I)
    if m:
        _emit("goal", m.group(1).strip().rstrip(".,"))

    # --- Tattoo ---
    m = re.search(
        r"\b(?:I have|I got|got|have)\s+a\s+tattoo\s+of\s+([a-zA-Z][\w\s]{1,40}?)(?:[,.]|$|\s+(?:on|at|in|and|but)\b)",
        text, re.I,
    )
    if m:
        _emit("tattoo", m.group(1).strip().rstrip(".,"))
    m = re.search(
        r"\b([a-zA-Z][\w\s]{1,30}?)\s+tattoo\b",
        text, re.I,
    )
    if m:
        val = m.group(1).strip().rstrip(".,")
        if val.lower() not in {"a", "the", "my", "his", "her", "their", "small", "large", "big", "cute", "cool", "new"}:
            _emit("tattoo", val)

    # --- Favorite books (list form) ---
    m = re.search(
        r"\bmy favou?rite books?\s+(?:are|is|include|were)\s+([A-Z][^.\n]{3,80}?)(?:[,.]?\s*(?:and|but|or|I|We)\b|[.!?]|$)",
        text, re.I,
    )
    if m:
        _emit("favorite_books", m.group(1).strip().rstrip(",."))
    # "I love reading X and Y" → favorite_books
    m = re.search(
        r"\b(?:I love|I enjoy|I adore)\s+reading\s+([A-Z][^.\n]{3,80}?)(?:\s+(?:and|or)\s+[A-Z][\w\s]{1,40})?(?:[.!?]|$)",
        text, re.I,
    )
    if m:
        _emit("favorite_books", m.group(0).replace("I love reading", "").replace("I enjoy reading", "").replace("I adore reading", "").strip().rstrip(".,"))

    # --- Musical instrument ---
    _INSTRUMENTS = r"piano|guitar|drums?|violin|cello|trumpet|flute|saxophone|bass|keyboard|ukulele|clarinet|viola|harp|banjo|mandolin"
    m = re.search(
        rf"\b(?:play(?:ing|s|ed)?|started?\s+playing|resume[d]?\s+playing|learning?)\s+(?:the\s+)?({_INSTRUMENTS})\b",
        text, re.I,
    )
    if m:
        _emit("instrument", m.group(1).lower())
    m = re.search(
        rf"\b(?:the\s+)?({_INSTRUMENTS})\s+(?:player|lesson|practice|class)\b",
        text, re.I,
    )
    if m:
        _emit("instrument", m.group(1).lower())

    return results


def extract_temporal_events(
    text: str, anchor_date: str | None, speaker: str | None
) -> list[dict[str, Any]]:
    """
    Extract (person, activity, date) temporal event triples.
    These become temporal_event entities in the graph, indexed by canonical_date.
    """
    if not speaker:
        return []

    results: list[dict[str, Any]] = []
    raw = str(text or "")
    subject = speaker.strip()
    subject_lc = _snake(subject)

    # Find all date expressions in text (full dates + month+year + year-only)
    date_patterns = [
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b",
        r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b",
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{4})\b",
        r"\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{4})\b",
        # Year-only: "in 2019", "back in 2020", "since 2021", "first in 2018"
        r"\b(?:in|since|back in|during|around|first in|started in|began in|by)\s+(20\d{2})\b",
    ]

    found_dates: list[tuple[int, int, str]] = []  # (start, end, iso_date)
    for pat in date_patterns:
        for m in re.finditer(pat, raw, re.IGNORECASE):
            raw_date = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            # Year-only → use YYYY-01-01
            if re.match(r"^20\d{2}$", raw_date.strip()):
                iso = f"{raw_date.strip()}-01-01"
            else:
                iso = _normalize_date_string(raw_date)
            if iso:
                found_dates.append((m.start(), m.end(), iso))

    # Relative year resolution anchored to the utterance date
    if anchor_date:
        try:
            anchor_year = int(anchor_date[:4])
        except (ValueError, IndexError):
            anchor_year = None

        if anchor_year:
            _rel_year_patterns = [
                (r"\b(?:last|past)\s+year\b", anchor_year - 1),
                (r"\ba\s+year\s+ago\b", anchor_year - 1),
                (r"\b(?:one|1)\s+year\s+ago\b", anchor_year - 1),
                (r"\b(?:two|2)\s+years?\s+ago\b", anchor_year - 2),
                (r"\ba\s+couple\s+(?:of\s+)?years?\s+ago\b", anchor_year - 2),
                (r"\b(?:three|3)\s+years?\s+ago\b", anchor_year - 3),
                (r"\ba\s+few\s+years?\s+ago\b", anchor_year - 3),
                (r"\b(?:four|4)\s+years?\s+ago\b", anchor_year - 4),
                (r"\b(?:five|5)\s+years?\s+ago\b", anchor_year - 5),
                (r"\bthis\s+year\b", anchor_year),
                (r"\bnext\s+year\b", anchor_year + 1),
            ]
            for pat, year in _rel_year_patterns:
                if 2000 <= year <= 2099:
                    for m in re.finditer(pat, raw, re.IGNORECASE):
                        iso = f"{year}-01-01"
                        found_dates.append((m.start(), m.end(), iso))

    # Relative day resolution anchored to the utterance date
    if anchor_date:
        anchor_dt = _to_date(anchor_date)
        if anchor_dt:
            _next_month_dt = (anchor_dt.replace(day=28) + timedelta(days=4)).replace(day=1)
            _fixed_rel_patterns = [
                (r"\byesterday\b", anchor_dt - timedelta(days=1)),
                (r"\blast\s+night\b", anchor_dt - timedelta(days=1)),
                (r"\bthe day before yesterday\b", anchor_dt - timedelta(days=2)),
                (r"\b(?:last|this past)\s+week\b", anchor_dt - timedelta(days=7)),
                (r"\btwo\s+(?:days?|nights?)\s+ago\b", anchor_dt - timedelta(days=2)),
                (r"\ba\s+(?:few|couple\s+of)\s+days?\s+ago\b", anchor_dt - timedelta(days=3)),
                (r"\blast\s+month\b", (anchor_dt.replace(day=1) - timedelta(days=1)).replace(day=1)),
                (r"\bnext\s+month\b", _next_month_dt),
                (r"\bthis\s+(?:coming\s+)?month\b", anchor_dt.replace(day=1)),
            ]
            for pat, resolved_dt in _fixed_rel_patterns:
                for m in re.finditer(pat, raw, re.IGNORECASE):
                    iso = _format_date(resolved_dt)
                    found_dates.append((m.start(), m.end(), iso))
            # Per-weekday: "last Sunday" → correct day-of-week delta, not always -7
            for wd_name in _WEEKDAY_MAP:
                pat = rf"\b(?:last|this past)\s+{wd_name}\b"
                for m in re.finditer(pat, raw, re.IGNORECASE):
                    iso = _weekday_relative(anchor_dt, wd_name, "last")
                    found_dates.append((m.start(), m.end(), iso))
            for wd_name in _WEEKDAY_MAP:
                pat = rf"\bnext\s+{wd_name}\b"
                for m in re.finditer(pat, raw, re.IGNORECASE):
                    iso = _weekday_relative(anchor_dt, wd_name, "next")
                    found_dates.append((m.start(), m.end(), iso))

    # Also try anchor_date as a fallback for activity-rich utterances
    if not found_dates and anchor_date:
        found_dates = [(0, 0, anchor_date)]

    if not found_dates:
        return results

    # Activity verbs that indicate datable events (past, present, future tense forms)
    _ACTIVITY_VERBS = (
        # Past tense
        "went|visited|attended|started|began|finished|completed|played|watched|saw|read|"
        "bought|sold|adopted|donated|moved|joined|left|quit|resumed|won|lost|signed|"
        "hired|fired|married|divorced|had|got|received|launched|opened|closed|created|"
        "built|painted|wrote|published|recorded|achieved|reached|scored|bowled|ran|"
        "competed|graduated|promoted|relocated|rescued|found|discovered|learned|earned|"
        "volunteered|made|cooked|ate|drank|drove|flew|traveled|met|celebrated|"
        "adopted|performed|competed|presented|submitted|completed|"
        # Present/future tense and participles (needed for 'next month', 'tomorrow')
        "visit|attend|attending|start|begin|finish|complete|play|playing|watch|"
        "perform|performing|compete|competing|travel|traveling|move|moving|join|joining|"
        "graduate|graduating|launch|launching|present|presenting|run|running|"
        "volunteer|volunteering|celebrate|celebrating|meet|meeting"
    )
    combined_activity = r"\b(?:" + _ACTIVITY_VERBS + r")\b"

    seen: set[str] = set()

    for (dstart, dend, iso_date) in found_dates:
        # Look for activity verb in a window around the date
        window_start = max(0, dstart - 150)
        window_end = min(len(raw), dend + 150)
        window = raw[window_start:window_end]

        m = re.search(combined_activity, window, re.IGNORECASE)
        if m:
            # Extract a short activity description
            # Take text around the verb (40 chars each side, within window)
            verb_start = m.start()
            activity_text = window[max(0, verb_start - 20):min(len(window), verb_start + 60)].strip()
            activity_text = re.sub(r"\s+", " ", activity_text).strip()
            if not activity_text:
                continue

            event_id = f"tevt_{subject_lc}_{iso_date.replace('-', '_')}"
            if event_id in seen:
                continue
            seen.add(event_id)

            content = f"{subject} on {iso_date}: {activity_text}"
            results.append({
                "id": event_id,
                "name": f"{subject} event {iso_date}",
                "type": "temporal_event",
                "subject": subject,
                "attribute": "activity",
                "value": activity_text[:200],
                "canonical_date": iso_date,
                "content": content[:400],
                "confidence": 0.75,
            })

    return results


def _looks_like_person_name(token: str) -> bool:
    t = token.strip()
    return len(t) >= 3 and t[0].isupper() and t.lower() not in _PERSON_SKIP


def _extract_people(text: str, speaker: str | None) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _emit(name: str, entity_type: str, confidence: float) -> None:
        canonical = _snake(name)
        if canonical in seen:
            return
        seen.add(canonical)
        entities.append({
            "id": canonical,
            "name": name,
            "type": entity_type,
            "content": name,
            "confidence": confidence,
        })

    if speaker:
        _emit(speaker, "speaker", 1.0)

    for token in re.findall(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*)\b", text):
        if _looks_like_person_name(token):
            _emit(token, "person", 0.5)

    return entities


def extract_v2_entities(text: str) -> list[dict[str, Any]]:
    """Extract high-signal graph entities from one utterance."""
    clean_text, anchor_date, speaker = _strip_prefixes(text)
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add_many(items: list[dict[str, Any]]) -> None:
        for item in items:
            entity_id = _snake(str(item.get("id") or item.get("name") or ""))
            if not entity_id or entity_id in seen:
                continue
            seen.add(entity_id)
            item["id"] = entity_id
            entities.append(item)

    if anchor_date:
        _add_many([{
            "id": f"date_{anchor_date.replace('-', '_')}",
            "name": anchor_date,
            "type": "date_anchor",
            "canonical_date": anchor_date,
            "content": f"anchor date {anchor_date}",
            "confidence": 1.0,
        }])

    _add_many(_extract_people(clean_text, speaker))
    _add_many(_extract_temporal_entities(clean_text, anchor_date))
    _add_many(_extract_personal_attributes(clean_text, speaker))
    _add_many(extract_temporal_events(clean_text, anchor_date, speaker))
    return entities
