import re
import time
from typing import List, Tuple, Set, Optional

# --- Shared Helpers ---
_SUBJECT_STOPWORDS = {"i", "we", "they", "he", "she", "it", "this", "that", "there", "here", "someone", "anyone"}
_RELATIONSHIP_STATUS_VALUES = {"single", "married", "engaged", "divorced", "widowed", "dating", "in a relationship"}
_IDENTITY_HINT_VALUES = {"transgender", "transgender woman", "transgender man", "non-binary", "nonbinary", "genderfluid", "cisgender"}

def _normalize_span(value: str) -> str:
    normalized = re.sub(r"\s+", " ", (value or "").strip())
    return normalized.strip(" .,:;!?\"'()[]{}")

def _normalize_subject(value: str) -> str:
    normalized = _normalize_span(value).lower()
    normalized = re.sub(r"^(the|a|an)\s+", "", normalized)
    normalized = re.sub(r"\b(?:is|are|was|were|am|be)$", "", normalized).strip()
    if not normalized or normalized in _SUBJECT_STOPWORDS:
        return ""
    tokens = re.findall(r"[a-z0-9_\-]+", normalized)
    if not tokens:
        return ""
    if any(token in _SUBJECT_STOPWORDS for token in tokens):
        return ""
    return normalized

def _normalize_object(value: str) -> str:
    normalized = _normalize_span(value).lower()
    normalized = re.sub(r"^(a|an|the)\s+", "", normalized)
    return normalized[:200]

# --- Old Logic (faithfully reimplemented from git baseline) ---
def old_extract_fact_candidates(text: str) -> List[Tuple[str, str, str, float]]:
    cleaned = " ".join((text or "").split())
    if not cleaned: return []
    candidates = []
    patterns = [
        (r"([A-Za-z0-9_\- ]+?)\s+phone\s+is\s+([A-Za-z0-9+\- ]{5,30})", "phone", 0.82),
        (r"([A-Za-z0-9_\- ]+?)\s+address\s+is\s+([A-Za-z0-9,\- ]{5,120})", "address", 0.78),
        (r"([A-Za-z0-9_\- ]+?)\s+plan\s+is\s+([A-Za-z0-9_\- ]{3,80})", "plan", 0.80),
        (r"([A-Za-z0-9_\- ]+?)\s+status\s+is\s+([A-Za-z0-9_\- ]{3,80})", "status", 0.72),
    ]
    for pattern, predicate, confidence in patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            subject = match.group(1).strip().lower()
            obj = match.group(2).strip().lower()
            if subject and obj:
                candidates.append((subject, predicate, obj, confidence))
    return candidates

# --- Current Logic (extracted from working tree) ---
def _infer_predicate_from_attr(attr: str, obj: str) -> str:
    attr_norm = (attr or "").lower().strip()
    obj_norm = (obj or "").lower().strip()
    if "relationship" in attr_norm: return "relationship_status"
    if "identity" in attr_norm or "gender" in attr_norm: return "identity"
    if "home country" in attr_norm or "origin" in attr_norm: return "origin_country"
    if "country" in attr_norm: return "country"
    if "status" in attr_norm and obj_norm in _RELATIONSHIP_STATUS_VALUES: return "relationship_status"
    if "status" in attr_norm: return "status"
    if "job" in attr_norm or "occupation" in attr_norm or "role" in attr_norm: return "occupation"
    if "plan" in attr_norm: return "plan"
    if "phone" in attr_norm: return "phone"
    if "address" in attr_norm: return "address"
    return "attribute"

def _infer_predicate_from_state(obj: str) -> str:
    obj_norm = (obj or "").lower().strip()
    if obj_norm in _RELATIONSHIP_STATUS_VALUES: return "relationship_status"
    if obj_norm in _IDENTITY_HINT_VALUES: return "identity"
    return "status"

def _append_candidate(bag, seen, subject, predicate, obj, confidence):
    sub = _normalize_subject(subject)
    pred = _normalize_span(predicate).lower().replace(" ", "_")
    ob = _normalize_object(obj)
    if not sub or not pred or not ob: return
    key = (sub, pred, ob)
    if key in seen: return
    seen.add(key)
    bag.append((sub, pred, ob, confidence))

def current_extract_fact_candidates(text: str, entities: object = None) -> List[Tuple[str, str, str, float]]:
    cleaned = " ".join((text or "").split())
    if not cleaned: return []
    candidates = []
    seen = set()
    subject_hints = [] # Simplified for benchmark
    speaker_hints: list[str] = []
    for match in re.finditer(r"(?P<speaker>[A-Za-z][A-Za-z0-9_\-]{1,40})\s*[:,-]\s*I\b", cleaned, flags=re.IGNORECASE):
        speaker = _normalize_subject(match.group("speaker"))
        if speaker and speaker not in speaker_hints: speaker_hints.append(speaker)
    if not subject_hints and speaker_hints: subject_hints = speaker_hints
    single_subject_hint = subject_hints[0] if len(subject_hints) == 1 else ""

    attribute_patterns = [
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>phone)\s+is\s+(?P<object>[A-Za-z0-9+\- ]{5,30})", 0.84),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>address)\s+is\s+(?P<object>[A-Za-z0-9,\- ]{5,120})", 0.80),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>plan|role|occupation|job|identity|gender|status|relationship status|home country|country)\s+is\s+(?P<object>[A-Za-z0-9_\- ]{2,120})", 0.76),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>identity|gender|status|relationship status)\s*[:=]\s*(?P<object>[A-Za-z0-9_\- ]{2,120})", 0.74),
    ]
    for pattern, confidence in attribute_patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            predicate = _infer_predicate_from_attr(match.group("attr"), match.group("object"))
            _append_candidate(candidates, seen, match.group("subject"), predicate, match.group("object"), confidence)

    speaker_state_pattern = r"(?P<speaker>[A-Za-z][A-Za-z0-9_\-]{1,40})\s*[:,-]\s*I\s*(?:am|'m)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80}?)(?=(?:\s+and\s+I\b)|(?:\s+but\s+I\b)|[.,;!?]|$)"
    for match in re.finditer(speaker_state_pattern, cleaned, flags=re.IGNORECASE):
        obj = match.group("object")
        _append_candidate(candidates, seen, match.group("speaker"), _infer_predicate_from_state(obj), obj, 0.73)

    if single_subject_hint:
        first_person_state_pattern = r"\bI\s*(?:am|'m)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80}?)(?=(?:\s+and\s+I\b)|(?:\s+but\s+I\b)|[.,;!?]|$)"
        for match in re.finditer(first_person_state_pattern, cleaned, flags=re.IGNORECASE):
            obj = match.group("object")
            _append_candidate(candidates, seen, single_subject_hint, _infer_predicate_from_state(obj), obj, 0.66)

    relation_patterns = [
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+moved\s+from\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "moved_from", 0.82),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+moved\s+to\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "moved_to", 0.82),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+works\s+as\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "occupation", 0.79),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+works\s+(?:at|for)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "works_for", 0.78),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+lives\s+in\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "lives_in", 0.78),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?:researched|researches)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{2,120})", "researched", 0.74),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+adopted\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{2,120})", "adopted", 0.74),
    ]
    for pattern, predicate, confidence in relation_patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            _append_candidate(candidates, seen, match.group("subject"), predicate, match.group("object"), confidence)
    return candidates

# --- Benchmark ---
samples = [
    "Caroline: I'm single and I moved from Sweden four years ago.",
    "John phone is 555-123-4567",
    "James works as a game developer and lives in Toronto."
]
iters = 5000

print(f"{'Sample':<60} | {'Old (ms)':<10} | {'New (ms)':<10} | {'Diff'}")
print("-" * 100)

for s in samples:
    # Old
    t0 = time.perf_counter()
    for _ in range(iters):
        old_extract_fact_candidates(s)
    t1 = time.perf_counter()
    old_avg = ((t1 - t0) / iters) * 1000

    # Current
    t0 = time.perf_counter()
    for _ in range(iters):
        current_extract_fact_candidates(s)
    t1 = time.perf_counter()
    current_avg = ((t1 - t0) / iters) * 1000

    diff_pct = ((current_avg - old_avg) / old_avg) * 100
    print(f"{s[:57]+'...':<60} | {old_avg:10.4f} | {current_avg:10.4f} | {diff_pct:+.1f}%")

