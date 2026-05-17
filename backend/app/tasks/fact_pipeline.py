"""Fact extraction and contradiction detection tasks (PR3)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

from celery.utils.log import get_task_logger
from sqlalchemy import and_, select

from app.core.celery_app import celery_app
from app.core.database import async_session_factory, set_tenant_context
from app.models.contradiction import Contradiction, ContradictionSeverity
from app.models.memory import MemoryMetadata
from app.models.memory_fact import MemoryFact, MemoryFactStatus
from app.tasks.async_runtime import run_async


logger = get_task_logger(__name__)


_SUBJECT_STOPWORDS = {
    "i",
    "we",
    "they",
    "he",
    "she",
    "it",
    "this",
    "that",
    "there",
    "here",
    "someone",
    "anyone",
}

_RELATIONSHIP_STATUS_VALUES = {
    "single",
    "married",
    "engaged",
    "divorced",
    "widowed",
    "dating",
    "in a relationship",
}

_IDENTITY_HINT_VALUES = {
    "transgender",
    "transgender woman",
    "transgender man",
    "non-binary",
    "nonbinary",
    "genderfluid",
    "cisgender",
}


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
    # Reject malformed subjects such as "m single and i" from clause over-capture.
    if any(token in _SUBJECT_STOPWORDS for token in tokens):
        return ""
    return normalized


def _normalize_object(value: str) -> str:
    normalized = _normalize_span(value).lower()
    normalized = re.sub(r"^(a|an|the)\s+", "", normalized)
    return normalized[:200]


def _extract_subject_hints(entities: object) -> list[str]:
    hints: list[str] = []

    def _visit(node: object) -> None:
        if isinstance(node, str):
            subject = _normalize_subject(node)
            if subject:
                hints.append(subject)
            return
        if isinstance(node, dict):
            for key in ("name", "entity", "canonical", "original", "subject"):
                if isinstance(node.get(key), str):
                    _visit(node.get(key))
            values = node.get("values")
            if isinstance(values, list):
                for item in values:
                    _visit(item)
            return
        if isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(entities)
    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        if hint not in seen:
            seen.add(hint)
            deduped.append(hint)
    return deduped


def _infer_predicate_from_attr(attr: str, obj: str) -> str:
    attr_norm = (attr or "").lower().strip()
    obj_norm = (obj or "").lower().strip()

    if "relationship" in attr_norm:
        return "relationship_status"
    if "identity" in attr_norm or "gender" in attr_norm:
        return "identity"
    if "home country" in attr_norm or "origin" in attr_norm:
        return "origin_country"
    if "country" in attr_norm:
        return "country"
    if "status" in attr_norm and obj_norm in _RELATIONSHIP_STATUS_VALUES:
        return "relationship_status"
    if "status" in attr_norm:
        return "status"
    if "job" in attr_norm or "occupation" in attr_norm or "role" in attr_norm:
        return "occupation"
    if "plan" in attr_norm:
        return "plan"
    if "phone" in attr_norm:
        return "phone"
    if "address" in attr_norm:
        return "address"
    return "attribute"


def _infer_predicate_from_state(obj: str) -> str:
    obj_norm = (obj or "").lower().strip()
    if obj_norm in _RELATIONSHIP_STATUS_VALUES:
        return "relationship_status"
    if obj_norm in _IDENTITY_HINT_VALUES:
        return "identity"
    return "status"


def _append_candidate(
    bag: list[tuple[str, str, str, float]],
    seen: set[tuple[str, str, str]],
    *,
    subject: str,
    predicate: str,
    obj: str,
    confidence: float,
) -> None:
    sub = _normalize_subject(subject)
    pred = _normalize_span(predicate).lower().replace(" ", "_")
    ob = _normalize_object(obj)
    if not sub or not pred or not ob:
        return
    key = (sub, pred, ob)
    if key in seen:
        return
    seen.add(key)
    bag.append((sub, pred, ob, confidence))


def _extract_fact_candidates(text: str, entities: object = None) -> list[tuple[str, str, str, float]]:
    """Heuristic extraction for world-model friendly fact statements.

    Uses generalized attribute and relation templates plus subject hints.
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []

    candidates: list[tuple[str, str, str, float]] = []
    seen: set[tuple[str, str, str]] = set()
    subject_hints = _extract_subject_hints(entities)
    speaker_hints: list[str] = []
    for match in re.finditer(
        r"(?P<speaker>[A-Za-z][A-Za-z0-9_\-]{1,40})\s*[:,-]\s*I\b",
        cleaned,
        flags=re.IGNORECASE,
    ):
        speaker = _normalize_subject(match.group("speaker"))
        if speaker and speaker not in speaker_hints:
            speaker_hints.append(speaker)
    if not subject_hints and speaker_hints:
        subject_hints = speaker_hints
    single_subject_hint = subject_hints[0] if len(subject_hints) == 1 else ""
    reliable_single_subject_hint = speaker_hints[0] if len(speaker_hints) == 1 else ""

    attribute_patterns = [
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>phone)\s+is\s+(?P<object>[A-Za-z0-9+\- ]{5,30})", 0.84),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>address)\s+is\s+(?P<object>[A-Za-z0-9,\- ]{5,120})", 0.80),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>plan|role|occupation|job|identity|gender|status|relationship status|home country|country)\s+is\s+(?P<object>[A-Za-z0-9_\- ]{2,120})", 0.76),
        (r"(?P<subject>[A-Za-z][A-Za-z0-9_\- ]{1,80})\s+(?P<attr>identity|gender|status|relationship status)\s*[:=]\s*(?P<object>[A-Za-z0-9_\- ]{2,120})", 0.74),
    ]
    for pattern, confidence in attribute_patterns:
        for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
            predicate = _infer_predicate_from_attr(match.group("attr"), match.group("object"))
            _append_candidate(
                candidates,
                seen,
                subject=match.group("subject"),
                predicate=predicate,
                obj=match.group("object"),
                confidence=confidence,
            )

    # Speaker-aware first-person facts, e.g. "Caroline: I'm single".
    speaker_state_pattern = (
        r"(?P<speaker>[A-Za-z][A-Za-z0-9_\-]{1,40})\s*[:,-]\s*"
        r"I\s*(?:am|'m)\s+"
        r"(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80}?)"
        r"(?=(?:\s+and\s+I\b)|(?:\s+but\s+I\b)|[.,;!?]|$)"
    )
    for match in re.finditer(speaker_state_pattern, cleaned, flags=re.IGNORECASE):
        obj = match.group("object")
        _append_candidate(
            candidates,
            seen,
            subject=match.group("speaker"),
            predicate=_infer_predicate_from_state(obj),
            obj=obj,
            confidence=0.73,
        )

    # First-person state fallback is only allowed when we have an explicit speaker
    # anchor in the text (e.g., "Caroline: I'm ...").
    # Do not infer first-person subject from entity metadata alone — this causes
    # cross-person contamination in relationship/identity facts.
    if reliable_single_subject_hint:
        first_person_state_pattern = (
            r"\bI\s*(?:am|'m)\s+"
            r"(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80}?)"
            r"(?=(?:\s+and\s+I\b)|(?:\s+but\s+I\b)|[.,;!?]|$)"
        )
        for match in re.finditer(first_person_state_pattern, cleaned, flags=re.IGNORECASE):
            obj = match.group("object")
            _append_candidate(
                candidates,
                seen,
                subject=reliable_single_subject_hint,
                predicate=_infer_predicate_from_state(obj),
                obj=obj,
                confidence=0.66,
            )

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
            _append_candidate(
                candidates,
                seen,
                subject=match.group("subject"),
                predicate=predicate,
                obj=match.group("object"),
                confidence=confidence,
            )

    if reliable_single_subject_hint:
        first_person_relation_patterns = [
            (r"\bI\s+moved\s+from\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "moved_from", 0.70),
            (r"\bI\s+moved\s+to\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "moved_to", 0.70),
            (r"\bI\s+work\s+as\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "occupation", 0.68),
            (r"\bI\s+work\s+(?:at|for)\s+(?P<object>[A-Za-z][A-Za-z0-9_\- ]{1,80})", "works_for", 0.68),
        ]
        for pattern, predicate, confidence in first_person_relation_patterns:
            for match in re.finditer(pattern, cleaned, flags=re.IGNORECASE):
                _append_candidate(
                    candidates,
                    seen,
                    subject=reliable_single_subject_hint,
                    predicate=predicate,
                    obj=match.group("object"),
                    confidence=confidence,
                )

    return candidates


async def extract_facts_from_memory(*, org_id: str, memory_id: str, actor_user_id: str) -> dict:
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            memory = await db.get(MemoryMetadata, memory_id)
            if not memory or memory.organization_id != org_id:
                raise ValueError(f"Memory {memory_id} not found for organization {org_id}")

            source_text = " ".join([memory.title or "", memory.content_preview or ""])
            candidates = _extract_fact_candidates(source_text, entities=memory.entities)

            created_ids: list[str] = []
            valid_from = memory.created_at or datetime.now(timezone.utc)

            for subject, predicate, obj, confidence in candidates:
                existing_stmt = select(MemoryFact).where(
                    and_(
                        MemoryFact.organization_id == org_id,
                        MemoryFact.subject == subject,
                        MemoryFact.predicate == predicate,
                        MemoryFact.object == obj,
                        MemoryFact.source_memory_id == memory_id,
                    )
                )
                existing = (await db.execute(existing_stmt)).scalar_one_or_none()
                if existing:
                    continue

                fact = MemoryFact(
                    id=str(uuid4()),
                    organization_id=org_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=confidence,
                    source_memory_id=memory_id,
                    valid_from=valid_from,
                    valid_to=None,
                    supersedes_fact_id=None,
                    status=MemoryFactStatus.ACTIVE,
                    contradiction_group_id=None,
                )
                db.add(fact)
                created_ids.append(fact.id)

            await db.flush()

            return {
                "memory_id": memory_id,
                "created_count": len(created_ids),
                "fact_ids": created_ids,
            }


async def detect_contradictions(*, org_id: str, fact_ids: list[str], actor_user_id: str) -> dict:
    async with async_session_factory() as db:
        async with db.begin():
            await set_tenant_context(db, actor_user_id, org_id, roles="system,org_admin", clearance_level=4)

            if not fact_ids:
                return {"created": 0, "pairs": []}

            facts_stmt = select(MemoryFact).where(
                and_(
                    MemoryFact.organization_id == org_id,
                    MemoryFact.id.in_(fact_ids),
                )
            )
            new_facts = list((await db.execute(facts_stmt)).scalars().all())

            created_pairs: list[tuple[str, str]] = []

            for fact in new_facts:
                other_stmt = select(MemoryFact).where(
                    and_(
                        MemoryFact.organization_id == org_id,
                        MemoryFact.subject == fact.subject,
                        MemoryFact.predicate == fact.predicate,
                        MemoryFact.id != fact.id,
                        MemoryFact.status == MemoryFactStatus.ACTIVE,
                    )
                )
                others = list((await db.execute(other_stmt)).scalars().all())

                for other in others:
                    if other.object.strip().lower() == fact.object.strip().lower():
                        continue

                    contradiction = Contradiction(
                        id=str(uuid4()),
                        organization_id=org_id,
                        fact_a=other.id,
                        fact_b=fact.id,
                        reason=(
                            f"Conflicting values for {fact.subject}.{fact.predicate}: "
                            f"'{other.object}' vs '{fact.object}'"
                        ),
                        severity=ContradictionSeverity.HIGH,
                        created_at=datetime.now(timezone.utc),
                        resolved_at=None,
                    )
                    db.add(contradiction)

                    group_id = str(uuid4())
                    fact.status = MemoryFactStatus.DISPUTED
                    other.status = MemoryFactStatus.DISPUTED
                    fact.contradiction_group_id = group_id
                    other.contradiction_group_id = group_id

                    created_pairs.append((other.id, fact.id))

            await db.flush()

            return {
                "created": len(created_pairs),
                "pairs": created_pairs,
            }


@celery_app.task(bind=True, max_retries=5, autoretry_for=(Exception,), dont_autoretry_for=(ValueError,), retry_backoff=True)
def fact_extractor_task(
    self,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    if storage != "long_term":
        return {"status": "skipped", "reason": "memory_not_long_term", "memory_id": memory_id}

    actor_user_id = initiator_user_id or "00000000-0000-0000-0000-000000000001"
    extraction = run_async(extract_facts_from_memory(org_id=org_id, memory_id=memory_id, actor_user_id=actor_user_id))
    contradiction = run_async(detect_contradictions(org_id=org_id, fact_ids=extraction["fact_ids"], actor_user_id=actor_user_id))

    return {
        "status": "ok",
        "org_id": org_id,
        "memory_id": memory_id,
        "trace_id": trace_id,
        "extraction": extraction,
        "contradictions": contradiction,
    }


def enqueue_fact_pipeline(
    *,
    org_id: str,
    memory_id: str,
    initiator_user_id: str | None = None,
    initiator_roles: str = "",
    initiator_clearance_level: int = 0,
    trace_id: str | None = None,
    storage: str = "long_term",
):
    broker = celery_app.conf.broker_url
    if not broker or str(broker).startswith("memory://"):
        return None

    try:
        return fact_extractor_task.si(
            org_id=org_id,
            memory_id=memory_id,
            initiator_user_id=initiator_user_id,
            trace_id=trace_id,
            storage=storage,
        ).apply_async(retry=False)
    except Exception as exc:
        logger.warning("fact pipeline enqueue failed for memory_id=%s: %s", memory_id, exc)
        return None
