"""Cognitive Gateway Service — Phase 49 (revised).

Exposes Ninai's full intelligence stack as five simple verbs for downstream
consumers (Tier-3 developers, AI agent platforms, enterprise integrations):

  write(content, ...)      → store + enrich a memory record
  read(query, ...)         → retrieve + rank context
  decide(context, ...)     → run anomaly detection, return decision + confidence
  plan(goal, ...)          → decompose goal into real subtasks, return ordered steps
  explain(memory_id)       → return audit trail for last decision on a memory

decide() delegates to AnomalyDetectionAgent's heuristic for real signal
detection rather than plain keyword scanning.

plan() uses GoalDecompositionAgent's extraction helpers (extract_subtasks,
detect_blocking_subtask) to pull real subtasks from goal text before falling
back to keyword-matched templates when the text has no extractable structure.

read() accepts an optional vector_fn callback for semantic ranking (production
path via Qdrant); falls back to token-overlap sort when absent.

Per-tenant capability gating:
  CognitiveGatewayCapabilities controls which verbs are available to each org.
  Default: all five verbs enabled.  Can be restricted to read+write for lite plans.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.agents.anomaly_detection_agent import run_heuristic as _anomaly_detect
from app.agents.debate_ensemble_agent import DebateEnsembleAgent
from app.agents.llm.llm_breaker import create_llm_client
from app.agents.memory_tier_manager_agent import MemoryTierManagerAgent
from app.agents.query_intelligence_agent import run_heuristic as _query_intelligence_detect
from app.core.config import settings
from app.core.requester_context import RequesterContext
from app.services.cognitive_fingerprint_service import CognitiveFingerprintService
from app.services.context_compression_service import ContextCompressionService
from app.services.corrective_rag_service import CorrectiveRagService
from app.services.enterprise_fallbacks import (
    detect_blocking_subtask,
    detect_goal,
    extract_subtasks,
)
from app.services.attention_retrieval_service import AttentionRetrievalService
from app.services.self_rag_service import SelfRagService


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------

ALL_VERBS = frozenset({"write", "read", "decide", "plan", "explain", "answer"})


@dataclass
class CognitiveGatewayCapabilities:
    """Per-tenant capabilities — which verbs are permitted."""
    enabled_verbs: frozenset[str] = field(default_factory=lambda: frozenset(ALL_VERBS))

    def is_enabled(self, verb: str) -> bool:
        return verb in self.enabled_verbs

    @classmethod
    def lite(cls) -> "CognitiveGatewayCapabilities":
        """Read + write only — for basic integration plans."""
        return cls(enabled_verbs=frozenset({"read", "write"}))

    @classmethod
    def standard(cls) -> "CognitiveGatewayCapabilities":
        """All verbs except plan — for most enterprise plans."""
        return cls(enabled_verbs=frozenset({"read", "write", "decide", "explain", "answer"}))

    @classmethod
    def full(cls) -> "CognitiveGatewayCapabilities":
        """All five verbs enabled."""
        return cls(enabled_verbs=frozenset(ALL_VERBS))


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class GatewayWriteResult:
    memory_id: str
    enriched: bool
    enrichment_summary: dict
    tags: list[str]
    created_at: datetime


@dataclass
class GatewayReadResult:
    memories: list[dict]
    total: int
    query: str
    context_assembled: bool
    retrieval_confidence: float = 0.0
    corrected_by: str | None = None
    reasoning_steps: list[dict] = field(default_factory=list)
    compression_ratio: float = 1.0
    information_density: float = 0.0


@dataclass
class GatewayDecideResult:
    decision: str
    confidence: float
    tone: str
    action_recommended: str | None
    enrichment: dict
    agents_run: list[str]
    debate_transcript: list[dict] = field(default_factory=list)
    fingerprint_alerts: list[dict] = field(default_factory=list)


@dataclass
class GatewayPlanResult:
    goal: str
    steps: list[dict]
    step_count: int
    blocking_step: str | None
    confidence: float


@dataclass
class GatewayExplainResult:
    memory_id: str
    decisions: list[dict]
    agents: list[str]
    confidence: float
    explainability_summary: str


@dataclass
class GatewayAnswerResult:
    answer: str
    model: str
    context_turns: int
    used_llm: bool
    answer_source: str
    llm_error: str | None = None
    llm_failure_mode: str | None = None
    llm_endpoint: str | None = None


@dataclass
class GatewayJudgeResult:
    equivalent: bool | None  # None = judge failed
    raw: str
    model: str


# ---------------------------------------------------------------------------
# Internal pipeline helpers (pure — no I/O, designed for heuristic path)
# ---------------------------------------------------------------------------

def _enrich_write_heuristic(
    content: str,
    title: str,
    tags: list[str],
    metadata: dict,
) -> dict:
    """Simulate enrichment for the write verb in heuristic mode."""
    lower = content.lower()
    enrichment: dict[str, Any] = {}

    if any(w in lower for w in ("critical", "urgent", "outage", "down", "breach")):
        enrichment["tone"] = "urgent"
    elif any(w in lower for w in ("warning", "caution", "slow", "latency")):
        enrichment["tone"] = "cautionary"
    else:
        enrichment["tone"] = "informational"

    if any(w in lower for w in ("auth", "login", "sso", "password", "token")):
        enrichment["domain"] = "security"
    elif any(w in lower for w in ("deploy", "pipeline", "ci", "build")):
        enrichment["domain"] = "devops"
    elif any(w in lower for w in ("billing", "payment", "invoice")):
        enrichment["domain"] = "finance"
    else:
        enrichment["domain"] = "general"

    enrichment["word_count"] = len(content.split())
    enrichment["has_metadata"] = bool(metadata)
    return enrichment


def _credibility_weight(mem: dict) -> float:
    """Extract credibility weight from a memory dict (0.1–1.0).

    Falls back to 1.0 if no credibility signal is present so that memories
    without an explicit score are not penalised.
    """
    raw = mem.get("credibility_score")
    if raw is None:
        raw = mem.get("enrichment", {}).get("credibility_score")
    try:
        val = float(raw) if raw is not None else 1.0
    except (TypeError, ValueError):
        val = 1.0
    return max(0.1, min(1.0, val))


_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalise_tokens(text: str) -> set[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return {t for t in _PUNCT_RE.sub(" ", text.lower()).split() if t}


def _classify_llm_failure(error_text: str | None) -> str | None:
    value = str(error_text or "").strip().lower()
    if not value:
        return None
    if "empty_response" in value:
        return "empty_response"
    if "circuit_breaker_open" in value:
        return "circuit_breaker_open"
    if "404" in value and "not found" in value:
        return "model_not_available"
    if any(token in value for token in ("401", "403", "unauthorized", "forbidden")):
        return "auth_error"
    if any(token in value for token in ("429", "too many requests")):
        return "rate_limited"
    if any(token in value for token in ("502", "503", "504", "bad gateway", "gateway timeout")):
        return "upstream_unavailable"
    if any(token in value for token in ("timeout", "timed out", "readtimeout", "connecttimeout")):
        return "timeout"
    if "http" in value or "transport" in value or "connection" in value:
        return "transport_error"
    return "llm_error"


def _effective_llm_num_ctx(requested_num_ctx: int, *, task: str) -> int:
    requested = max(256, int(requested_num_ctx or 0) or 2048)
    if task == "judge":
        configured = int(getattr(settings, "VLLM_MAX_NUM_CTX_JUDGE", 1024) or 1024)
        default_cap = 1024
    else:
        configured = int(getattr(settings, "VLLM_MAX_NUM_CTX_ANSWER", 2048) or 2048)
        default_cap = 2048
    cap = max(512, min(configured, default_cap))
    return max(512, min(requested, cap))


def _effective_llm_max_concurrency(*, task: str) -> int:
    configured = max(1, int(getattr(settings, "VLLM_MAX_CONCURRENCY", 2) or 2))
    if task == "judge":
        return max(1, min(configured, 2))
    return max(1, min(configured, 2))


def _build_compact_answer_prompt(question: str, memories: list[dict]) -> tuple[str, list[str]]:
    fact_lines: list[str] = []
    seen_facts: set[tuple[str, str, str]] = set()
    for memory in memories[:24]:
        extra = dict(memory.get("extra_metadata") or {})
        support_candidates: list[dict[str, Any]] = []
        fact_support = extra.get("fact_support")
        if isinstance(fact_support, dict):
            support_candidates.append(fact_support)
        for item in list(extra.get("fact_supporting_facts") or []):
            if isinstance(item, dict):
                support_candidates.append(item)
        for item in support_candidates:
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not (subject and predicate and obj):
                continue
            key = (subject.lower(), predicate.lower(), obj.lower())
            if key in seen_facts:
                continue
            seen_facts.add(key)
            fact_lines.append(f"- FACT: {subject} | {predicate} | {obj}")
            if len(fact_lines) >= 4:
                break
        if len(fact_lines) >= 4:
            break

    context_lines: list[str] = []
    total_chars = 0
    max_chars = max(900, int(getattr(settings, "VLLM_MAX_PROMPT_CHARS_ANSWER", 1600) or 1600))
    reserved_chars = sum(len(line) + 1 for line in fact_lines)
    conversation_budget = max(700, max_chars - reserved_chars - 280)
    for memory in memories[:24]:
        line = _answer_context_line(memory)
        if not line:
            continue
        projected = total_chars + len(line) + 1
        if context_lines and projected > conversation_budget:
            break
        context_lines.append(line)
        total_chars = projected
    context_text = "\n".join(context_lines)
    fact_text = "\n".join(fact_lines)
    evidence_block = []
    if fact_text:
        evidence_block.extend(["Facts:", fact_text, ""])
    evidence_block.extend(["Conversation:", context_text])
    evidence_text = "\n".join(evidence_block)
    prompt = (
        "Answer the question using only the evidence below.\n"
        "Prefer FACT lines over conversation when they directly answer the question.\n"
        "Answer about the primary subject named in the question, not a distractor.\n"
        "Output ONLY the answer - a name, place, date, or short phrase (1-20 words).\n"
        "Use EXACT words from the evidence for named entities.\n"
        "Do not paraphrase or describe - copy the specific name/fact.\n\n"
        f"{evidence_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return prompt, context_lines


def _resolve_answer_prompt(question: str, memories: list[dict], prompt_override: str | None) -> tuple[str, list[str]]:
    if not prompt_override:
        return _build_compact_answer_prompt(question, memories)

    override = str(prompt_override or "")
    if not memories:
        return override, []

    override_lines = override.count("\n") + 1
    override_chars = len(override)
    oversized = override_chars > 3500 or override_lines > 60
    embedded_context = ("conversation:" in override.lower()) or ("context:" in override.lower())
    if oversized or embedded_context:
        logger.info(
            "gateway.answer compacting prompt_override question=%r chars=%s lines=%s",
            question[:120],
            override_chars,
            override_lines,
        )
        return _build_compact_answer_prompt(question, memories)

    return override, []


def _assemble_read_context(
    memories: list[dict],
    query: str,
    requester: "RequesterContext | None" = None,
) -> list[dict]:
    """Sort memories by credibility-weighted keyword-overlap relevance score.

    Score = (token_overlap + domain_boost) * credibility_weight

    domain_boost: memories whose content overlaps with the requester's
    dominant_domains get a 0.5-per-token additive boost.  This surfaces
    domain-relevant signals for e.g. a CFO without suppressing unrelated
    memories entirely.

    In production this is replaced by a Qdrant vector similarity query via
    the vector_fn callback accepted by CognitiveGatewayService.read().
    """
    query_tokens = _normalise_tokens(query)

    # Build domain token set from requester profile (empty if not available)
    domain_tokens: set[str] = set()
    if requester and requester.dominant_domains:
        for domain in requester.dominant_domains:
            domain_tokens.update(_normalise_tokens(domain.replace("_", " ")))

    scored = []
    for mem in memories:
        content_tokens = _normalise_tokens(str(mem.get("content") or ""))
        overlap = len(query_tokens & content_tokens)
        domain_boost = len(domain_tokens & content_tokens) * 0.5 if domain_tokens else 0.0
        weighted = (overlap + domain_boost) * _credibility_weight(mem)
        scored.append((weighted, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


def _coerce_rank_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:
        return None
    return score


_CGS_SPEAKER_RE = re.compile(r"^\[([A-Za-z][a-z]{1,})\]")
_CGS_DERIVED_PREFIXES = frozenset({
    "Fact", "Temporal", "About", "Relationship", "Summary", "Event", "Observation", "Note",
})
_CGS_SKIP_WORDS = {
    "what", "where", "who", "when", "which", "how", "why", "is", "are", "was", "were",
    "did", "do", "does", "has", "have", "had", "will", "would", "could", "should",
    "can", "may", "might", "not", "no", "the", "this", "that", "those", "these",
}


def _extract_question_subject(query: str) -> "str | None":
    """Return the primary proper-name subject of a question query, if any."""
    for token in re.findall(r"\b([A-Z][a-z]{2,})\b", query):
        if token.lower() not in _CGS_SKIP_WORDS:
            return token
    return None


def _extract_question_entities(query: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"\b([A-Z][a-z]{2,})\b", str(query or "")):
        lowered = token.lower()
        if lowered in _CGS_SKIP_WORDS or lowered in seen:
            continue
        seen.add(lowered)
        entities.append(token)
    return entities


def _memory_mentions_entity(memory: dict[str, Any], entity: str) -> bool:
    target = str(entity or "").strip().lower()
    if not target:
        return False

    text_parts = [
        str(memory.get("title") or ""),
        str(memory.get("content_preview") or ""),
        str(memory.get("content") or ""),
    ]
    entities = memory.get("entities") or {}
    if isinstance(entities, dict):
        for value in entities.values():
            if isinstance(value, list):
                text_parts.extend(str(item or "") for item in value)
            else:
                text_parts.append(str(value or ""))

    extra = dict(memory.get("extra_metadata") or {})
    fact_support = dict(extra.get("fact_support") or {})
    text_parts.extend(str(fact_support.get(key) or "") for key in ("subject", "predicate", "object"))
    for fact in extra.get("fact_supporting_facts") or []:
        if isinstance(fact, dict):
            text_parts.extend(str(fact.get(key) or "") for key in ("subject", "predicate", "object"))

    combined = " ".join(part.strip().lower() for part in text_parts if str(part).strip())
    if not combined:
        return False
    return target in combined


def _memory_is_about_entity(memory: dict[str, Any], entity: str) -> bool:
    target = str(entity or "").strip()
    if not target:
        return False

    preview = " ".join(
        str(memory.get(key) or "").strip()
        for key in ("title", "content_preview", "content")
    )
    if not preview:
        return False

    lowered = preview.lower()
    subject = target.lower()
    if not _memory_mentions_entity(memory, target):
        return False

    patterns = (
        rf"\b{re.escape(subject)}['’]s\b",
        rf"\b{re.escape(subject)}\s+(?:is|was|will|would|did|does|has|have|had|realized|planned|plans|wants|wanted|needs|needed|likes|liked|loves|loved|researched|researching|moved|move|went|goes|attended|supports|chose|chooses)\b",
        rf"\babout\s+{re.escape(subject)}\b",
        rf"\bfor\s+{re.escape(subject)}\b",
    )
    if any(re.search(pattern, lowered) for pattern in patterns):
        return True

    extra = dict(memory.get("extra_metadata") or {})
    fact_support = dict(extra.get("fact_support") or {})
    if str(fact_support.get("subject") or "").strip().lower() == subject:
        return True
    for fact in extra.get("fact_supporting_facts") or []:
        if isinstance(fact, dict) and str(fact.get("subject") or "").strip().lower() == subject:
            return True
    return False


def _subject_relevance(memory: dict[str, Any], subject: str | None, distractors: list[str]) -> float:
    if not subject:
        return 0.0

    score = 0.0
    preview = str(memory.get("content_preview") or memory.get("content") or "").strip()
    speaker_match = _CGS_SPEAKER_RE.match(preview)
    if speaker_match and speaker_match.group(1).lower() == subject.lower():
        score += 1.1
    if _memory_mentions_entity(memory, subject):
        score += 1.4

    lower_distractors = [item.lower() for item in distractors if item and item.lower() != subject.lower()]
    if lower_distractors and not _memory_mentions_entity(memory, subject):
        distractor_hits = sum(1 for item in lower_distractors if _memory_mentions_entity(memory, item))
        score -= min(0.75, distractor_hits * 0.3)
    return score


def _detect_speaker_mismatch(query: str, memories: list[dict], top_n: int = 3) -> bool:
    """Return True when all top retrieved memories are from a speaker other than the query subject.

    Used to detect unanswerable questions where the evidence exists but belongs to the
    wrong conversation participant.  When True, the verified context should be cleared so
    the downstream LLM answers 'Not mentioned' rather than hallucinating from the wrong
    speaker's content.
    """
    subject = _extract_question_subject(query)
    if not subject:
        return False

    candidates = memories[:top_n]
    if not candidates:
        return False

    # If the evidence is clearly about the requested subject, do not discard it
    # just because someone else is the speaker for that turn.
    if any(_memory_is_about_entity(memory, subject) for memory in candidates):
        return False

    mismatch_count = 0
    for m in candidates:
        preview = str(m.get("content_preview") or "")
        sm = _CGS_SPEAKER_RE.match(preview)
        if not sm:
            continue
        speaker = sm.group(1)
        if speaker in _CGS_DERIVED_PREFIXES:
            continue
        if speaker.lower() != subject.lower():
            mismatch_count += 1

    speaker_turn_count = len([
        m for m in candidates
        if _CGS_SPEAKER_RE.match(str(m.get("content_preview") or ""))
        and _CGS_SPEAKER_RE.match(str(m.get("content_preview") or "")).group(1) not in _CGS_DERIVED_PREFIXES
    ])
    return speaker_turn_count > 0 and mismatch_count == speaker_turn_count


def _prepare_read_candidates(
    memories: list[dict],
    query: str,
    requester: "RequesterContext | None",
    limit: int,
) -> list[dict]:
    """Preserve upstream search ranking when candidates already carry scores."""
    if not memories:
        return []

    window = max(int(limit or 0), min(len(memories), max(10, int(limit or 0) * 3)))
    query_entities = _extract_question_entities(query)
    query_subject = query_entities[0] if query_entities else None
    distractors = query_entities[1:]
    scored_candidates: list[tuple[int, float, float, dict[str, Any]]] = []
    for idx, memory in enumerate(memories):
        existing = _coerce_rank_score(memory.get("_retrieval_score", memory.get("score")))
        if existing is None:
            continue
        enriched = dict(memory)
        enriched["_retrieval_score"] = existing
        scored_candidates.append((idx, existing, _subject_relevance(enriched, query_subject, distractors), enriched))

    if scored_candidates:
        scored_candidates.sort(key=lambda item: (-(item[1] + item[2]), -item[1], item[0]))
        return [item[3] for item in scored_candidates[:window]]

    return _assemble_read_context(memories, query, requester)[:window]


def _memory_field_tokens(memory: dict[str, Any]) -> set[str]:
    """Collect normalized tokens from content and key enrichment fields."""
    tokens: set[str] = set()

    content = str(memory.get("content") or "")
    tokens.update(_normalise_tokens(content))

    enrichment = memory.get("enrichment") or {}

    for field in ("entity_name", "entity_type"):
        raw = memory.get(field)
        if raw is None and isinstance(enrichment, dict):
            raw = enrichment.get(field)
        if isinstance(raw, str) and raw.strip():
            tokens.update(_normalise_tokens(raw))

    for field in ("resolved_aliases", "key_entities"):
        raw = memory.get(field)
        if raw is None and isinstance(enrichment, dict):
            raw = enrichment.get(field)
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    tokens.update(_normalise_tokens(item))

    tags = memory.get("tags")
    if tags is None and isinstance(enrichment, dict):
        tags = enrichment.get("tags") or enrichment.get("normalized_tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                tokens.update(_normalise_tokens(tag))

    return tokens


def _apply_query_filters(memories: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply QueryIntelligence dynamic filters with conservative fail-open behavior."""
    if not filters:
        return memories

    filtered = memories

    entity_type = filters.get("entity_type")
    if isinstance(entity_type, str) and entity_type.strip():
        et = entity_type.strip().lower()
        next_filtered = []
        for mem in filtered:
            enrichment = mem.get("enrichment") or {}
            mem_type = mem.get("entity_type")
            if mem_type is None and isinstance(enrichment, dict):
                mem_type = enrichment.get("entity_type")
            if isinstance(mem_type, str) and mem_type.strip().lower() == et:
                next_filtered.append(mem)
        if next_filtered:
            filtered = next_filtered

    tags = filters.get("tags")
    if isinstance(tags, list) and tags:
        requested = {str(t).strip().lower() for t in tags if str(t).strip()}
        if requested:
            next_filtered = []
            for mem in filtered:
                enrichment = mem.get("enrichment") or {}
                mem_tags = mem.get("tags")
                if mem_tags is None and isinstance(enrichment, dict):
                    mem_tags = enrichment.get("tags") or enrichment.get("normalized_tags")
                norm_tags = {str(t).strip().lower() for t in (mem_tags or []) if str(t).strip()}
                if requested.issubset(norm_tags):
                    next_filtered.append(mem)
            if next_filtered:
                filtered = next_filtered

    if filters.get("has_temporal_data") is True:
        next_filtered = []
        for mem in filtered:
            enrichment = mem.get("enrichment") or {}
            has_temporal = bool(mem.get("timestamp") or mem.get("created_at"))
            if isinstance(enrichment, dict):
                has_temporal = has_temporal or bool(
                    enrichment.get("temporal_confidence")
                    or enrichment.get("timeline")
                    or enrichment.get("event_time")
                )
            if has_temporal:
                next_filtered.append(mem)
        if next_filtered:
            filtered = next_filtered

    min_credibility = filters.get("min_credibility")
    if min_credibility is not None:
        try:
            min_score = float(min_credibility)
        except (TypeError, ValueError):
            min_score = None
        if min_score is not None:
            next_filtered = [m for m in filtered if _credibility_weight(m) >= min_score]
            if next_filtered:
                filtered = next_filtered

    exclude_levels = filters.get("exclude_uncertainty_levels")
    if isinstance(exclude_levels, list) and exclude_levels:
        blocked = {str(l).strip().lower() for l in exclude_levels if str(l).strip()}
        if blocked:
            next_filtered = []
            for mem in filtered:
                enrichment = mem.get("enrichment") or {}
                level = mem.get("uncertainty_level")
                if level is None and isinstance(enrichment, dict):
                    level = enrichment.get("uncertainty_level")
                if str(level or "").strip().lower() not in blocked:
                    next_filtered.append(mem)
            if next_filtered:
                filtered = next_filtered

    return filtered


def _prioritize_entity_matches(
    memories: list[dict[str, Any]],
    entities: list[str],
) -> tuple[list[dict[str, Any]], int]:
    """Move entity-matched memories to the front while preserving relative order."""
    if not memories or not entities:
        return memories, 0

    entity_tokens: set[str] = set()
    for ent in entities:
        if isinstance(ent, str) and ent.strip():
            entity_tokens.update(_normalise_tokens(ent))

    if not entity_tokens:
        return memories, 0

    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for mem in memories:
        tokens = _memory_field_tokens(mem)
        if tokens & entity_tokens:
            matched.append(mem)
        else:
            unmatched.append(mem)

    return matched + unmatched, len(matched)


def _escalate_action(requester: "RequesterContext | None") -> str:
    """Return a role-appropriate escalation action string."""
    if requester and requester.job_role:
        role = requester.job_role.lower()
        if any(r in role for r in ("ceo", "cto", "cfo", "vp", "director", "exec")):
            return "Brief executive team and prepare board communication."
        if any(r in role for r in ("engineer", "sre", "devops", "oncall", "on-call")):
            return "Declare P1, page on-call team, and open incident channel."
        if any(r in role for r in ("csm", "customer", "success", "support")):
            return "Notify affected enterprise accounts and escalate to account managers."
    return "Declare P1 and page on-call engineer."


def _heuristic_decide(
    content: str,
    enrichment: dict,
    requester: "RequesterContext | None" = None,
) -> GatewayDecideResult:
    """Decision from content and enrichment signals.

    Delegates to AnomalyDetectionAgent.run_heuristic() for real anomaly
    signal extraction.  Caller-supplied anomaly_detected / anomaly_score
    values take precedence and override agent outputs (allows pre-computed
    pipeline enrichment to flow through unchanged).
    """
    agents_run = ["anomaly_detection", "entity_resolution", "narrative_synthesis"]

    # Run real anomaly detection; caller overrides take precedence
    agent_signals = _anomaly_detect(enrichment)
    merged: dict[str, Any] = {**agent_signals, **enrichment}

    anomaly = bool(merged.get("anomaly_detected"))
    score = float(merged.get("anomaly_score") or 0.0)
    tone = str(merged.get("tone") or "informational")

    lower = content.lower()

    # Crisis urgency (requester paging outside business hours) → lower the
    # escalation threshold so marginal signals still get routed correctly.
    crisis_offset = 0.1 if (requester and requester.urgency_signal == "crisis") else 0.0

    if anomaly and score >= (0.9 - crisis_offset):
        decision = "escalate"
        confidence = min(0.95, 0.70 + score * 0.25)
        action: str | None = _escalate_action(requester)
    elif anomaly and score >= (0.7 - crisis_offset):
        decision = "investigate"
        confidence = 0.70 + score * 0.10
        action = "Assign to on-call team for investigation."
    elif any(w in lower for w in ("critical", "urgent", "outage")):
        decision = "escalate"
        confidence = 0.75
        action = _escalate_action(requester)
    elif any(w in lower for w in ("warning", "caution")):
        decision = "monitor"
        confidence = 0.65
        action = "Monitor and review in next stand-up."
    else:
        decision = "acknowledge"
        confidence = 0.55
        action = None

    debate_transcript: list[dict] = []
    if confidence >= 0.7:
        debate = DebateEnsembleAgent().generate_transcript(
            content=content,
            decision=decision,
            confidence=confidence,
            enrichment=merged,
        )
        debate_transcript = list(debate.get("debate_transcript") or [])
        moderator_decision = str(debate.get("moderator_decision") or decision)
        if moderator_decision:
            decision_rank = {
                "acknowledge": 0,
                "monitor": 1,
                "investigate": 2,
                "escalate": 3,
            }
            base_rank = decision_rank.get(decision, 1)
            mod_rank = decision_rank.get(moderator_decision, base_rank)
            # Moderator synthesis can strengthen or keep severity, but should
            # not downgrade a stronger safety posture.
            if mod_rank >= base_rank:
                decision = moderator_decision
        agents_run.append("debate_ensemble")

    return GatewayDecideResult(
        decision=decision,
        confidence=round(confidence, 4),
        tone=tone,
        action_recommended=action,
        enrichment=merged,
        agents_run=agents_run,
        debate_transcript=debate_transcript,
    )


_TECHNICAL_TOOLS = {"causal.explain", "entity_resolution", "causal.predict", "goal_decomposition"}
_NARRATIVE_TOOLS = {"narrative.synthesize", "narrative.generate"}


def _filter_steps_for_requester(
    steps: list[dict],
    requester: "RequesterContext",
) -> list[dict]:
    """Trim and reorder plan steps based on who is reading the plan.

    Rules:
    - pre_meeting urgency → cap at 3 steps (decision-maker needs concise list)
    - crisis urgency → push action steps first (act now, explain later)
    - executive roles (CEO/CFO/VP/Director) → drop purely technical tool steps
    - engineering/sre roles → drop narrative-synthesis-only steps
    """
    role = (requester.job_role or "").lower()
    urgency = requester.urgency_signal

    is_executive = any(r in role for r in ("ceo", "cfo", "cto", "vp", "director", "exec", "chief"))
    is_engineer = any(r in role for r in ("engineer", "sre", "devops", "oncall", "developer"))

    filtered = list(steps)

    if is_executive:
        filtered = [
            s for s in filtered
            if s.get("tool") not in _TECHNICAL_TOOLS
        ]

    if is_engineer:
        filtered = [
            s for s in filtered
            if s.get("tool") not in _NARRATIVE_TOOLS
        ]

    if urgency == "crisis":
        # Bring action/dispatch steps to the front
        action_steps = [s for s in filtered if s.get("tool") and "action" in s.get("tool", "")]
        other_steps = [s for s in filtered if s not in action_steps]
        filtered = action_steps + other_steps

    if urgency == "pre_meeting":
        filtered = filtered[:3]

    # Always keep at least one step
    return filtered or steps[:1]


def _heuristic_plan(
    goal: str,
    context: dict,
    requester: "RequesterContext | None" = None,
) -> GatewayPlanResult:
    """Goal decomposition using real subtask extraction.

    First attempts structural extraction via GoalDecompositionAgent helpers
    (numbered lists, bullet points, action-verb sentences).  Falls back to
    keyword-matched templates when extraction yields no subtasks — which is
    the common case for short, unstructured goal strings.
    """
    step_num = 0

    def _step(action: str, tool: str | None = None) -> dict:
        nonlocal step_num
        step_num += 1
        return {"step_id": f"s{step_num}", "action": action, "tool": tool}

    # --- Try real extraction first (require ≥ 2 subtasks; single-sentence
    #     extraction is just the goal itself and doesn't add value over templates) ---
    raw_subtasks = extract_subtasks(goal)
    if len(raw_subtasks) >= 2:
        steps = [_step(t) for t in raw_subtasks]
        blocking = (
            detect_blocking_subtask(raw_subtasks, goal)
            or context.get("blocking_subtask")
            or None
        )
        confidence = 0.85 if len(steps) >= 3 else 0.70
        if requester:
            steps = _filter_steps_for_requester(steps, requester)
        return GatewayPlanResult(
            goal=goal,
            steps=steps,
            step_count=len(steps),
            blocking_step=blocking,
            confidence=confidence,
        )

    # --- Keyword-template fallback (unstructured goal strings) ---
    lower = goal.lower()

    if "report" in lower or "summarise" in lower or "summarize" in lower:
        steps = [
            _step("Retrieve relevant memories", "memory.search"),
            _step("Assemble temporal context", "temporal_reasoning"),
            _step("Generate narrative summary", "narrative.synthesize"),
        ]
    elif "investigate" in lower or "debug" in lower or "diagnose" in lower:
        steps = [
            _step("Search memory for related incidents", "memory.search"),
            _step("Identify affected entities", "entity_resolution"),
            _step("Check causal graph for known patterns", "causal.explain"),
            _step("Synthesise investigation report", "narrative.synthesize"),
        ]
    elif "escalate" in lower or "p1" in lower or "incident" in lower:
        steps = [
            _step("Retrieve P1 playbook", "playbook.match"),
            _step("Identify escalation targets", "org_attention"),
            _step("Dispatch P1 notification", "action.dispatch"),
            _step("Open incident episode", "episode.create"),
        ]
    else:
        steps = [
            _step("Retrieve context for goal", "memory.search"),
            _step("Decompose goal into subtasks", "goal_decomposition"),
            _step("Execute subtasks", None),
            _step("Validate completion", None),
        ]

    blocking = context.get("blocking_subtask") or None
    confidence = 0.80 if len(steps) >= 3 else 0.65

    # Apply requester-aware step filtering.
    if requester:
        steps = _filter_steps_for_requester(steps, requester)

    return GatewayPlanResult(
        goal=goal,
        steps=[s for s in steps if s],
        step_count=len(steps),
        blocking_step=blocking,
        confidence=confidence,
    )


def _heuristic_explain(memory_id: str, audit_records: list[dict]) -> GatewayExplainResult:
    """Assemble explainability summary from audit records."""
    agents = list({r.get("agent_name") or "" for r in audit_records if r.get("agent_name")})
    agents = [a for a in agents if a]

    if not audit_records:
        return GatewayExplainResult(
            memory_id=memory_id,
            decisions=[],
            agents=[],
            confidence=0.0,
            explainability_summary="No audit records found for this memory.",
        )

    recent = audit_records[-5:]
    confidence_vals = [
        float(r["confidence"]) for r in recent
        if "confidence" in r and _is_float(r["confidence"])
    ]
    avg_conf = sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.5

    summary_parts = [f"Memory {memory_id} was processed by {len(agents)} agent(s)."]
    if agents:
        summary_parts.append(f"Agents: {', '.join(agents[:5])}.")
    summary_parts.append(f"Average confidence: {avg_conf:.2f}.")

    return GatewayExplainResult(
        memory_id=memory_id,
        decisions=recent,
        agents=agents,
        confidence=round(avg_conf, 4),
        explainability_summary=" ".join(summary_parts),
    )


def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _memory_entity_hints(memory: dict[str, Any], limit: int = 6) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()

    def _push(value: Any) -> None:
        if len(hints) >= limit:
            return
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or len(cleaned) > 80:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            hints.append(cleaned)
            return
        if isinstance(value, list):
            for item in value:
                _push(item)
                if len(hints) >= limit:
                    return
            return
        if isinstance(value, dict):
            for item in value.values():
                _push(item)
                if len(hints) >= limit:
                    return

    _push(memory.get("entities") or {})
    extra = memory.get("extra_metadata") or {}
    if isinstance(extra, dict):
        _push(extra.get("resolved_entities") or [])
        _push(extra.get("key_entities") or [])

    return hints


def _memory_relation_hints(memory: dict[str, Any], limit: int = 4) -> list[str]:
    rels: list[str] = []
    seen: set[str] = set()

    def _add(kind: Any, value: Any) -> None:
        if len(rels) >= limit:
            return
        kind_s = str(kind or "related_to").strip().lower()[:32]
        value_s = str(value or "").strip()
        if not value_s or len(value_s) > 80:
            return
        text = f"{kind_s}:{value_s}"
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        rels.append(text)

    extra = memory.get("extra_metadata") or {}
    if isinstance(extra, dict):
        raw = extra.get("semantic_relationships") or []
        if isinstance(raw, list):
            for rel in raw:
                if not isinstance(rel, dict):
                    continue
                _add(rel.get("type"), rel.get("concept") or rel.get("target") or rel.get("source"))
                if len(rels) >= limit:
                    break

    return rels


def _answer_context_line(memory: dict[str, Any]) -> str:
    content = str(memory.get("content") or "").strip()
    if not content:
        return ""

    entities = _memory_entity_hints(memory)
    relations = _memory_relation_hints(memory)
    if not entities and not relations:
        return content

    lines = [content]
    if entities:
        lines.append("Entities: " + ", ".join(entities))
    if relations:
        lines.append("Relations: " + "; ".join(relations))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gateway context chain (stateful multi-call sessions via Redis)
# ---------------------------------------------------------------------------

@dataclass
class GatewayContextSession:
    """Accumulated state across multiple gateway verb calls sharing a context_id."""

    context_id: str
    org_id: str
    prior_decision: str | None = None
    prior_steps: list = field(default_factory=list)
    prior_memories: list = field(default_factory=list)
    accumulated_enrichment: dict = field(default_factory=dict)
    working_set_summary: dict = field(default_factory=lambda: {
        "working_set": [],
        "archival": [],
        "working_set_size": 0,
        "archival_size": 0,
        "loaded_ids": [],
        "offloaded_ids": [],
        "updated_at": None,
    })
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    call_count: int = 0


def _ctx_key(context_id: str, org_id: str) -> str:
    return f"gateway:ctx:{context_id}:{org_id}"


async def load_gateway_context(context_id: str, org_id: str) -> GatewayContextSession | None:
    """Load a gateway context session from Redis. Returns None if missing or expired."""
    try:
        from app.core.redis import get_redis_client  # type: ignore[import]
        r = get_redis_client()
        raw = r.get(_ctx_key(context_id, org_id))
        if raw is None:
            return None
        data = json.loads(raw)
        return GatewayContextSession(**data)
    except Exception:
        return None


async def save_gateway_context(ctx: GatewayContextSession, ttl: int = 3600) -> None:
    """Persist updated context to Redis with TTL (default 1 hour)."""
    try:
        from app.core.redis import get_redis_client  # type: ignore[import]
        r = get_redis_client()
        ctx.updated_at = datetime.now(timezone.utc).isoformat()
        r.set(_ctx_key(ctx.context_id, ctx.org_id), json.dumps(dataclasses.asdict(ctx)), ex=ttl)
    except Exception:
        pass  # Context chaining degrades gracefully when Redis is unavailable


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------

class CognitiveGatewayService:
    """Five-verb gateway to Ninai's full intelligence stack.

    Each verb is:
    1. Capability-gated per org/tenant
    2. Backed by real agent logic (anomaly detection, goal decomposition)
       with token-overlap / template fallbacks for offline/test paths
    3. Returns a typed result object

    read() accepts an optional vector_fn(query, memories) → list[dict]
    callback for Qdrant-backed semantic ranking in production.
    """

    def __init__(
        self,
        *,
        capabilities: CognitiveGatewayCapabilities | None = None,
    ) -> None:
        self._caps = capabilities or CognitiveGatewayCapabilities.full()
        self._tier_manager = MemoryTierManagerAgent()
        self._fingerprint = CognitiveFingerprintService()

    def _check(self, verb: str) -> None:
        if not self._caps.is_enabled(verb):
            raise PermissionError(f"Verb '{verb}' not enabled for this tenant.")
    
    def _extract_single_hop_fact(self, query: str, memories: list[dict]) -> dict[str, Any] | None:
        """Extract shortest factual span from verified context for attribute queries."""
        if not memories:
            return None
        
        # Look for direct fact markers in top memories
        query_lower = query.lower()
        for memory in memories[:3]:
            content = str(memory.get("content") or "").lower()
            title = str(memory.get("title") or "").lower()
            
            # Try to find direct statement containing question entity/attribute
            for text_source in [title, content]:
                if not text_source:
                    continue
                # Extract entity-attribute pairs like "[Caroline] Transgender woman" or "Caroline: Single"
                bracket_match = re.search(r'\[([^\]]+)\]\s*[:-]?\s*([^.\n]+)', text_source)
                if bracket_match:
                    entity, fact = bracket_match.groups()
                    fact = fact.strip()
                    if len(fact.split()) <= 5:  # Short fact span
                        return {
                            "fact": fact,
                            "memory_id": memory.get("id"),
                            "confidence": 0.95,
                        }
                # Alternative: look for "is ", "was ", etc.
                for marker in [" is ", " was ", " moved from ", " lives in "]:
                    if marker in text_source:
                        parts = text_source.split(marker)
                        if len(parts) >= 2:
                            fact = parts[-1].strip().split(".")[0].split(",")[0][:100]
                            if fact and len(fact.split()) <= 8:
                                return {
                                    "fact": fact,
                                    "memory_id": memory.get("id"),
                                    "confidence": 0.85,
                                }
        
        return None

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    async def write(
        self,
        *,
        content: str,
        title: str = "",
        tags: list[str] | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayWriteResult:
        """Store and enrich a memory record."""
        self._check("write")
        _tags = list(tags or [])
        _meta = metadata or {}
        enrichment = _enrich_write_heuristic(content, title, _tags, _meta)

        import uuid
        mid = memory_id or str(uuid.uuid4())

        # Merge requester context into enrichment so downstream agents
        # know the provenance of this write (who wrote it, their role, when).
        if requester:
            _meta.setdefault("_requester_job_role", requester.job_role)
            _meta.setdefault("_requester_timezone", requester.timezone)
            _meta.setdefault("_requester_urgency", requester.urgency_signal)
            # Seed domain from requester's top domain if enrichment didn't resolve one
            if enrichment.get("domain") == "general" and requester.dominant_domains:
                enrichment["domain"] = requester.dominant_domains[0]

        result = GatewayWriteResult(
            memory_id=mid,
            enriched=True,
            enrichment_summary=enrichment,
            tags=_tags,
            created_at=datetime.now(timezone.utc),
        )

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.accumulated_enrichment.update(enrichment)
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[{"id": mid, "content": content}],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    async def read(
        self,
        *,
        query: str,
        memories: list[dict] | None = None,
        limit: int = 10,
        context_token_budget: int = 4000,
        filter_tags: list[str] | None = None,
        vector_fn: Callable[[str, list[dict]], list[dict]] | None = None,
        external_connector_fn: Callable[[str, int], list[dict]] | None = None,
        cross_encoder_fn: Callable[[str, list[dict]], list[dict]] | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayReadResult:
        """Retrieve and rank context for a query.

        vector_fn(query, memories) → sorted list[dict]
          When provided, used instead of token-overlap sort.
          Production: pass a closure over the org's Qdrant collection.
          Tests / offline: omit to use the token-overlap fallback.
        """
        self._check("read")

        # Merge prior memories from context chain as additional candidates
        _mems = list(memories or [])
        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id)
            if ctx and ctx.prior_memories:
                _mems = ctx.prior_memories + _mems

        # Optional tag-based scoping: keep only memories that carry ALL filter_tags.
        # Useful for conversation-scoped retrieval without passing the full candidate list.
        if filter_tags:
            _tag_set = set(filter_tags)
            _mems = [m for m in _mems if _tag_set.issubset(set(m.get("tags") or []))]

        # Query intelligence (Phase 27): derive intent + entities + dynamic filters,
        # then apply conservative filtering and entity-priority ordering before ranking.
        qi = _query_intelligence_detect(query, {})
        qi_entities = [e for e in (qi.get("extracted_entities") or []) if isinstance(e, str) and e.strip()]
        qi_filters = qi.get("dynamic_filters") if isinstance(qi.get("dynamic_filters"), dict) else {}

        pre_qi_count = len(_mems)
        _mems = _apply_query_filters(_mems, qi_filters)
        _mems, entity_match_count = _prioritize_entity_matches(_mems, qi_entities)

        ranking_query = query
        if qi_entities:
            ranking_query = f"{query} {' '.join(qi_entities[:6])}".strip()

        if vector_fn is not None:
            # Caller-supplied semantic ranking takes precedence; skip attention rerank.
            ranked = vector_fn(ranking_query, _mems)[:limit]
        else:
            ranked = _prepare_read_candidates(_mems, ranking_query, requester, limit)
            # Attention-weighted reranking (Phase 58): re-scores candidates using
            # recency, query-token relevance, and goal/incident alignment before
            # passing to SelfRag. Works on plain dicts — no DB access required.
            _query_tokens = frozenset(
                re.findall(r"\b[a-z0-9_]+\b", ranking_query.lower())
            )
            ranked = AttentionRetrievalService().rank(
                memories=ranked,
                active_goals=[],
                active_incidents=[],
                query_tokens=_query_tokens,
                now=datetime.now(timezone.utc),
                limit=limit,
            )

        # Self-RAG verification before answer-time consumption.
        # strict_support requires keyword overlap between query and memory — valid for
        # open-corpus vector search but over-filters pre-scoped memory pools where the
        # caller has already narrowed candidates.  Always use non-strict mode so that
        # memories with sufficient content and credibility are not discarded purely
        # because the query phrasing differs from the stored text.
        self_rag = SelfRagService()
        verification = self_rag.verify_and_filter(
            query=query,
            memories=ranked,
            strict_support=False,
        )
        verified = verification.verified_memories

        # Speaker mismatch guard: when all top retrieved memories are from a different
        # speaker than the question subject, the question is likely unanswerable.
        # Clearing verified lets the downstream LLM respond with "Not mentioned".
        if _detect_speaker_mismatch(query, verified[:3]):
            verified = []

        # Single-hop fact extraction: if query appears to be attribute lookup (who/what/where/status)
        # and verified context is sparse, extract shortest factual span instead of full answer generation.
        qi_intent = qi.get("query_intent", "").lower() if isinstance(qi.get("query_intent"), str) else ""
        is_single_hop_style = qi_intent in {"retrieve", "find_attribute", "find_person", "find_location"} and len(ranked) <= 5
        single_hop_extraction = None
        if is_single_hop_style and verified:
            single_hop_extraction = self._extract_single_hop_fact(query, verified)
            if single_hop_extraction:
                # Bypass compression and return early with extracted fact
                result = GatewayReadResult(
                    memories=verified[:limit],
                    total=len(verified),
                    query=query,
                    context_assembled=True,
                    retrieval_confidence=0.9 if single_hop_extraction.get("confidence") else 0.6,
                    corrected_by="single_hop_extraction",
                    reasoning_steps=[
                        {
                            "step": "query_intelligence",
                            "intent": qi_intent,
                            "confidence": qi.get("confidence"),
                            "entities": qi_entities,
                            "applied_filters": qi_filters,
                            "candidate_count_before": pre_qi_count,
                            "candidate_count_after": len(_mems),
                            "entity_match_count": entity_match_count,
                        },
                        {
                            "step": "single_hop_extraction",
                            "extracted_fact": single_hop_extraction.get("fact"),
                            "source_memory_id": single_hop_extraction.get("memory_id"),
                        },
                        *verification.reasoning_steps,
                    ],
                    compression_ratio=1.0,
                    information_density=1.0,
                )
                if context_id and org_id:
                    ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                        context_id=context_id, org_id=org_id
                    )
                    ctx.prior_memories = verified[:limit]
                    ctx.call_count += 1
                    await save_gateway_context(ctx)
                return result

        # CRAG corrective pass if verified context quality is low.
        corrective = CorrectiveRagService().apply(
            query=query,
            memories=verified,
            limit=limit,
            external_connector_fn=external_connector_fn,
            cross_encoder_fn=cross_encoder_fn,
        )

        compression = ContextCompressionService().compress(
            memories=corrective.memories,
            token_budget=context_token_budget,
        )

        result = GatewayReadResult(
            memories=compression.memories,
            total=len(compression.memories),
            query=query,
            context_assembled=len(compression.memories) > 0,
            retrieval_confidence=corrective.retrieval_confidence,
            corrected_by=corrective.corrected_by,
            reasoning_steps=[
                {
                    "step": "query_intelligence",
                    "intent": qi.get("query_intent"),
                    "confidence": qi.get("confidence"),
                    "entities": qi_entities,
                    "applied_filters": qi_filters,
                    "candidate_count_before": pre_qi_count,
                    "candidate_count_after": len(_mems),
                    "entity_match_count": entity_match_count,
                },
                *verification.reasoning_steps,
            ],
            compression_ratio=compression.compression_ratio,
            information_density=compression.information_density,
        )

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.prior_memories = compression.memories
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=list(compression.memories or []),
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # decide
    # ------------------------------------------------------------------

    async def decide(
        self,
        *,
        content: str,
        enrichment: dict | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayDecideResult:
        """Run anomaly detection pipeline and return a decision with confidence."""
        self._check("decide")

        _enrichment = dict(enrichment or {})
        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id)
            if ctx:
                # Merge accumulated enrichment from prior calls
                merged = {**ctx.accumulated_enrichment, **_enrichment}
                _enrichment = merged

        result = _heuristic_decide(content, _enrichment, requester)

        # Fingerprint the anomaly detection agent's output distribution.
        fp_result = self._fingerprint.detect_anomaly("anomaly_detection", result.enrichment)
        self._fingerprint.update_fingerprint("anomaly_detection", result.enrichment)
        result.fingerprint_alerts = [
            {
                "agent": a.agent_name,
                "field": a.field,
                "current_value": a.current_value,
                "expected_mean": a.expected_mean,
                "z_score": a.z_score,
            }
            for a in fp_result.alerts
        ]
        if fp_result.anomalous and "cognitive_fingerprint" not in result.agents_run:
            result.agents_run.append("cognitive_fingerprint")

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.prior_decision = result.decision
            ctx.accumulated_enrichment.update(result.enrichment)
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # plan
    # ------------------------------------------------------------------

    async def plan(
        self,
        *,
        goal: str,
        context: dict | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
        requester: RequesterContext | None = None,
    ) -> GatewayPlanResult:
        """Decompose a goal into ordered, actionable steps."""
        self._check("plan")

        _ctx = dict(context or {})
        if context_id and org_id:
            prior = await load_gateway_context(context_id, org_id)
            if prior and prior.prior_decision:
                _ctx.setdefault("prior_decision", prior.prior_decision)

        result = _heuristic_plan(goal, _ctx, requester)

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.prior_steps = result.steps
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result

    # ------------------------------------------------------------------
    # answer
    # ------------------------------------------------------------------

    async def answer(
        self,
        *,
        question: str,
        memories: list[dict],
        model: str | None = None,
        num_ctx: int = 32768,
        prompt_override: str | None = None,
        timeout_seconds: float | None = None,
        keep_alive: int | None = None,
    ) -> GatewayAnswerResult:
        """Generate an answer to a question using retrieved memory context.

        Calls the server-side vLLM instance so LLM inference stays in-cluster.
        Does not fall back to heuristic extraction when vLLM is unavailable.

        If prompt_override is provided it is sent to vLLM directly, bypassing
        the default context-assembly and prompt template.

        timeout_seconds: override the default VLLM_TIMEOUT_SECONDS config.
          Benchmark callers pass 200s+ to avoid 30s hard-timeout failures on
          large models (qwen2.5:32b on CPU takes 60-120s including model load).
        keep_alive: seconds to keep the model loaded (-1 = forever). Pass -1
          during long benchmark runs to prevent model eviction between calls.
        """
        self._check("answer")

        effective_num_ctx = _effective_llm_num_ctx(num_ctx, task="answer")
        resolved_prompt: str | None = None
        resolved_context_lines: list[str] | None = None
        if prompt_override:
            resolved_prompt, resolved_context_lines = _resolve_answer_prompt(question, memories, prompt_override)
            if resolved_prompt != str(prompt_override or "") or resolved_context_lines:
                prompt_override = None

        if prompt_override is None and resolved_prompt is not None:
            prompt = resolved_prompt
            context_lines = list(resolved_context_lines or [])
        elif prompt_override:
            prompt = prompt_override
            context_lines: list[str] = []
        else:
            context_lines = []
            for memory in memories[:80]:
                line = _answer_context_line(memory)
                if line:
                    context_lines.append(line)
            context_text = "\n".join(context_lines)
            prompt = (
                "Answer the question using only the conversation turns below.\n"
                "Output ONLY the answer — a name, place, date, or short phrase (1-20 words).\n"
                "Use EXACT words from the conversation for named entities (people, places, countries).\n"
                "Do not paraphrase or describe — copy the specific name/fact.\n\n"
                f"Conversation:\n{context_text}\n\n"
                f"Question: {question}\nAnswer:"
            )

        _model = model or str(settings.get_llm_model("agents"))
        # Use caller-provided timeout when available; the config default (30s) is too short
        # for large models that need 60-120s to load+infer on first call after idle.
        _default_timeout = float(getattr(settings, "VLLM_TIMEOUT_SECONDS", 30.0))
        _timeout = float(timeout_seconds) if timeout_seconds is not None else _default_timeout
        # Clamp: minimum 30s, maximum 300s to avoid unbounded waits
        _timeout = max(30.0, min(300.0, _timeout))

        used_llm = False
        answer_source = "gateway_empty"
        llm_error: str | None = None
        llm_failure_mode: str | None = None
        answer_text = ""
        model_used = _model
        llm_endpoint: str | None = None

        try:
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=_model,
                timeout_seconds=_timeout,
                max_concurrency=_effective_llm_max_concurrency(task="answer"),
            )
            # keep_alive is handled by the base client (hardcoded to -1 = never evict).
            # The caller's keep_alive hint is noted for logging but not forwarded as a kwarg.
            _ = keep_alive  # acknowledged; base client always uses keep_alive=-1
            # Guard against transport/client edge cases where the underlying
            # call can stall longer than configured HTTP timeouts.
            answer_text = await asyncio.wait_for(
                client.complete_text(prompt=prompt, num_ctx=effective_num_ctx),
                timeout=_timeout + 5.0,
            )
            used_llm = bool(answer_text)
            model_used = str(getattr(client, "last_model_used", "") or _model)
            llm_endpoint = str(getattr(client, "last_endpoint_used", "") or "") or None
            if used_llm:
                answer_source = "gateway"
            else:
                llm_error = str(getattr(client, "last_error", "") or "empty_response")
                llm_failure_mode = _classify_llm_failure(llm_error)
                if llm_failure_mode and llm_failure_mode != "empty_response":
                    answer_source = "gateway_error"
        except Exception as exc:
            llm_error = repr(exc)
            llm_failure_mode = _classify_llm_failure(llm_error)
            answer_source = "gateway_error"
            logger.warning(
                "vllm answer fallback: model=%s timeout=%s exc=%s: %s",
                _model,
                _timeout,
                type(exc).__name__,
                exc,
            )

        return GatewayAnswerResult(
            answer=answer_text,
            model=model_used,
            context_turns=len(context_lines),
            used_llm=used_llm,
            answer_source=answer_source,
            llm_error=llm_error,
            llm_failure_mode=llm_failure_mode,
            llm_endpoint=llm_endpoint,
        )

    # ------------------------------------------------------------------
    # judge
    # ------------------------------------------------------------------

    async def judge(
        self,
        *,
        question: str,
        gold: str,
        generated: str,
        model: str | None = None,
    ) -> GatewayJudgeResult:
        """Semantic equivalence check: is generated answer correct for question?

        Returns equivalent=True/False, or None if the judge call fails.
        """
        self._check("answer")  # same capability gate as answer

        prompt = (
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"System answer: {generated}\n"
            "Is the system answer semantically equivalent to the gold answer for this question? "
            "Reply with only \"yes\" or \"no\". No explanation.\nAnswer:"
        )

        _model = model or str(settings.get_llm_model("agents"))
        raw = ""

        try:
            client = create_llm_client(
                base_url=str(getattr(settings, "VLLM_BASE_URL", "http://localhost:11434")),
                model=_model,
                timeout_seconds=60.0,
                max_concurrency=_effective_llm_max_concurrency(task="judge"),
            )
            raw = await client.complete_text(prompt=prompt, num_ctx=512, temperature=0.0)
            _model = str(getattr(client, "last_model_used", "") or _model)
        except Exception:
            pass

        if not raw:
            return GatewayJudgeResult(equivalent=None, raw="", model=_model)

        equivalent = raw.strip().lower().startswith("y")
        return GatewayJudgeResult(equivalent=equivalent, raw=raw, model=_model)

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------

    async def explain(
        self,
        *,
        memory_id: str,
        audit_records: list[dict] | None = None,
        context_id: str | None = None,
        org_id: str | None = None,
    ) -> GatewayExplainResult:
        """Return audit trail and explainability summary for a memory."""
        self._check("explain")
        result = _heuristic_explain(memory_id, audit_records or [])

        if context_id and org_id:
            ctx = await load_gateway_context(context_id, org_id) or GatewayContextSession(
                context_id=context_id, org_id=org_id
            )
            ctx.working_set_summary = self._tier_manager.reconcile(
                working_set=list(ctx.working_set_summary.get("working_set") or []),
                archival=list(ctx.working_set_summary.get("archival") or []),
                incoming=[],
            )
            ctx.call_count += 1
            await save_gateway_context(ctx)

        return result
