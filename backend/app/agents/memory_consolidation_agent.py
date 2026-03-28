"""Memory Consolidation Engine — Phase 16.

Per-memory agent that decides whether a memory should be kept as-is,
flagged as a merge candidate with similar memories, compressed into a
summary, or archived.

Depends on:
  - SemanticNormalizationAgent (Phase 6)  — domain, intent, tags
  - EntityResolutionAgent (Phase 7)       — entities
  - MemoryDecayAgent (Phase 15)           — freshness_score, is_stale

Inputs (via context["memory"]):
  - content:           raw text of the memory

Inputs (via context["memory"]["enrichment"]):
  - domain:            domain string (Phase 6)
  - entities:          dict of resolved entities (Phase 7)
  - is_stale:          bool (Phase 15)
  - reference_count:   int (Phase 6 / Phase 15)
  - similar_memories:  list[dict] — pre-fetched vector/text candidates (optional)
                       each: {id, content_preview, similarity_score, domain, freshness_score}

Outputs:
  - consolidation_action:  "keep" | "merge_candidate" | "compress" | "archive"
  - merge_candidates:      list[{id, similarity_score, reason}]
  - compression_summary:   str | None
  - redundancy_score:      float 0..1
  - deduplication_key:     str
  - candidate_count:       int
  - confidence:            float
  - rationale:             "heuristic" | "llm"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "and", "or", "but", "not", "it",
    "this", "that", "its", "as", "up", "out", "if", "about", "into",
})

_MIN_MERGE_SIMILARITY = 0.70
_HIGH_SIMILARITY = 0.85
_MAX_MERGE_CANDIDATES = 10
_MAX_DEDUP_KEY_TOKENS = 15
_STALE_REDUNDANCY_BOOST = 0.10
_LOW_REF_REDUNDANCY_BOOST = 0.10
_LOW_REF_THRESHOLD = 2
_ARCHIVE_REDUNDANCY_THRESHOLD = 0.50


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanum tokens with stopwords and short words removed."""
    return [
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t not in _STOPWORDS and len(t) >= 3
    ]


def build_deduplication_key(
    *, domain: str, entities: dict[str, Any], content: str
) -> str:
    """Build a normalised fingerprint for exact/near-exact dedup detection.

    Key = domain + sorted entity names + top content tokens (sorted),
    joined by '|'. Case-insensitive, stopwords excluded.
    """
    parts: list[str] = [str(domain or "").lower().strip()]

    entity_tokens: set[str] = set()
    for key, value in (entities or {}).items():
        entity_tokens.add(str(key).lower().strip())
        if isinstance(value, list):
            for v in value:
                entity_tokens.add(str(v).lower().strip())
        else:
            entity_tokens.add(str(value).lower().strip())

    parts.extend(sorted(entity_tokens))

    content_tokens = sorted(set(_tokenize(content)))[:_MAX_DEDUP_KEY_TOKENS]
    parts.extend(content_tokens)

    return "|".join(t for t in parts if t)


def filter_merge_candidates(
    similar_memories: list[dict[str, Any]],
    *,
    domain: str,
    min_similarity: float = _MIN_MERGE_SIMILARITY,
) -> list[dict[str, Any]]:
    """Return merge candidates from similar_memories.

    Same-domain memories above min_similarity, or cross-domain memories
    above _HIGH_SIMILARITY, qualify. Sorted by similarity desc.
    """
    results: list[dict[str, Any]] = []
    for mem in similar_memories or []:
        mem_id = str(mem.get("id") or "")
        if not mem_id:
            continue
        sim = float(mem.get("similarity_score") or 0.0)
        mem_domain = str(mem.get("domain") or "").lower()
        this_domain = str(domain or "").lower()
        same_domain = mem_domain == this_domain

        if same_domain and sim >= min_similarity:
            reason = "same_domain_similar"
        elif not same_domain and sim >= _HIGH_SIMILARITY:
            reason = "cross_domain_near_duplicate"
        else:
            continue

        results.append({
            "id": mem_id,
            "similarity_score": round(sim, 4),
            "reason": reason,
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:_MAX_MERGE_CANDIDATES]


def compute_redundancy_score(
    *,
    merge_candidates: list[dict[str, Any]],
    is_stale: bool,
    reference_count: int,
    similar_memories: list[dict[str, Any]] | None = None,
) -> float:
    """Compute redundancy score 0..1.

    Base: max similarity across all similar_memories (falls back to
    max of merge_candidates if similar_memories not supplied).
    Boosts: staleness (+0.10), low reference count (+0.10).
    """
    if similar_memories is not None:
        base = max(
            (float(m.get("similarity_score") or 0.0) for m in similar_memories),
            default=0.0,
        )
    else:
        base = max(
            (float(c.get("similarity_score") or 0.0) for c in merge_candidates),
            default=0.0,
        )
    boost = 0.0
    if is_stale:
        boost += _STALE_REDUNDANCY_BOOST
    if int(reference_count or 0) < _LOW_REF_THRESHOLD:
        boost += _LOW_REF_REDUNDANCY_BOOST
    return round(min(base + boost, 1.0), 4)


def decide_action(
    *,
    merge_candidates: list[dict[str, Any]],
    is_stale: bool,
    redundancy_score: float,
    reference_count: int,
) -> str:
    """Select consolidation action for this memory.

    Priority:
    1. merge_candidate — similar memories found
    2. archive         — stale and highly redundant
    3. compress        — stale and infrequently accessed
    4. keep            — default
    """
    if merge_candidates:
        return "merge_candidate"
    if is_stale and float(redundancy_score) >= _ARCHIVE_REDUNDANCY_THRESHOLD:
        return "archive"
    if is_stale and int(reference_count or 0) <= 1:
        return "compress"
    return "keep"


def build_compression_summary(
    *, action: str, domain: str, content: str
) -> str | None:
    """Generate a compact summary for compress/archive actions.

    Returns None for keep and merge_candidate actions.
    """
    if action in ("keep", "merge_candidate"):
        return None
    prefix = str(domain or "general").capitalize()
    excerpt = (content or "").strip()[:150]
    if excerpt and not excerpt.endswith((".", "?", "!")):
        excerpt = excerpt + "\u2026"
    return f"[{prefix}] {excerpt}" if excerpt else f"[{prefix}]"


class MemoryConsolidationAgent(BaseAgent):
    """Phase 16: Per-memory consolidation decision engine.

    Analyses each enriched memory and decides whether to keep it, flag
    it for merging with similar memories, compress it into a summary, or
    archive it. Feeds into ConsolidationService for actual DB operations.
    """

    name = "MemoryConsolidationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SemanticNormalizationAgent", "EntityResolutionAgent", "MemoryDecayAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        action = outputs.get("consolidation_action")
        if action not in ("keep", "merge_candidate", "compress", "archive"):
            raise ValueError(f"invalid consolidation_action: {action!r}")

        rs = outputs.get("redundancy_score")
        if not isinstance(rs, (int, float)) or not (0.0 <= float(rs) <= 1.0):
            raise ValueError("redundancy_score must be a float in [0, 1]")

        if not isinstance(outputs.get("merge_candidates"), list):
            raise ValueError("merge_candidates must be a list")

        if not isinstance(outputs.get("candidate_count"), int):
            raise ValueError("candidate_count must be an int")

        if not isinstance(outputs.get("deduplication_key"), str):
            raise ValueError("deduplication_key must be a str")

    def _heuristic(
        self,
        *,
        content: str,
        domain: str,
        entities: dict[str, Any],
        is_stale: bool,
        reference_count: int,
        similar_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        dedup_key = build_deduplication_key(
            domain=domain, entities=entities, content=content
        )
        candidates = filter_merge_candidates(similar_memories, domain=domain)
        redundancy_score = compute_redundancy_score(
            merge_candidates=candidates,
            is_stale=is_stale,
            reference_count=reference_count,
            similar_memories=similar_memories,
        )
        action = decide_action(
            merge_candidates=candidates,
            is_stale=is_stale,
            redundancy_score=redundancy_score,
            reference_count=reference_count,
        )
        compression_summary = build_compression_summary(
            action=action, domain=domain, content=content
        )

        if candidates:
            confidence = 0.85
        elif is_stale and redundancy_score >= _ARCHIVE_REDUNDANCY_THRESHOLD:
            confidence = 0.75
        elif is_stale:
            confidence = 0.70
        else:
            confidence = 0.60

        return {
            "consolidation_action": action,
            "merge_candidates": candidates,
            "compression_summary": compression_summary,
            "redundancy_score": redundancy_score,
            "deduplication_key": dedup_key,
            "candidate_count": len(candidates),
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        domain = str(enrichment.get("domain") or "")
        entities = dict(enrichment.get("entities") or {})
        is_stale = bool(enrichment.get("is_stale", False))
        reference_count = int(enrichment.get("reference_count") or 0)
        similar_memories = list(enrichment.get("similar_memories") or [])

        strategy = getattr(settings, "MEMORY_CONSOLIDATION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                domain=domain,
                entities=entities,
                is_stale=is_stale,
                reference_count=reference_count,
                similar_memories=similar_memories,
            )
        else:
            sim_json = similar_memories[:5]
            prompt = (
                "You are a memory consolidation engine for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Decide the consolidation action for this memory.\n\n"
                f"DOMAIN: {domain}\n"
                f"IS_STALE: {is_stale}\n"
                f"REFERENCE_COUNT: {reference_count}\n"
                f"SIMILAR_MEMORIES: {sim_json}\n"
                f"CONTENT: {content[:300]}\n\n"
                "Return JSON with keys:\n"
                "- consolidation_action: one of keep|merge_candidate|compress|archive\n"
                "- merge_candidates: list of {id, similarity_score, reason} objects\n"
                "- compression_summary: string or null\n"
                "- redundancy_score: float 0..1\n"
                "- deduplication_key: string\n"
                "- candidate_count: int\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(settings.get_ollama_model("agents")),
                timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
            )
            resp = await client.complete_json(
                prompt=prompt,
                schema_hint={},
                tool_event_sink=context.get("tool_event_sink"),
            )
            if (
                isinstance(resp, dict)
                and resp.get("consolidation_action") in ("keep", "merge_candidate", "compress", "archive")
                and isinstance(resp.get("merge_candidates"), list)
                and isinstance(resp.get("redundancy_score"), (int, float))
                and isinstance(resp.get("candidate_count"), int)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    domain=domain,
                    entities=entities,
                    is_stale=is_stale,
                    reference_count=reference_count,
                    similar_memories=similar_memories,
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.6)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
