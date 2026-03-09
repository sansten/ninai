"""Credibility Scoring — Phase 19.

Scores each memory's source trustworthiness based on author role, citation
depth, corroboration from other memories, and conflict exposure from Phase 13.

Feeds into:
  - ConflictDetectionAgent (Phase 13) — adjusts conflict severity weighting
  - MemoryConsolidationAgent (Phase 16) — prefers higher-credibility sources

Depends on:
  - EntityResolutionAgent (Phase 7)   — entities
  - ConflictDetectionAgent (Phase 13) — conflicts

Inputs (via context["memory"]):
  - content:       raw text of the memory
  - source_type:   optional str — e.g. "api", "document", "text", "import", "system"
  - author_role:   optional str — e.g. "engineer", "analyst", "external", "system"

Inputs (via context["memory"]["enrichment"]):
  - entities:               resolved entities dict (Phase 7)
  - conflicts:              list[{entity, severity, ...}] — Phase 13
  - corroborating_memories: list[{id, content, entities}] — pre-fetched memories
                            sharing key entities with this one

Outputs:
  - credibility_score:  float 0..1
  - source_tier:        "primary" | "secondary" | "tertiary" | "unknown"
  - corroboration_count: int
  - credibility_flags:  list[str]
  - confidence:         float
  - rationale:          "heuristic" | "llm"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Source tier tables
# ---------------------------------------------------------------------------

_PRIMARY_SOURCE_TYPES: frozenset[str] = frozenset({
    "api", "system", "internal", "webhook",
})

_SECONDARY_SOURCE_TYPES: frozenset[str] = frozenset({
    "document", "import", "integration", "upload", "file",
})

_TERTIARY_SOURCE_TYPES: frozenset[str] = frozenset({
    "scrape", "crawl", "aggregated", "derived",
})

_PRIMARY_AUTHOR_ROLES: frozenset[str] = frozenset({
    "engineer", "developer", "analyst", "manager", "owner",
    "admin", "author", "contributor", "lead", "architect",
    "system", "service",
})

_SECONDARY_AUTHOR_ROLES: frozenset[str] = frozenset({
    "reporter", "observer", "reviewer", "auditor",
    "external_partner", "vendor", "consultant",
})

_TERTIARY_AUTHOR_ROLES: frozenset[str] = frozenset({
    "anonymous", "aggregator", "bot", "unknown",
})

# Hearsay patterns that push tier toward tertiary
_HEARSAY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bi heard\b", re.IGNORECASE),
    re.compile(r"\breportedly\b", re.IGNORECASE),
    re.compile(r"\bapparently\b", re.IGNORECASE),
    re.compile(r"\brumored?\b", re.IGNORECASE),
    re.compile(r"\bsomeone said\b", re.IGNORECASE),
    re.compile(r"\bword is\b", re.IGNORECASE),
)

# Citation signals that increase trustworthiness
_CITATION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\baccording to\b", re.IGNORECASE),
    re.compile(r"\bper\s+\w", re.IGNORECASE),
    re.compile(r"\bsource:", re.IGNORECASE),
    re.compile(r"\bref:", re.IGNORECASE),
    re.compile(r"\breference:", re.IGNORECASE),
    re.compile(r"\bcite:", re.IGNORECASE),
    re.compile(r"\bsee also:", re.IGNORECASE),
    re.compile(r"\bbased on\b", re.IGNORECASE),
    re.compile(r"\bvia\s+\w", re.IGNORECASE),
    re.compile(r"\bcitation needed\b", re.IGNORECASE),
    re.compile(r"\bsee\s+\[", re.IGNORECASE),
)

# Base credibility by tier
_TIER_BASE: dict[str, float] = {
    "primary": 0.80,
    "secondary": 0.60,
    "tertiary": 0.35,
    "unknown": 0.45,
}

_VALID_TIERS: frozenset[str] = frozenset(_TIER_BASE.keys())

_SEVERITY_HIGH = "high"


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def classify_source_tier(
    *,
    source_type: str,
    author_role: str,
    content: str,
) -> str:
    """Determine the source tier from available signals.

    Priority order:
    1. source_type drives the tier if recognised
    2. author_role overrides when source_type is absent or neutral
    3. Hearsay patterns in content push to tertiary
    4. Falls back to "unknown" when no signals are available
    """
    st = (source_type or "").strip().lower()
    ar = (author_role or "").strip().lower()

    # If we have a recognised source_type, it takes precedence
    if st in _PRIMARY_SOURCE_TYPES:
        return "primary"
    if st in _SECONDARY_SOURCE_TYPES:
        # But a tertiary author role still drags it down
        if ar in _TERTIARY_AUTHOR_ROLES:
            return "tertiary"
        return "secondary"
    if st in _TERTIARY_SOURCE_TYPES:
        return "tertiary"

    # No meaningful source_type — fall back to author_role
    if ar in _PRIMARY_AUTHOR_ROLES:
        return "primary"
    if ar in _SECONDARY_AUTHOR_ROLES:
        return "secondary"
    if ar in _TERTIARY_AUTHOR_ROLES:
        return "tertiary"

    # Hearsay in content pushes to tertiary even without explicit signals
    if any(p.search(content) for p in _HEARSAY_PATTERNS):
        return "tertiary"

    return "unknown"


def count_citation_depth(content: str) -> int:
    """Count distinct citation signals in the content (capped at 10)."""
    if not content:
        return 0
    count = 0
    for pattern in _CITATION_PATTERNS:
        matches = pattern.findall(content)
        count += len(matches)
    return min(count, 10)


def count_corroboration(corroborating_memories: list[dict[str, Any]]) -> int:
    """Return number of pre-fetched memories that corroborate this one."""
    if not corroborating_memories:
        return 0
    return len([m for m in corroborating_memories if isinstance(m, dict)])


def count_conflict_exposure(
    *,
    entities: Any,
    conflicts: list[dict[str, Any]],
) -> tuple[int, int]:
    """Count how many of this memory's entities appear in detected conflicts.

    Returns (total_conflict_count, high_severity_count).
    """
    entity_names = _extract_entity_names(entities)
    if not entity_names or not conflicts:
        return 0, 0

    total = 0
    high = 0
    for conflict in conflicts:
        c_entity = str(conflict.get("entity") or "").lower().strip()
        if c_entity and c_entity in entity_names:
            total += 1
            if conflict.get("severity") == _SEVERITY_HIGH:
                high += 1

    return total, high


def _extract_entity_names(entities: Any) -> set[str]:
    """Flatten heterogeneous entity payloads to a set of lowercase names."""
    names: set[str] = set()

    if isinstance(entities, dict):
        for key, value in entities.items():
            names.add(str(key).lower().strip())
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        for candidate in (
                            item.get("canonical"),
                            item.get("original"),
                            item.get("name"),
                        ):
                            if candidate:
                                names.add(str(candidate).lower().strip())
                    elif value is not None:
                        names.add(str(item).lower().strip())
            elif isinstance(value, dict):
                for candidate in (
                    value.get("canonical"),
                    value.get("original"),
                    value.get("name"),
                ):
                    if candidate:
                        names.add(str(candidate).lower().strip())
            elif value is not None:
                names.add(str(value).lower().strip())

    elif isinstance(entities, list):
        for item in entities:
            if isinstance(item, dict):
                for candidate in (
                    item.get("canonical"),
                    item.get("original"),
                    item.get("name"),
                    item.get("entity"),
                ):
                    if candidate:
                        names.add(str(candidate).lower().strip())
            elif item is not None:
                names.add(str(item).lower().strip())

    elif entities is not None:
        names.add(str(entities).lower().strip())

    return {n for n in names if n}


def build_credibility_flags(
    *,
    source_type: str,
    author_role: str,
    source_tier: str,
    citation_depth: int,
    content: str,
    entities: Any,
    conflicts: list[dict[str, Any]],
) -> list[str]:
    """Build a list of warning flags that explain credibility concerns."""
    flags: list[str] = []

    # No source information at all
    if not (source_type or "").strip() and not (author_role or "").strip():
        flags.append("no_source_info")

    # Low citation depth for substantial content
    if citation_depth == 0 and len(content) > 200:
        flags.append("low_citation_depth")

    # Tertiary or unknown source tier
    if source_tier == "tertiary":
        flags.append("tertiary_source")
    elif source_tier == "unknown" and citation_depth == 0:
        flags.append("unverified_claim")

    # Hearsay detected in content
    if any(p.search(content) for p in _HEARSAY_PATTERNS):
        flags.append("hearsay_content")

    # Flag entities appearing in high-severity conflicts
    entity_names = _extract_entity_names(entities)
    for conflict in conflicts or []:
        c_entity = str(conflict.get("entity") or "").lower().strip()
        if c_entity and c_entity in entity_names and conflict.get("severity") == _SEVERITY_HIGH:
            flags.append(f"conflicted_entity:{c_entity}")

    return flags


def compute_credibility_score(
    *,
    source_tier: str,
    citation_depth: int,
    corroboration_count: int,
    high_severity_conflict_count: int,
) -> float:
    """Compute credibility_score in [0.05, 0.95].

    Formula:
      base              = tier base score
      citation_bonus    = min(0.10, 0.02 × citation_depth)
      corroboration_bonus = min(0.10, 0.025 × corroboration_count)
      conflict_penalty  = 0.05 × min(3, high_severity_conflict_count)
      score = clamp(base + bonuses - penalty, 0.05, 0.95)
    """
    base = _TIER_BASE.get(source_tier, _TIER_BASE["unknown"])
    citation_bonus = min(0.10, 0.02 * citation_depth)
    corroboration_bonus = min(0.10, 0.025 * corroboration_count)
    conflict_penalty = 0.05 * min(3, high_severity_conflict_count)
    score = base + citation_bonus + corroboration_bonus - conflict_penalty
    return round(min(max(score, 0.05), 0.95), 4)


def compute_agent_confidence(
    *,
    source_type: str,
    author_role: str,
    corroboration_count: int,
) -> float:
    """Confidence in the credibility assessment itself.

    Higher when we have explicit source metadata and corroborating memories.
    """
    has_source = bool((source_type or "").strip() or (author_role or "").strip())
    if has_source and corroboration_count >= 2:
        return 0.82
    if has_source:
        return 0.75
    if corroboration_count >= 1:
        return 0.65
    return 0.55


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CredibilityAgent(BaseAgent):
    """Phase 19: Score each memory's source trustworthiness.

    Assigns a credibility_score and source_tier to each memory, flags
    trust concerns, and surfaces which entities are under conflict exposure.
    Outputs feed ConflictDetectionAgent and MemoryConsolidationAgent
    to bias toward higher-credibility sources.
    """

    name = "CredibilityAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["EntityResolutionAgent", "ConflictDetectionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        score = outputs.get("credibility_score")
        if not isinstance(score, (int, float)) or not (0.0 <= float(score) <= 1.0):
            raise ValueError("credibility_score must be a float in [0, 1]")

        tier = outputs.get("source_tier")
        if tier not in _VALID_TIERS:
            raise ValueError(f"invalid source_tier: {tier!r}")

        if not isinstance(outputs.get("corroboration_count"), int):
            raise ValueError("corroboration_count must be an int")

        if not isinstance(outputs.get("credibility_flags"), list):
            raise ValueError("credibility_flags must be a list")

    def _heuristic(
        self,
        *,
        content: str,
        source_type: str,
        author_role: str,
        entities: Any,
        conflicts: list[dict[str, Any]],
        corroborating_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        source_tier = classify_source_tier(
            source_type=source_type,
            author_role=author_role,
            content=content,
        )
        citation_depth = count_citation_depth(content)
        corroboration_count = count_corroboration(corroborating_memories)
        _total_conflicts, high_conflicts = count_conflict_exposure(
            entities=entities,
            conflicts=conflicts,
        )

        credibility_score = compute_credibility_score(
            source_tier=source_tier,
            citation_depth=citation_depth,
            corroboration_count=corroboration_count,
            high_severity_conflict_count=high_conflicts,
        )

        credibility_flags = build_credibility_flags(
            source_type=source_type,
            author_role=author_role,
            source_tier=source_tier,
            citation_depth=citation_depth,
            content=content,
            entities=entities,
            conflicts=conflicts,
        )

        confidence = compute_agent_confidence(
            source_type=source_type,
            author_role=author_role,
            corroboration_count=corroboration_count,
        )

        return {
            "credibility_score": credibility_score,
            "source_tier": source_tier,
            "corroboration_count": corroboration_count,
            "credibility_flags": credibility_flags,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        source_type = str(memory.get("source_type") or "").strip()
        author_role = str(memory.get("author_role") or "").strip()

        entities = enrichment.get("entities") or enrichment.get("resolved_entities") or {}
        conflicts = list(enrichment.get("conflicts") or [])
        corroborating_memories = list(enrichment.get("corroborating_memories") or [])

        strategy = getattr(settings, "CREDIBILITY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                source_type=source_type,
                author_role=author_role,
                entities=entities,
                conflicts=conflicts,
                corroborating_memories=corroborating_memories,
            )
        else:
            prompt = (
                "You are an enterprise memory credibility scoring engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Score the trustworthiness of this memory based on source signals, "
                "citation depth, corroboration, and conflict exposure.\n\n"
                f"CONTENT: {content[:400]}\n"
                f"SOURCE_TYPE: {source_type or 'unknown'}\n"
                f"AUTHOR_ROLE: {author_role or 'unknown'}\n"
                f"ENTITIES: {list(_extract_entity_names(entities))[:5]}\n"
                f"CONFLICTS: {conflicts[:5]}\n"
                f"CORROBORATING_MEMORIES_COUNT: {len(corroborating_memories)}\n\n"
                "Return JSON with keys:\n"
                "- credibility_score: float 0..1\n"
                "- source_tier: one of primary|secondary|tertiary|unknown\n"
                "- corroboration_count: int\n"
                "- credibility_flags: list of strings\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation"
            )
            client = create_ollama_client(
                base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                model=str(getattr(settings, "OLLAMA_MODEL", "llama3.1:8b")),
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
                and resp.get("source_tier") in _VALID_TIERS
                and isinstance(resp.get("credibility_score"), (int, float))
                and isinstance(resp.get("credibility_flags"), list)
                and isinstance(resp.get("corroboration_count"), int)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    source_type=source_type,
                    author_role=author_role,
                    entities=entities,
                    conflicts=conflicts,
                    corroborating_memories=corroborating_memories,
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.55)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
