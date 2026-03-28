"""Memory Sleep Cycle — Phase 40.

Per-memory agent that runs during the nightly offline sleep cycle.
Analyses each enriched memory and assigns a sleep action:
  - retain     — healthy memory, no intervention needed
  - strengthen — low freshness but high reference value; schedule a boost
  - merge      — redundant candidates exist; flag for offline merge
  - prune      — stale and redundant; mark for archival

Also performs a creative-association pass: discovers non-obvious links
between this memory and other high-similarity neighbours that are NOT
already merge candidates.

Depends on:
  - MemoryDecayAgent (Phase 15)          — freshness_score, is_stale
  - MemoryConsolidationAgent (Phase 16)  — consolidation_action, merge_candidates,
                                           redundancy_score
  - EpisodicGroupingAgent (Phase 18)     — episode_key, episode_cohesion,
                                           cross_episode_links

Inputs (via context["memory"]):
  - content:       raw text of the memory (optional, used for LLM path)

Inputs (via context["memory"]["enrichment"]):
  - freshness_score:       float 0..1  (Phase 15)
  - is_stale:              bool        (Phase 15)
  - reference_count:       int
  - consolidation_action:  str         (Phase 16) keep|merge_candidate|compress|archive
  - merge_candidates:      list[dict]  (Phase 16) [{id, similarity_score, reason}]
  - redundancy_score:      float 0..1  (Phase 16)
  - episode_key:           str         (Phase 18)
  - episode_cohesion:      float 0..1  (Phase 18)
  - cross_episode_links:   list[str]   (Phase 18) related episode keys
  - similar_memories:      list[dict]  [{id, similarity_score, domain, ...}]

Outputs:
  - sleep_action:       "retain" | "strengthen" | "merge" | "prune"
  - association_links:  list[str] — memory IDs for novel cross-links
  - association_reason: str | None
  - strength_delta:     float — intended freshness adjustment
  - prune_reason:       str | None
  - confidence:         float 0..1
  - rationale:          "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# Thresholds ----------------------------------------------------------------

_PRUNE_REDUNDANCY_THRESHOLD = 0.50
_STRENGTHEN_FRESHNESS_MAX = 0.30
_STRENGTHEN_REF_MIN = 3
_ASSOCIATION_SIMILARITY_MIN = 0.75
_MAX_ASSOCIATION_LINKS = 5

# Strength deltas per action
_DELTA: dict[str, float] = {
    "retain":    0.0,
    "strengthen": 0.20,
    "merge":    -0.20,
    "prune":     0.0,
}

_VALID_ACTIONS = frozenset({"retain", "strengthen", "merge", "prune"})


# ---------------------------------------------------------------------------
# Pure heuristic helpers
# ---------------------------------------------------------------------------

def decide_sleep_action(
    *,
    consolidation_action: str,
    is_stale: bool,
    redundancy_score: float,
    merge_candidates: list[dict[str, Any]],
    freshness_score: float,
    reference_count: int,
) -> str:
    """Priority-ordered sleep action selection.

    1. prune      — Phase 16 said "archive", or stale + highly redundant
    2. merge      — merge candidates exist from Phase 16
    3. strengthen — low freshness but still referenced
    4. retain     — default
    """
    if consolidation_action == "archive" or (
        is_stale and float(redundancy_score) >= _PRUNE_REDUNDANCY_THRESHOLD
    ):
        return "prune"

    if merge_candidates:
        return "merge"

    if (
        float(freshness_score) < _STRENGTHEN_FRESHNESS_MAX
        and int(reference_count) >= _STRENGTHEN_REF_MIN
    ):
        return "strengthen"

    return "retain"


def build_association_links(
    similar_memories: list[dict[str, Any]],
    merge_candidate_ids: set[str],
) -> list[str]:
    """Return novel cross-link IDs: high-similarity memories not already flagged for merge."""
    links: list[str] = []
    for mem in similar_memories or []:
        mem_id = str(mem.get("id") or "")
        if not mem_id or mem_id in merge_candidate_ids:
            continue
        sim = float(mem.get("similarity_score") or 0.0)
        if sim >= _ASSOCIATION_SIMILARITY_MIN:
            links.append(mem_id)
        if len(links) >= _MAX_ASSOCIATION_LINKS:
            break
    return links


def build_prune_reason(
    *,
    consolidation_action: str,
    is_stale: bool,
    redundancy_score: float,
) -> str | None:
    if consolidation_action == "archive":
        return "consolidation_agent_flagged_archive"
    if is_stale and float(redundancy_score) >= _PRUNE_REDUNDANCY_THRESHOLD:
        return f"stale_and_redundant(score={redundancy_score:.2f})"
    return None


def build_association_reason(links: list[str]) -> str | None:
    if not links:
        return None
    return f"high_similarity_cross_link({len(links)} memories)"


def action_confidence(
    action: str,
    *,
    consolidation_action: str,
    merge_candidates: list[dict[str, Any]],
    is_stale: bool,
    redundancy_score: float,
) -> float:
    if action == "prune":
        return 0.80 if consolidation_action == "archive" else 0.70
    if action == "merge":
        return 0.80
    if action == "strengthen":
        return 0.75
    return 0.60


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MemorySleepAgent(BaseAgent):
    """Phase 40: Per-memory sleep cycle decision engine.

    Runs offline (nightly) to strengthen valuable-but-fading memories,
    merge near-duplicates, prune stale content, and build creative
    associations across episode boundaries.
    """

    name = "MemorySleepAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["MemoryDecayAgent", "MemoryConsolidationAgent", "EpisodicGroupingAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        action = outputs.get("sleep_action")
        if action not in _VALID_ACTIONS:
            raise ValueError(f"invalid sleep_action: {action!r}")

        if not isinstance(outputs.get("association_links"), list):
            raise ValueError("association_links must be a list")

        delta = outputs.get("strength_delta")
        if not isinstance(delta, (int, float)):
            raise ValueError("strength_delta must be a float")

        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    def _heuristic(
        self,
        *,
        consolidation_action: str,
        is_stale: bool,
        redundancy_score: float,
        merge_candidates: list[dict[str, Any]],
        freshness_score: float,
        reference_count: int,
        similar_memories: list[dict[str, Any]],
    ) -> dict[str, Any]:
        action = decide_sleep_action(
            consolidation_action=consolidation_action,
            is_stale=is_stale,
            redundancy_score=redundancy_score,
            merge_candidates=merge_candidates,
            freshness_score=freshness_score,
            reference_count=reference_count,
        )

        merge_ids = {str(c.get("id") or "") for c in merge_candidates}
        assoc_links = build_association_links(similar_memories, merge_ids)
        prune_reason = build_prune_reason(
            consolidation_action=consolidation_action,
            is_stale=is_stale,
            redundancy_score=redundancy_score,
        )
        assoc_reason = build_association_reason(assoc_links)
        confidence = action_confidence(
            action,
            consolidation_action=consolidation_action,
            merge_candidates=merge_candidates,
            is_stale=is_stale,
            redundancy_score=redundancy_score,
        )

        return {
            "sleep_action": action,
            "association_links": assoc_links,
            "association_reason": assoc_reason,
            "strength_delta": _DELTA[action],
            "prune_reason": prune_reason,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        freshness_score = float(enrichment.get("freshness_score") or 0.5)
        is_stale = bool(enrichment.get("is_stale", False))
        reference_count = int(enrichment.get("reference_count") or 0)
        consolidation_action = str(enrichment.get("consolidation_action") or "keep")
        merge_candidates = list(enrichment.get("merge_candidates") or [])
        redundancy_score = float(enrichment.get("redundancy_score") or 0.0)
        similar_memories = list(enrichment.get("similar_memories") or [])

        strategy = getattr(settings, "MEMORY_SLEEP_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = self._heuristic(
                consolidation_action=consolidation_action,
                is_stale=is_stale,
                redundancy_score=redundancy_score,
                merge_candidates=merge_candidates,
                freshness_score=freshness_score,
                reference_count=reference_count,
                similar_memories=similar_memories,
            )
        else:
            prompt = (
                "You are a memory sleep cycle engine for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Decide the sleep action for this memory.\n\n"
                f"CONSOLIDATION_ACTION: {consolidation_action}\n"
                f"IS_STALE: {is_stale}\n"
                f"REDUNDANCY_SCORE: {redundancy_score}\n"
                f"FRESHNESS_SCORE: {freshness_score}\n"
                f"REFERENCE_COUNT: {reference_count}\n"
                f"MERGE_CANDIDATES_COUNT: {len(merge_candidates)}\n"
                f"SIMILAR_MEMORIES_COUNT: {len(similar_memories)}\n"
                f"CONTENT: {content[:200]}\n\n"
                "Return JSON with keys:\n"
                "- sleep_action: one of retain|strengthen|merge|prune\n"
                "- association_links: list of memory ID strings for novel cross-links\n"
                "- association_reason: string or null\n"
                "- strength_delta: float\n"
                "- prune_reason: string or null\n"
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
                and resp.get("sleep_action") in _VALID_ACTIONS
                and isinstance(resp.get("association_links"), list)
                and isinstance(resp.get("strength_delta"), (int, float))
                and isinstance(resp.get("confidence"), (int, float))
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    consolidation_action=consolidation_action,
                    is_stale=is_stale,
                    redundancy_score=redundancy_score,
                    merge_candidates=merge_candidates,
                    freshness_score=freshness_score,
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
