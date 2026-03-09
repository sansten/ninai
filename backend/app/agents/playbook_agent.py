"""Playbook / Skill Memory — Phase 20.

Detects recurring action sequences in episodic groups and promotes them to
reusable playbooks. Matches each incoming memory against the catalog of known
playbooks by domain, entity overlap, and step content similarity, tracking
which step in the sequence the memory belongs to. Flags complete unmatched
sequences as candidates for new playbook promotion.

Depends on:
  - EpisodicGroupingAgent (Phase 18)  — episode_key, episode_role
  - EntityResolutionAgent (Phase 7)   — entities

Inputs (via context["memory"]):
  - content:   raw text of the memory

Inputs (via context["memory"]["enrichment"]):
  - episode_key:         str  — stable episode cluster key (Phase 18)
  - episode_role:        str  — opener|body|climax|closer (Phase 18)
  - domain:              str  — normalized domain (Phase 6)
  - entities:            dict — resolved entities (Phase 7)
  - playbook_candidates: list[dict] — pre-fetched matching playbooks
                         each: {id, name, domain, steps: [str],
                                success_rate: float, occurrence_count: int}

Outputs:
  - matched_playbook_id:      str | None — id of the best matching playbook
  - playbook_confidence:      float 0..1 — match strength
  - is_new_playbook_candidate: bool — True when this closes an unmatched sequence
  - step_position:            int | None — 1-based position in matched playbook
  - confidence:               float
  - rationale:                "heuristic" | "llm"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# Roles that indicate the end of an episode arc (sequence complete).
_CLOSING_ROLES: frozenset[str] = frozenset({"closer", "climax"})

# Minimum match score to consider a playbook matched.
_MATCH_THRESHOLD = 0.30

# Common stop words excluded from action token extraction.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "was", "this", "that", "with", "from",
    "have", "been", "not", "but", "will", "can", "all", "one", "has",
    "its", "into", "also", "then", "than", "when", "they", "their",
    "about", "such", "more", "some", "each", "these", "those", "most",
    "much", "many", "only", "over", "other", "our", "out", "she", "him",
    "her", "his", "now", "two", "any", "may", "new", "use", "via",
    "how", "why", "who", "what", "did", "get", "got", "had",
})


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def extract_action_tokens(text: str) -> set[str]:
    """Extract lowercase word tokens (3+ chars) from text, minus stop words."""
    if not text:
        return set()
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def score_step_match(step_text: str, content_tokens: set[str]) -> float:
    """Fraction of step tokens present in content_tokens.

    Returns 0.0 when the step is empty or no content tokens are provided.
    """
    if not step_text or not content_tokens:
        return 0.0
    step_tokens = extract_action_tokens(step_text)
    if not step_tokens:
        return 0.0
    overlap = content_tokens & step_tokens
    return round(len(overlap) / len(step_tokens), 4)


def find_best_step(
    steps: list[str],
    content_tokens: set[str],
) -> tuple[int, float]:
    """Return (1-based step position, score) of the best matching step.

    Returns (0, 0.0) when steps is empty.
    """
    if not steps:
        return 0, 0.0
    best_pos = 0
    best_score = 0.0
    for i, step in enumerate(steps):
        s = score_step_match(step, content_tokens)
        if s > best_score:
            best_score = s
            best_pos = i + 1
    return best_pos, best_score


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
                    elif item is not None:
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


def match_against_playbooks(
    *,
    content: str,
    domain: str,
    entities: Any,
    playbook_candidates: list[dict[str, Any]],
) -> tuple[str | None, float, int | None]:
    """Find the best matching playbook for this memory.

    Returns:
        (matched_playbook_id, playbook_confidence, step_position)
        or (None, 0.0, None) when no candidates or no meaningful match.
    """
    if not playbook_candidates:
        return None, 0.0, None

    content_tokens = extract_action_tokens(content)
    entity_tokens = _extract_entity_names(entities)
    all_tokens = content_tokens | entity_tokens

    norm_domain = str(domain or "").lower().strip()

    best_id: str | None = None
    best_confidence = 0.0
    best_step: int | None = None

    for candidate in playbook_candidates:
        if not isinstance(candidate, dict):
            continue
        steps = candidate.get("steps") or []
        if not steps:
            continue

        step_pos, step_score = find_best_step(list(steps), all_tokens)
        if step_score == 0.0:
            continue

        # Domain alignment factor
        cand_domain = str(candidate.get("domain") or "").lower().strip()
        domain_factor = 1.0 if (cand_domain and cand_domain == norm_domain) else 0.6

        # Reliability factor: weight by success rate and occurrence count
        success_rate = float(candidate.get("success_rate") or 0.5)
        occurrence_count = int(candidate.get("occurrence_count") or 1)
        reliability = 0.7 + 0.3 * min(1.0, (success_rate * occurrence_count) / 5.0)

        score = round(step_score * domain_factor * reliability, 4)

        if score > best_confidence:
            best_confidence = score
            best_id = str(candidate.get("id") or "")
            best_step = step_pos if step_pos > 0 else None

    if best_confidence < _MATCH_THRESHOLD:
        return None, 0.0, None

    return best_id, best_confidence, best_step


def detect_new_playbook_candidate(
    *,
    episode_role: str,
    matched_confidence: float,
) -> bool:
    """True when this memory closes a complete sequence not covered by a playbook.

    We flag new candidates only at the end of an episode arc (closer/climax)
    and only when no existing playbook claimed the sequence.
    """
    if episode_role not in _CLOSING_ROLES:
        return False
    return matched_confidence < _MATCH_THRESHOLD


def compute_agent_confidence(
    *,
    matched_confidence: float,
    is_new_candidate: bool,
    playbook_candidates_count: int,
) -> float:
    """Confidence in the playbook assessment.

    Higher when we have a strong match or substantial candidate catalog.
    """
    if matched_confidence >= 0.70:
        return 0.82
    if matched_confidence >= 0.50:
        return 0.75
    if matched_confidence >= _MATCH_THRESHOLD:
        return 0.68
    if is_new_candidate:
        return 0.62
    if playbook_candidates_count > 0:
        return 0.58
    return 0.50


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlaybookAgent(BaseAgent):
    """Phase 20: Match memories to reusable action playbooks.

    Scans each enriched memory against a catalog of known playbooks, determines
    which step in the sequence this memory represents, and flags complete
    unmatched sequences as candidates for new playbook promotion.
    """

    name = "PlaybookAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["EpisodicGroupingAgent", "EntityResolutionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        pb_confidence = outputs.get("playbook_confidence")
        if not isinstance(pb_confidence, (int, float)) or not (0.0 <= float(pb_confidence) <= 1.0):
            raise ValueError("playbook_confidence must be a float in [0, 1]")

        if not isinstance(outputs.get("is_new_playbook_candidate"), bool):
            raise ValueError("is_new_playbook_candidate must be a bool")

        step_pos = outputs.get("step_position")
        if step_pos is not None and not isinstance(step_pos, int):
            raise ValueError("step_position must be an int or None")

        matched_id = outputs.get("matched_playbook_id")
        if matched_id is not None and not isinstance(matched_id, str):
            raise ValueError("matched_playbook_id must be a str or None")

    def _heuristic(
        self,
        *,
        content: str,
        domain: str,
        entities: Any,
        episode_role: str,
        playbook_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        matched_id, playbook_confidence, step_position = match_against_playbooks(
            content=content,
            domain=domain,
            entities=entities,
            playbook_candidates=playbook_candidates,
        )
        is_new = detect_new_playbook_candidate(
            episode_role=episode_role,
            matched_confidence=playbook_confidence,
        )
        confidence = compute_agent_confidence(
            matched_confidence=playbook_confidence,
            is_new_candidate=is_new,
            playbook_candidates_count=len(playbook_candidates),
        )
        return {
            "matched_playbook_id": matched_id,
            "playbook_confidence": playbook_confidence,
            "is_new_playbook_candidate": is_new,
            "step_position": step_position,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        domain = str(enrichment.get("domain") or "").strip()
        entities = enrichment.get("entities") or enrichment.get("resolved_entities") or {}
        episode_role = str(enrichment.get("episode_role") or "body").strip()
        playbook_candidates = list(enrichment.get("playbook_candidates") or [])

        strategy = getattr(settings, "PLAYBOOK_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                domain=domain,
                entities=entities,
                episode_role=episode_role,
                playbook_candidates=playbook_candidates,
            )
        else:
            prompt = (
                "You are an enterprise playbook matching engine for a memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Match this memory against the provided playbook candidates and "
                "identify which step in a known playbook sequence this memory "
                "represents. Flag if this closes a new unmatched sequence.\n\n"
                f"CONTENT: {content[:400]}\n"
                f"DOMAIN: {domain}\n"
                f"EPISODE_ROLE: {episode_role}\n"
                f"ENTITIES: {sorted(list(_extract_entity_names(entities)))[:5]}\n"
                f"PLAYBOOK_CANDIDATES: {playbook_candidates[:5]}\n\n"
                "Return JSON with keys:\n"
                "- matched_playbook_id: str or null\n"
                "- playbook_confidence: float 0..1\n"
                "- is_new_playbook_candidate: bool\n"
                "- step_position: int or null\n"
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
                and isinstance(resp.get("playbook_confidence"), (int, float))
                and isinstance(resp.get("is_new_playbook_candidate"), bool)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    domain=domain,
                    entities=entities,
                    episode_role=episode_role,
                    playbook_candidates=playbook_candidates,
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.50)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
