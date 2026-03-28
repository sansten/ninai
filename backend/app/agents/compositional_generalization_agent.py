"""Compositional Generalization Engine — Phase 41.

Remixes known playbooks to solve novel problems the agent has never seen
exactly. Given a novel goal, it decomposes it into subtasks, maps each
subtask to the closest step from existing playbooks, and assembles a new
composed procedure. Steps that cannot be sourced from any playbook are
flagged as novel steps that require human authoring or LLM synthesis.

Example: "Deploy to on-prem?" → adapts AWS playbook using abstract
procedure primitives that survive environment substitution.

Depends on:
  - PlaybookAgent (Phase 20)               — playbook_candidates catalog
  - PlaybookExecutionTrackerAgent (Phase 33) — historical step coverage
  - GoalDecompositionAgent (Phase 21)       — subtasks from goal

Inputs (via context["memory"]):
  - content:   raw text describing the novel goal or task

Inputs (via context["memory"]["enrichment"]):
  - domain:              str  — normalized domain (Phase 6)
  - subtasks:            list[str] — goal subtasks (Phase 21)
  - playbook_candidates: list[dict] — pre-fetched playbooks
                         each: {id, name, domain, steps: [str],
                                success_rate: float, occurrence_count: int}

Outputs:
  - composed_procedure:     list[str]  — ordered steps of the composed plan
  - source_playbooks:       list[str]  — playbook IDs used as source material
  - adaptation_confidence:  float 0..1 — fraction of steps successfully sourced
  - novel_steps:            list[str]  — steps not found in any existing playbook
  - confidence:             float
  - rationale:              "heuristic" | "llm"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# Minimum token overlap fraction to consider a playbook step a usable source.
_STEP_MATCH_THRESHOLD = 0.25

# Stop words excluded from token extraction.
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
# Pure helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Extract lowercase word tokens (3+ chars) minus stop words."""
    if not text:
        return set()
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _step_score(step_text: str, query_tokens: set[str]) -> float:
    """Jaccard-like overlap: intersection / union of tokens.

    Returns 0.0 when either side is empty.
    """
    if not step_text or not query_tokens:
        return 0.0
    step_tokens = _tokenize(step_text)
    if not step_tokens:
        return 0.0
    intersection = query_tokens & step_tokens
    union = query_tokens | step_tokens
    return round(len(intersection) / len(union), 4)


def find_best_source_step(
    subtask: str,
    playbook_candidates: list[dict[str, Any]],
) -> tuple[str | None, str | None, float]:
    """Find the best matching step across all candidates for a subtask.

    Returns:
        (playbook_id, step_text, score)
        or (None, None, 0.0) when no step exceeds the threshold.
    """
    query_tokens = _tokenize(subtask)
    if not query_tokens:
        return None, None, 0.0

    best_pb_id: str | None = None
    best_step: str | None = None
    best_score = 0.0

    for candidate in playbook_candidates:
        if not isinstance(candidate, dict):
            continue
        steps = candidate.get("steps") or []
        pb_id = str(candidate.get("id") or "")
        success_rate = float(candidate.get("success_rate") or 0.5)

        for step in steps:
            raw_score = _step_score(str(step), query_tokens)
            if raw_score == 0.0:
                continue
            # Weight by playbook success rate so reliable playbooks are preferred.
            weighted = round(raw_score * (0.7 + 0.3 * success_rate), 4)
            if weighted > best_score:
                best_score = weighted
                best_pb_id = pb_id
                best_step = str(step)

    if best_score < _STEP_MATCH_THRESHOLD:
        return None, None, 0.0

    return best_pb_id, best_step, best_score


def compose_procedure(
    subtasks: list[str],
    playbook_candidates: list[dict[str, Any]],
    content_tokens: set[str],
) -> tuple[list[str], list[str], list[str], float]:
    """Assemble a composed procedure from subtasks and playbook candidates.

    Returns:
        (composed_procedure, source_playbook_ids, novel_steps, adaptation_confidence)
    """
    if not subtasks:
        return [], [], [], 0.0

    composed: list[str] = []
    source_set: list[str] = []
    novel: list[str] = []
    sourced_count = 0

    for subtask in subtasks:
        pb_id, step_text, score = find_best_source_step(subtask, playbook_candidates)
        if pb_id is not None and step_text is not None:
            # Prefer the exact step text from the playbook; the subtask provides
            # intent but the playbook step is already validated.
            composed.append(step_text)
            if pb_id not in source_set:
                source_set.append(pb_id)
            sourced_count += 1
        else:
            # No existing playbook covers this subtask — it's novel.
            composed.append(subtask)
            novel.append(subtask)

    total = len(subtasks)
    adaptation_confidence = round(sourced_count / total, 4) if total > 0 else 0.0

    return composed, source_set, novel, adaptation_confidence


def compute_agent_confidence(
    *,
    adaptation_confidence: float,
    novel_step_count: int,
    total_steps: int,
    candidate_count: int,
) -> float:
    """Overall agent confidence in the composed procedure."""
    if total_steps == 0 or candidate_count == 0:
        return 0.40
    if adaptation_confidence >= 0.80:
        return 0.85
    if adaptation_confidence >= 0.60:
        return 0.75
    if adaptation_confidence >= 0.40:
        return 0.65
    if adaptation_confidence >= 0.20:
        return 0.55
    return 0.45


def _extract_subtasks_from_content(content: str) -> list[str]:
    """Fallback: split content into sentence-like chunks as rough subtasks."""
    if not content:
        return []
    sentences = re.split(r"[.!?;]\s+", content.strip())
    result = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10:
            result.append(s[:200])
    return result[:10]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CompositionalGeneralizationAgent(BaseAgent):
    """Phase 41: Remix known playbooks to solve novel goals.

    Decomposes the incoming goal into subtasks (using GoalDecompositionAgent
    output or content-based extraction), maps each subtask to the closest step
    from any known playbook, and assembles a composed procedure. Steps with no
    playbook match are returned as novel_steps for human or LLM authoring.
    """

    name = "CompositionalGeneralizationAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["PlaybookAgent", "PlaybookExecutionTrackerAgent", "GoalDecompositionAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        composed = outputs.get("composed_procedure")
        if not isinstance(composed, list):
            raise ValueError("composed_procedure must be a list")

        sources = outputs.get("source_playbooks")
        if not isinstance(sources, list):
            raise ValueError("source_playbooks must be a list")

        adapt_conf = outputs.get("adaptation_confidence")
        if not isinstance(adapt_conf, (int, float)) or not (0.0 <= float(adapt_conf) <= 1.0):
            raise ValueError("adaptation_confidence must be a float in [0, 1]")

        novel = outputs.get("novel_steps")
        if not isinstance(novel, list):
            raise ValueError("novel_steps must be a list")

        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    def _heuristic(
        self,
        *,
        content: str,
        subtasks: list[str],
        playbook_candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Prefer GoalDecompositionAgent subtasks; fall back to sentence-split.
        effective_subtasks = subtasks if subtasks else _extract_subtasks_from_content(content)

        content_tokens = _tokenize(content)

        composed, source_pbs, novel, adapt_conf = compose_procedure(
            subtasks=effective_subtasks,
            playbook_candidates=playbook_candidates,
            content_tokens=content_tokens,
        )

        confidence = compute_agent_confidence(
            adaptation_confidence=adapt_conf,
            novel_step_count=len(novel),
            total_steps=len(composed),
            candidate_count=len(playbook_candidates),
        )

        return {
            "composed_procedure": composed,
            "source_playbooks": source_pbs,
            "adaptation_confidence": adapt_conf,
            "novel_steps": novel,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        subtasks = list(enrichment.get("subtasks") or [])
        playbook_candidates = list(enrichment.get("playbook_candidates") or [])

        strategy = getattr(settings, "COMPOSITIONAL_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                subtasks=subtasks,
                playbook_candidates=playbook_candidates,
            )
        else:
            prompt = (
                "You are a compositional generalization engine for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given a novel goal and a set of known playbooks, compose a new procedure "
                "by remixing steps from the playbooks. Steps that cannot be sourced from "
                "any playbook should be flagged as novel_steps.\n\n"
                f"GOAL: {content[:400]}\n"
                f"SUBTASKS: {subtasks[:10]}\n"
                f"PLAYBOOK_CANDIDATES: {playbook_candidates[:5]}\n\n"
                "Return JSON with keys:\n"
                "- composed_procedure: list[str] — ordered steps\n"
                "- source_playbooks: list[str] — playbook IDs used\n"
                "- adaptation_confidence: float 0..1\n"
                "- novel_steps: list[str] — steps not from any playbook\n"
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
                and isinstance(resp.get("composed_procedure"), list)
                and isinstance(resp.get("adaptation_confidence"), (int, float))
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    subtasks=subtasks,
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
