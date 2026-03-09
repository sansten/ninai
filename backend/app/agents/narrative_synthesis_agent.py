"""Narrative Synthesis — Phase 23.

Reads fully-enriched memory objects and synthesises a human-readable
narrative for expert briefings, meeting prep, and onboarding.  Consumes
outputs written by all prior agents and assembles them into a coherent,
prioritised summary a human expert can act on.

Depends on (reads enrichment from):
  - EntityResolutionAgent (Phase 7)       — resolved_entities, canonical_entity
  - TemporalReasoningAgent (Phase 17)     — temporal_anchor, sequence_position
  - EpisodicGroupingAgent (Phase 18)      — episode_id, episode_label
  - PlaybookAgent (Phase 20)              — matched_playbook_id, playbook_confidence
  - GoalDecompositionAgent (Phase 21)     — goal_detected, subtasks, blocking_subtask,
                                            completion_fraction
  - UncertaintyReportingAgent (Phase 22)  — uncertainty_level, review_recommended,
                                            unresolved_conflicts
  - CredibilityAgent (Phase 19)           — credibility_score, source_tier
  - ConflictDetectionAgent (Phase 13)     — conflict_count
  - AdaptiveConflictResolutionAgent (Ph14)— escalation_targets

Outputs:
  - narrative_text:   str               — 2–5 sentence human-readable summary
  - key_entities:     list[str]         — primary entities mentioned in the narrative
  - timeline_summary: str | None        — temporal context when available
  - action_items:     list[str]         — concrete follow-ups for a human expert
  - tone:             "informational" | "cautionary" | "urgent"
  - confidence:       float
  - rationale:        "heuristic" | "llm"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_TONES = {"informational", "cautionary", "urgent"}
_LOW_CREDIBILITY_THRESHOLD = 0.40   # below this → credibility is critically low
_PLAYBOOK_CONFIDENCE_MIN = 0.55     # below this → don't mention the playbook match
_URGENT_CONFLICT_COUNT = 3          # at or above → tone is urgent


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _is_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def extract_key_entities(enrichment: dict[str, Any]) -> list[str]:
    """Collect the most prominent entity names from prior agent outputs.

    Prefers the canonical entity from Phase 7, then falls back to the
    resolved_entities list.  Returns at most 5 names.
    """
    entities: list[str] = []

    canonical = enrichment.get("canonical_entity")
    if canonical and isinstance(canonical, str):
        entities.append(canonical)

    resolved = enrichment.get("resolved_entities") or []
    if isinstance(resolved, list):
        for e in resolved:
            if isinstance(e, dict):
                name = (
                    e.get("name")
                    or e.get("canonical_name")
                    or e.get("entity")
                )
                if name and isinstance(name, str) and name not in entities:
                    entities.append(name)
            elif isinstance(e, str) and e not in entities:
                entities.append(e)

    return entities[:5]


def build_timeline_summary(enrichment: dict[str, Any]) -> str | None:
    """Build a short temporal context string if temporal signals are present."""
    anchor = enrichment.get("temporal_anchor")
    seq = enrichment.get("sequence_position")

    if not anchor and seq is None:
        return None

    parts: list[str] = []
    if anchor:
        parts.append(f"anchored at {anchor}")
    if seq is not None:
        parts.append(f"sequence position {seq}")

    return "Temporal context: " + ", ".join(parts) + "."


def collect_action_items(enrichment: dict[str, Any]) -> list[str]:
    """Extract concrete follow-up actions from enrichment signals."""
    items: list[str] = []

    if enrichment.get("review_recommended"):
        items.append("Human review recommended — uncertainty level is elevated.")

    blocking = enrichment.get("blocking_subtask")
    if blocking and isinstance(blocking, str):
        items.append(f"Unblock subtask: {blocking}")

    escalation: list = list(enrichment.get("escalation_targets") or [])
    if escalation:
        targets = ", ".join(str(t) for t in escalation[:3])
        items.append(f"Escalate unresolved conflict to: {targets}")

    cred = enrichment.get("credibility_score")
    if cred is not None and _is_float(cred) and float(cred) < _LOW_CREDIBILITY_THRESHOLD:
        items.append("Verify source credibility — score is critically low.")

    return items


def determine_tone(enrichment: dict[str, Any]) -> str:
    """Choose narrative tone based on uncertainty and conflict signals."""
    level = str(enrichment.get("uncertainty_level") or "").lower()
    conflict_count = int(enrichment.get("conflict_count") or 0)

    if level == "critical" or conflict_count >= _URGENT_CONFLICT_COUNT:
        return "urgent"

    if (
        level in {"high", "medium"}
        or conflict_count >= 1
        or enrichment.get("review_recommended")
    ):
        return "cautionary"

    return "informational"


def build_narrative_text(
    enrichment: dict[str, Any],
    key_entities: list[str],
    timeline_summary: str | None,
    tone: str,
) -> str:
    """Assemble a 2–5 sentence narrative from enrichment data."""
    sentences: list[str] = []

    # Opening — subject of this memory
    episode_label = enrichment.get("episode_label")
    canonical = enrichment.get("canonical_entity")
    subject = episode_label or canonical or (key_entities[0] if key_entities else None)

    if subject:
        sentences.append(f"This memory concerns {subject}.")
    else:
        sentences.append("This memory captures an enterprise event.")

    # Temporal context
    if timeline_summary:
        sentences.append(timeline_summary)

    # Goal progress
    if enrichment.get("goal_detected"):
        fraction = enrichment.get("completion_fraction")
        blocking = enrichment.get("blocking_subtask")
        if fraction is not None and _is_float(fraction):
            pct = int(float(fraction) * 100)
            goal_sent = f"A goal has been identified with {pct}% of subtasks complete."
            if blocking:
                goal_sent += f" The blocking subtask is: {blocking}."
            sentences.append(goal_sent)

    # Playbook match
    pb_id = enrichment.get("matched_playbook_id")
    pb_conf = enrichment.get("playbook_confidence")
    if (
        pb_id
        and pb_conf is not None
        and _is_float(pb_conf)
        and float(pb_conf) >= _PLAYBOOK_CONFIDENCE_MIN
    ):
        conf_pct = int(float(pb_conf) * 100)
        sentences.append(
            f"A matching playbook ({pb_id}) was found with {conf_pct}% confidence."
        )

    # Alert sentence
    conflict_count = int(enrichment.get("conflict_count") or 0)
    if tone == "urgent":
        if conflict_count:
            sentences.append(
                f"Urgent: {conflict_count} conflict(s) detected with critical"
                f" uncertainty — immediate review required."
            )
        else:
            sentences.append(
                "Uncertainty is at a critical level — immediate human review is required."
            )
    elif tone == "cautionary":
        if enrichment.get("review_recommended"):
            sentences.append(
                "Human review is recommended due to elevated uncertainty or unresolved conflicts."
            )
        elif conflict_count:
            sentences.append(
                f"Caution: {conflict_count} conflict(s) detected; verification is advised."
            )

    return " ".join(sentences)


def compute_narrative_confidence(
    enrichment: dict[str, Any],
    key_entities: list[str],
) -> float:
    """Confidence in the narrative — higher when enrichment is richer."""
    base = 0.60

    if enrichment.get("episode_label") or enrichment.get("canonical_entity"):
        base += 0.05
    if enrichment.get("temporal_anchor"):
        base += 0.05
    if enrichment.get("goal_detected"):
        base += 0.05
    if key_entities:
        base += 0.05
    if enrichment.get("uncertainty_level"):
        base += 0.05  # full uncertainty analysis gives narrative solid grounding

    if not key_entities and not enrichment.get("episode_label"):
        base -= 0.10

    return round(min(0.90, max(0.40, base)), 4)


def run_heuristic(enrichment: dict[str, Any]) -> dict[str, Any]:
    """Full heuristic narrative synthesis without LLM."""
    key_entities = extract_key_entities(enrichment)
    timeline_summary = build_timeline_summary(enrichment)
    action_items = collect_action_items(enrichment)
    tone = determine_tone(enrichment)
    narrative_text = build_narrative_text(enrichment, key_entities, timeline_summary, tone)
    confidence = compute_narrative_confidence(enrichment, key_entities)

    return {
        "narrative_text": narrative_text,
        "key_entities": key_entities,
        "timeline_summary": timeline_summary,
        "action_items": action_items,
        "tone": tone,
        "confidence": confidence,
        "rationale": "heuristic",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class NarrativeSynthesisAgent(BaseAgent):
    """Phase 23: Synthesise a human-readable narrative from enriched memory.

    Reads outputs from all prior agents and assembles a concise, prioritised
    summary including key entities, temporal context, goal status, playbook
    matches, and any action items a human expert needs to act on.
    """

    name = "NarrativeSynthesisAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "EntityResolutionAgent",
            "EpisodicGroupingAgent",
            "TemporalReasoningAgent",
            "GoalDecompositionAgent",
            "UncertaintyReportingAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        narrative = outputs.get("narrative_text")
        if not isinstance(narrative, str) or not narrative.strip():
            raise ValueError("narrative_text must be a non-empty string")

        if not isinstance(outputs.get("key_entities"), list):
            raise ValueError("key_entities must be a list")

        if not isinstance(outputs.get("action_items"), list):
            raise ValueError("action_items must be a list")

        tone = outputs.get("tone")
        if tone not in _VALID_TONES:
            raise ValueError(
                f"tone must be one of {_VALID_TONES}, got {tone!r}"
            )

        ts = outputs.get("timeline_summary")
        if ts is not None and not isinstance(ts, str):
            raise ValueError("timeline_summary must be a string or None")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment: dict[str, Any] = memory.get("enrichment") or {}

        strategy = getattr(settings, "NARRATIVE_SYNTHESIS_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = run_heuristic(enrichment)
        else:
            prompt = (
                "You are an enterprise memory narrative engine. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Synthesise a concise human-readable summary from the enrichment signals below. "
                "The summary should be 2–5 sentences, suitable for an expert briefing.\n\n"
                f"CANONICAL_ENTITY: {enrichment.get('canonical_entity')}\n"
                f"RESOLVED_ENTITIES: {enrichment.get('resolved_entities', [])}\n"
                f"EPISODE_LABEL: {enrichment.get('episode_label')}\n"
                f"TEMPORAL_ANCHOR: {enrichment.get('temporal_anchor')}\n"
                f"SEQUENCE_POSITION: {enrichment.get('sequence_position')}\n"
                f"GOAL_DETECTED: {enrichment.get('goal_detected')}\n"
                f"BLOCKING_SUBTASK: {enrichment.get('blocking_subtask')}\n"
                f"COMPLETION_FRACTION: {enrichment.get('completion_fraction')}\n"
                f"MATCHED_PLAYBOOK_ID: {enrichment.get('matched_playbook_id')}\n"
                f"PLAYBOOK_CONFIDENCE: {enrichment.get('playbook_confidence')}\n"
                f"UNCERTAINTY_LEVEL: {enrichment.get('uncertainty_level')}\n"
                f"REVIEW_RECOMMENDED: {enrichment.get('review_recommended')}\n"
                f"CONFLICT_COUNT: {enrichment.get('conflict_count')}\n"
                f"ESCALATION_TARGETS: {enrichment.get('escalation_targets', [])}\n"
                f"CREDIBILITY_SCORE: {enrichment.get('credibility_score')}\n\n"
                "Return JSON with keys:\n"
                "- narrative_text: 2–5 sentence summary string\n"
                "- key_entities: list of entity name strings\n"
                "- timeline_summary: string or null\n"
                "- action_items: list of action strings\n"
                "- tone: one of informational|cautionary|urgent\n"
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
                and isinstance(resp.get("narrative_text"), str)
                and resp.get("narrative_text", "").strip()
                and isinstance(resp.get("key_entities"), list)
                and isinstance(resp.get("action_items"), list)
                and resp.get("tone") in _VALID_TONES
            ):
                outputs = resp
            else:
                outputs = run_heuristic(enrichment)

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.60)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
