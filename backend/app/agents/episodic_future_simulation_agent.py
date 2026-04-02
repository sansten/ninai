"""Episodic future simulation agent (Phase 67)."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.llm.ollama_breaker import create_ollama_client
from app.agents.types import AgentContext, AgentResult
from app.core.config import settings


_WORD_RE = re.compile(r"\b[a-z0-9_]+\b", re.IGNORECASE)
_ESCALATION_TOKENS = {"critical", "alert", "failure", "down", "error"}


def _tokenize(text: str) -> set[str]:
    return set(t.lower() for t in _WORD_RE.findall(text or ""))


def _episode_text(episode: dict[str, Any]) -> str:
    parts = [
        str(episode.get("content") or ""),
        str(episode.get("summary") or ""),
        str(episode.get("event_description") or ""),
    ]
    tags = episode.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    parts.extend(str(t) for t in tags)
    return " ".join(parts)


def _overlap_score(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dominant_tag(episode: dict[str, Any]) -> str:
    tags = episode.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    cleaned = [str(t).strip().lower() for t in tags if str(t).strip()]
    return cleaned[0] if cleaned else "untagged"


def _severity_change(text: str) -> str:
    tokens = _tokenize(text)
    if tokens & _ESCALATION_TOKENS:
        return "increase"
    if any(t in tokens for t in {"resolved", "stable", "recovered", "fixed", "mitigated"}):
        return "decrease"
    return "stable"


def _extract_entities(current_state: dict[str, Any], template: dict[str, Any]) -> list[str]:
    entities = current_state.get("entities") or []
    if isinstance(entities, str):
        entities = [entities]
    entities = [str(e).strip() for e in entities if str(e).strip()]

    if entities:
        return entities[:5]

    tags = template.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    tag_entities = [str(t).strip() for t in tags if str(t).strip()]
    return tag_entities[:5]


def run_heuristic(
    *,
    current_state: dict[str, Any],
    planned_action: str,
    historical_episodes: list[dict[str, Any]],
    simulation_steps: int = 3,
) -> dict[str, Any]:
    action = str(planned_action or "").strip() or "Perform planned action"
    steps = max(1, int(simulation_steps if simulation_steps is not None else 3))
    history = list(historical_episodes or [])

    action_tokens = _tokenize(action)

    matching_indices: list[int] = []
    for i, ep in enumerate(history):
        score = _overlap_score(action_tokens, _tokenize(_episode_text(ep)))
        if score >= 0.2:
            matching_indices.append(i)

    sequel_templates: list[dict[str, Any]] = []
    for idx in matching_indices:
        nxt = idx + 1
        if nxt < len(history):
            sequel_templates.append(history[nxt])

    deduped_templates: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    for t in sequel_templates:
        key = _dominant_tag(t)
        if key in seen_tags:
            continue
        seen_tags.add(key)
        deduped_templates.append(t)

    simulated_episodes: list[dict[str, Any]] = []

    step0_entities = _extract_entities(current_state, {})
    simulated_episodes.append(
        {
            "step": 0,
            "event_description": action,
            "probability": 0.9,
            "severity_change": "stable",
            "entities_affected": step0_entities,
        }
    )

    base_prob = 0.75
    for step in range(1, steps):
        if deduped_templates:
            template = deduped_templates[(step - 1) % len(deduped_templates)]
            desc = str(
                template.get("event_description")
                or template.get("content")
                or template.get("summary")
                or f"Follow-up event after action: {action}"
            )
            entities = _extract_entities(current_state, template)
        else:
            template = {}
            desc = f"Likely follow-up step {step} after: {action}"
            entities = _extract_entities(current_state, template)

        probability = round(max(0.01, base_prob * (0.8 ** step)), 4)
        simulated_episodes.append(
            {
                "step": step,
                "event_description": desc,
                "probability": probability,
                "severity_change": _severity_change(desc),
                "entities_affected": entities,
            }
        )

    risk_events = [
        ep
        for ep in simulated_episodes
        if ep.get("severity_change") == "increase" and float(ep.get("probability") or 0.0) > 0.5
    ]

    max_risk_probability = max((float(ep.get("probability") or 0.0) for ep in risk_events), default=0.0)
    success_probability = round(max(0.0, min(1.0, 0.9 * (1.0 - max_risk_probability))), 4)

    recommended_precautions: list[str] = []
    for ep in risk_events:
        entities_text = ", ".join(ep.get("entities_affected") or []) or "key systems"
        desc = str(ep.get("event_description") or "").strip()
        recommended_precautions.append(f"Monitor {entities_text} closely before proceeding.")
        recommended_precautions.append(f"Roll back if {desc[:50]} occurs.")

    # Keep deterministic order while removing duplicates.
    seen_precautions: Counter[str] = Counter()
    deduped_precautions: list[str] = []
    for p in recommended_precautions:
        if seen_precautions[p] > 0:
            continue
        seen_precautions[p] += 1
        deduped_precautions.append(p)

    return {
        "simulated_episodes": simulated_episodes,
        "success_probability": success_probability,
        "risk_events": risk_events,
        "recommended_precautions": deduped_precautions,
        "confidence": 0.6,
        "rationale": "heuristic",
    }


class EpisodicFutureSimulationAgent(BaseAgent):
    name = "EpisodicFutureSimulationAgent"
    version = "v1"

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        if not isinstance(outputs.get("simulated_episodes"), list):
            raise ValueError("simulated_episodes must be a list")
        prob = outputs.get("success_probability")
        if not isinstance(prob, (int, float)) or not (0.0 <= float(prob) <= 1.0):
            raise ValueError("success_probability must be float between 0 and 1")
        if not isinstance(outputs.get("risk_events"), list):
            raise ValueError("risk_events must be a list")
        if not isinstance(outputs.get("recommended_precautions"), list):
            raise ValueError("recommended_precautions must be a list")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        enrichment = (context.get("memory") or {}).get("enrichment") or {}

        current_state = dict(enrichment.get("current_state") or {})
        planned_action = str(enrichment.get("planned_action") or "")
        historical_episodes = list(enrichment.get("historical_episodes") or [])
        simulation_steps = int(enrichment.get("simulation_steps") or 3)

        strategy = getattr(settings, "EPISODIC_FUTURE_SIMULATION_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        if strategy == "heuristic":
            outputs = run_heuristic(
                current_state=current_state,
                planned_action=planned_action,
                historical_episodes=historical_episodes,
                simulation_steps=simulation_steps,
            )
        else:
            prompt = (
                "You simulate likely future episode sequences from an action plan. Output JSON only.\n\n"
                f"CURRENT_STATE: {current_state}\n"
                f"PLANNED_ACTION: {planned_action}\n"
                f"HISTORICAL_EPISODES: {historical_episodes[:80]}\n"
                f"SIMULATION_STEPS: {simulation_steps}\n\n"
                "Return JSON with keys:\n"
                "- simulated_episodes: list[{step, event_description, probability, severity_change, entities_affected}]\n"
                "- success_probability: float\n"
                "- risk_events: list[dict]\n"
                "- recommended_precautions: list[str]\n"
                "- confidence: float\n"
                "- rationale: str"
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
                and isinstance(resp.get("simulated_episodes"), list)
                and isinstance(resp.get("success_probability"), (int, float))
                and isinstance(resp.get("risk_events"), list)
                and isinstance(resp.get("recommended_precautions"), list)
            ):
                outputs = resp
            else:
                outputs = run_heuristic(
                    current_state=current_state,
                    planned_action=planned_action,
                    historical_episodes=historical_episodes,
                    simulation_steps=simulation_steps,
                )

        finished_at = datetime.now(timezone.utc)
        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence") or 0.5),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=trace_id,
        )
        self.validate_outputs(result)
        return result
