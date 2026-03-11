"""Theory of Mind & Multi-Agent Modeling Agent — Phase 38.

Builds per-user and per-agent belief models: infers what users already know,
estimates peer agent capabilities, and adjusts collaboration tone accordingly.

Depends on:
  - CredibilityAgent          (Phase 19) — credibility_scores in enrichment
  - OrgAttentionAgent         (Phase 10) — attention_signals in enrichment
  - FeedbackIntegrationAgent  (Phase 24) — feedback_signals in enrichment

Inputs (via context["memory"]["enrichment"]):
  - user_interactions:   list[dict]       — [{query, domain, word_count, timestamp}]
  - feedback_signals:    list[dict]       — [{sentiment, category, domain}]
  - peer_agent_results:  dict[str, list]  — agent_name → [{status, confidence, domain}]
  - attention_signals:   dict             — {focus_domain, attention_score}
  - credibility_scores:  dict[str, float] — entity/source → credibility float

Outputs:
  - user_model:          dict  — {knowledge_level, inferred_expertise,
                                   preferred_detail_level, interaction_count}
  - peer_agent_models:   list[dict] — [{agent_name, reliability_score,
                                         strength_domains, weakness_domains, confidence}]
  - inferred_intent:     str  — most likely user goal domain
  - tone_adjustment:     str  — "technical" | "simplified" | "neutral"
  - confidence:          float
  - rationale:           str  — "heuristic" | "llm"
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

_KNOWLEDGE_LEVELS = frozenset({"novice", "intermediate", "expert"})
_TONE_ADJUSTMENTS = frozenset({"technical", "simplified", "neutral"})
_DETAIL_LEVELS = frozenset({"brief", "standard", "detailed"})

_EXPERT_WORD_THRESHOLD = 15        # avg words/query to qualify as expert
_EXPERT_INTERACTION_THRESHOLD = 5  # min interactions to qualify as expert
_INTERMEDIATE_WORD_THRESHOLD = 8   # avg words/query for intermediate
_INTERMEDIATE_INTERACTION_THRESHOLD = 2
_HIGH_CONFIDENCE_THRESHOLD = 0.75  # peer domain counted as "strength"
_LOW_CONFIDENCE_THRESHOLD = 0.40   # peer domain counted as "weakness"
_POSITIVE_FEEDBACK_THRESHOLD = 3   # positive signals needed to prefer detailed output
_MAX_EXPERTISE_DOMAINS = 5
_MAX_PEER_MODELS = 20


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _infer_knowledge_level(avg_word_count: float, interaction_count: int) -> str:
    if (avg_word_count >= _EXPERT_WORD_THRESHOLD
            and interaction_count >= _EXPERT_INTERACTION_THRESHOLD):
        return "expert"
    if (avg_word_count >= _INTERMEDIATE_WORD_THRESHOLD
            or interaction_count >= _INTERMEDIATE_INTERACTION_THRESHOLD):
        return "intermediate"
    return "novice"


def _tone_from_level(knowledge_level: str) -> str:
    if knowledge_level == "expert":
        return "technical"
    if knowledge_level == "novice":
        return "simplified"
    return "neutral"


def _detail_from_level(knowledge_level: str, positive_feedback: int) -> str:
    if knowledge_level == "expert" or positive_feedback >= _POSITIVE_FEEDBACK_THRESHOLD:
        return "detailed"
    if knowledge_level == "novice":
        return "brief"
    return "standard"


def _build_user_model(
    user_interactions: list[dict],
    feedback_signals: list[dict],
) -> dict:
    """Derive a mental model of the user from interaction history and feedback."""
    interaction_count = len(user_interactions)

    if interaction_count == 0:
        return {
            "knowledge_level": "novice",
            "inferred_expertise": [],
            "preferred_detail_level": "standard",
            "interaction_count": 0,
        }

    word_counts: list[int] = []
    domain_counts: dict[str, int] = {}

    for entry in user_interactions:
        wc = entry.get("word_count")
        if wc is None:
            query = str(entry.get("query") or "")
            wc = len(query.split())
        word_counts.append(int(wc))

        domain = str(entry.get("domain") or "").strip()
        if domain and domain.lower() != "unknown":
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    avg_word_count = sum(word_counts) / len(word_counts)
    knowledge_level = _infer_knowledge_level(avg_word_count, interaction_count)

    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    inferred_expertise = [d for d, _ in sorted_domains[:_MAX_EXPERTISE_DOMAINS]]

    positive_feedback = sum(
        1 for f in feedback_signals
        if str(f.get("sentiment") or "").lower() in {"positive", "helpful", "good"}
    )
    preferred_detail_level = _detail_from_level(knowledge_level, positive_feedback)

    return {
        "knowledge_level": knowledge_level,
        "inferred_expertise": inferred_expertise,
        "preferred_detail_level": preferred_detail_level,
        "interaction_count": interaction_count,
    }


def _build_peer_agent_models(peer_agent_results: dict[str, list[dict]]) -> list[dict]:
    """Build capability models for each peer agent from their result history."""
    models: list[dict] = []

    for agent_name, results in list(peer_agent_results.items())[:_MAX_PEER_MODELS]:
        if not isinstance(results, list) or not results:
            continue

        success_count = sum(
            1 for r in results if isinstance(r, dict) and str(r.get("status") or "") == "success"
        )
        valid_results = [r for r in results if isinstance(r, dict)]
        if not valid_results:
            continue

        reliability_score = round(success_count / len(valid_results), 3)

        strength_domains: list[str] = []
        weakness_domains: list[str] = []
        conf_total = 0.0

        for r in valid_results:
            domain = str(r.get("domain") or "").strip()
            conf = float(r.get("confidence") or 0.5)
            conf_total += conf
            if domain:
                if conf >= _HIGH_CONFIDENCE_THRESHOLD and domain not in strength_domains:
                    strength_domains.append(domain)
                elif conf < _LOW_CONFIDENCE_THRESHOLD and domain not in weakness_domains:
                    weakness_domains.append(domain)

        avg_conf = round(conf_total / len(valid_results), 3)

        models.append({
            "agent_name": agent_name,
            "reliability_score": reliability_score,
            "strength_domains": strength_domains,
            "weakness_domains": weakness_domains,
            "confidence": avg_conf,
        })

    return models


def _infer_intent(user_interactions: list[dict], attention_signals: dict) -> str:
    """Return the most likely user intent domain."""
    if user_interactions:
        domain_counts: dict[str, int] = {}
        for entry in user_interactions:
            domain = str(entry.get("domain") or "").strip()
            if domain and domain.lower() != "unknown":
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if domain_counts:
            return max(domain_counts, key=lambda k: domain_counts[k])

    focus = str(attention_signals.get("focus_domain") or "").strip()
    if focus:
        return focus

    return "general"


async def run_heuristic_async(
    user_interactions: list[dict],
    feedback_signals: list[dict],
    peer_agent_results: dict[str, list[dict]],
    attention_signals: dict,
) -> dict[str, Any]:
    """Heuristic path — no LLM required."""
    user_model = _build_user_model(user_interactions, feedback_signals)
    peer_models = _build_peer_agent_models(peer_agent_results)
    inferred_intent = _infer_intent(user_interactions, attention_signals)
    tone_adjustment = _tone_from_level(user_model["knowledge_level"])

    if peer_models:
        avg_conf = round(sum(m["confidence"] for m in peer_models) / len(peer_models), 3)
    else:
        avg_conf = 0.5

    return {
        "user_model": user_model,
        "peer_agent_models": peer_models,
        "inferred_intent": inferred_intent,
        "tone_adjustment": tone_adjustment,
        "confidence": avg_conf,
        "rationale": "heuristic",
    }


def _safe_default() -> dict[str, Any]:
    return {
        "user_model": {
            "knowledge_level": "novice",
            "inferred_expertise": [],
            "preferred_detail_level": "standard",
            "interaction_count": 0,
        },
        "peer_agent_models": [],
        "inferred_intent": "general",
        "tone_adjustment": "neutral",
        "confidence": 0.5,
        "rationale": "heuristic",
    }


def _parse_llm_output(resp: Any) -> dict[str, Any] | None:
    """Validate and normalise LLM JSON; return None if unusable."""
    if not isinstance(resp, dict):
        return None

    user_model = resp.get("user_model")
    if not isinstance(user_model, dict):
        return None

    knowledge_level = user_model.get("knowledge_level")
    if knowledge_level not in _KNOWLEDGE_LEVELS:
        return None

    tone_adjustment = resp.get("tone_adjustment")
    if tone_adjustment not in _TONE_ADJUSTMENTS:
        return None

    confidence = resp.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        return None

    peer_models_raw = resp.get("peer_agent_models")
    if not isinstance(peer_models_raw, list):
        return None

    inferred_intent = resp.get("inferred_intent")
    if not isinstance(inferred_intent, str) or not inferred_intent.strip():
        return None

    # Normalise user_model
    detail_level = str(user_model.get("preferred_detail_level") or "standard")
    if detail_level not in _DETAIL_LEVELS:
        detail_level = "standard"

    normalised_user_model = {
        "knowledge_level": str(knowledge_level),
        "inferred_expertise": list(user_model.get("inferred_expertise") or []),
        "preferred_detail_level": detail_level,
        "interaction_count": int(user_model.get("interaction_count") or 0),
    }

    # Normalise peer models — skip entries with no agent_name
    peer_models: list[dict] = []
    for pm in peer_models_raw:
        if not isinstance(pm, dict):
            continue
        agent_name = pm.get("agent_name")
        if not agent_name:
            continue
        peer_models.append({
            "agent_name": str(agent_name),
            "reliability_score": round(float(pm.get("reliability_score") or 0.5), 3),
            "strength_domains": list(pm.get("strength_domains") or []),
            "weakness_domains": list(pm.get("weakness_domains") or []),
            "confidence": round(float(pm.get("confidence") or 0.5), 3),
        })

    return {
        "user_model": normalised_user_model,
        "peer_agent_models": peer_models,
        "inferred_intent": str(inferred_intent).strip(),
        "tone_adjustment": str(tone_adjustment),
        "confidence": round(float(confidence), 3),
        "rationale": "llm",
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TheoryOfMindAgent(BaseAgent):
    """Phase 38: Build mental models of users and peer agents to improve collaboration.

    Estimates what a user already knows, identifies peer agent strengths and weaknesses,
    infers the user's underlying intent, and selects an appropriate response tone.
    """

    name = "TheoryOfMindAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return [
            "CredibilityAgent",
            "OrgAttentionAgent",
            "FeedbackIntegrationAgent",
        ]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        user_model = outputs.get("user_model")
        if not isinstance(user_model, dict):
            raise ValueError("user_model must be a dict")

        knowledge_level = user_model.get("knowledge_level")
        if knowledge_level not in _KNOWLEDGE_LEVELS:
            raise ValueError(
                f"knowledge_level must be one of {_KNOWLEDGE_LEVELS}, got {knowledge_level!r}"
            )

        peer_models = outputs.get("peer_agent_models")
        if not isinstance(peer_models, list):
            raise ValueError("peer_agent_models must be a list")

        inferred_intent = outputs.get("inferred_intent")
        if not isinstance(inferred_intent, str) or not inferred_intent:
            raise ValueError("inferred_intent must be a non-empty string")

        tone = outputs.get("tone_adjustment")
        if tone not in _TONE_ADJUSTMENTS:
            raise ValueError(
                f"tone_adjustment must be one of {_TONE_ADJUSTMENTS}, got {tone!r}"
            )

        confidence = outputs.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")

        enrichment: dict[str, Any] = (context.get("memory") or {}).get("enrichment") or {}

        user_interactions: list[dict] = enrichment.get("user_interactions") or []
        feedback_signals: list[dict] = enrichment.get("feedback_signals") or []
        peer_agent_results: dict[str, list[dict]] = enrichment.get("peer_agent_results") or {}
        attention_signals: dict = enrichment.get("attention_signals") or {}

        strategy = str(getattr(settings, "AGENT_STRATEGY", "llm") or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic":
            outputs = await run_heuristic_async(
                user_interactions, feedback_signals, peer_agent_results, attention_signals
            )
        else:
            prompt = (
                "You are a theory-of-mind modeling engine for an enterprise AI OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Given the signals below, build belief models of the user and peer agents.\n\n"
                f"USER_INTERACTIONS: {user_interactions}\n"
                f"FEEDBACK_SIGNALS: {feedback_signals}\n"
                f"PEER_AGENT_RESULTS: {peer_agent_results}\n"
                f"ATTENTION_SIGNALS: {attention_signals}\n\n"
                "Return JSON with keys:\n"
                "  user_model: dict with:\n"
                "    knowledge_level        str  (novice|intermediate|expert)\n"
                "    inferred_expertise     list[str]  (top domains)\n"
                "    preferred_detail_level str  (brief|standard|detailed)\n"
                "    interaction_count      int\n"
                "  peer_agent_models: list of dicts, each with:\n"
                "    agent_name             str\n"
                "    reliability_score      float (0-1)\n"
                "    strength_domains       list[str]\n"
                "    weakness_domains       list[str]\n"
                "    confidence             float (0-1)\n"
                "  inferred_intent:   str   (most likely user goal domain)\n"
                "  tone_adjustment:   str   (technical|simplified|neutral)\n"
                "  confidence:        float (0-1)\n"
            )
            try:
                client = create_ollama_client(
                    base_url=str(getattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")),
                    model=str(getattr(settings, "OLLAMA_MODEL", "qwen2.5:7b")),
                    timeout_seconds=float(getattr(settings, "OLLAMA_TIMEOUT_SECONDS", 5.0)),
                    max_concurrency=int(getattr(settings, "OLLAMA_MAX_CONCURRENCY", 2)),
                )
                resp = await client.complete_json(
                    prompt=prompt,
                    schema_hint={},
                    tool_event_sink=context.get("tool_event_sink"),
                )
                parsed = _parse_llm_output(resp)
            except Exception:
                parsed = None

            if parsed is not None:
                outputs = parsed
            else:
                outputs = await run_heuristic_async(
                    user_interactions, feedback_signals, peer_agent_results, attention_signals
                )

        finished_at = datetime.now(timezone.utc)

        result = AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=float(outputs.get("confidence", 0.5)),
            outputs=outputs,
            warnings=[],
            errors=[],
            started_at=started_at,
            finished_at=finished_at,
            trace_id=str(trace_id) if trace_id else None,
        )
        self.validate_outputs(result)
        return result
