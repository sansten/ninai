"""Emotional & Affective Memory — Phase 42.

Tags memories with emotional valence, arousal level, empathy flags, and
escalation recommendations. Adapts downstream agent tone based on detected
emotional context so that high-stakes or distressed interactions receive
appropriately calibrated responses.

Builds on:
  - FeedbackIntegrationAgent (Phase 24)  — user sentiment signals
  - HumanReviewQueueAgent (Phase 32)     — escalation pathway
  - NarrativeSynthesisAgent (Phase 23)   — tone context

Inputs (via context["memory"]):
  - content:          raw text of the memory

Inputs (via context["memory"]["enrichment"]):
  - feedback_signals: list[dict] — each: {text, rating, sentiment}
  - narrative_tone:   str        — narrative tone label from Phase 23
  - urgency_score:    float 0..1 — from ProactiveMemoryPushAgent / Phase 11

Outputs:
  - emotional_valence:       "positive" | "negative" | "neutral" | "mixed"
  - arousal_level:           float 0..1   — intensity of emotional activation
  - empathy_flag:            bool         — True when distress/empathy cues detected
  - escalation_recommended:  bool         — True when human review is warranted
  - tone_guidance:           str          — suggested downstream tone
  - confidence:              float 0..1
  - rationale:               "heuristic" | "llm"
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
# Lexicons
# ---------------------------------------------------------------------------

_POSITIVE_WORDS: frozenset[str] = frozenset({
    "great", "excellent", "wonderful", "amazing", "fantastic", "happy",
    "pleased", "delighted", "success", "celebrate", "congratulations",
    "good", "love", "appreciate", "helpful", "thank", "thanks", "perfect",
    "brilliant", "outstanding", "resolved", "fixed", "improved", "better",
    "gain", "growth", "achieve", "accomplished", "effective", "efficient",
    "satisfied", "positive", "optimistic", "excited", "motivated",
})

_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "awful", "horrible", "failure", "failed", "broken",
    "error", "crash", "problem", "issue", "frustrated", "angry", "upset",
    "disappointed", "confused", "difficult", "impossible", "urgent", "panic",
    "crisis", "emergency", "blocked", "stuck", "wrong", "loss", "lost",
    "harm", "damage", "risk", "danger", "threat", "worry", "concern",
    "stressed", "overwhelmed", "helpless", "hopeless", "negative",
    "degraded", "outage", "incident", "critical", "severe",
})

_DISTRESS_WORDS: frozenset[str] = frozenset({
    "help", "please", "desperate", "cannot", "can't", "won't", "broken",
    "stuck", "emergency", "urgent", "panic", "crisis", "critical", "severe",
    "outage", "incident", "down", "failing", "catastrophic", "disaster",
    "blocked", "helpless", "hopeless", "overwhelmed", "frustrated",
})

_HIGH_AROUSAL_WORDS: frozenset[str] = frozenset({
    "urgent", "immediately", "now", "asap", "critical", "emergency",
    "panic", "crisis", "outage", "incident", "disaster", "catastrophic",
    "extreme", "severe", "alarm", "alert", "escalate", "breaking",
})

# Maps narrative tone → valence hint
_TONE_VALENCE_HINTS: dict[str, str] = {
    "positive": "positive",
    "encouraging": "positive",
    "celebratory": "positive",
    "negative": "negative",
    "critical": "negative",
    "alarming": "negative",
    "neutral": "neutral",
    "informative": "neutral",
    "mixed": "mixed",
}

# Tone guidance by valence/arousal profile
_TONE_GUIDANCE: dict[str, str] = {
    "positive_low": "affirming",
    "positive_high": "enthusiastic",
    "negative_low": "empathetic",
    "negative_high": "urgent_empathetic",
    "neutral_low": "neutral",
    "neutral_high": "attentive",
    "mixed_low": "balanced",
    "mixed_high": "cautious_empathetic",
}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens from text."""
    if not text:
        return []
    return re.findall(r"\b[a-z]{2,}\b", text.lower())


def score_valence(text: str) -> tuple[int, int]:
    """Return (positive_count, negative_count) for words in text."""
    tokens = set(_tokenize(text))
    pos = len(tokens & _POSITIVE_WORDS)
    neg = len(tokens & _NEGATIVE_WORDS)
    return pos, neg


def classify_valence(pos: int, neg: int) -> str:
    """Derive valence label from positive and negative word counts."""
    if pos == 0 and neg == 0:
        return "neutral"
    if pos > 0 and neg > 0:
        return "mixed"
    if pos > neg:
        return "positive"
    return "negative"


def compute_arousal(text: str, feedback_signals: list[dict[str, Any]]) -> float:
    """Estimate arousal (0..1) from high-arousal word density and feedback ratings.

    High-arousal words contribute 0.6 weight; low numeric ratings (1-2) from
    feedback_signals contribute 0.4 weight as a distress signal.
    """
    tokens = _tokenize(text)
    if not tokens:
        word_arousal = 0.0
    else:
        high_count = sum(1 for t in tokens if t in _HIGH_AROUSAL_WORDS)
        word_arousal = min(1.0, high_count / max(1, len(tokens)) * 10)

    # Negative feedback ratings signal arousal.
    distress_ratings = [
        s.get("rating", 3) for s in feedback_signals
        if isinstance(s, dict) and isinstance(s.get("rating"), (int, float))
        and float(s["rating"]) <= 2
    ]
    feedback_arousal = min(1.0, len(distress_ratings) / max(1, len(feedback_signals))) if feedback_signals else 0.0

    arousal = round(0.6 * word_arousal + 0.4 * feedback_arousal, 4)
    return min(1.0, arousal)


def check_empathy_flag(text: str, valence: str, arousal: float) -> bool:
    """Return True when content shows distress cues warranting empathetic response."""
    tokens = set(_tokenize(text))
    distress_hits = tokens & _DISTRESS_WORDS
    if distress_hits:
        return True
    if valence == "negative" and arousal >= 0.3:
        return True
    if valence == "mixed" and arousal >= 0.6:
        return True
    return False


def check_escalation(
    *,
    valence: str,
    arousal: float,
    empathy_flag: bool,
    urgency_score: float,
    feedback_signals: list[dict[str, Any]],
) -> bool:
    """Return True when the memory should be routed to the human review queue."""
    # Explicit urgency threshold.
    if urgency_score >= 0.8:
        return True
    # High-arousal negative content with empathy cues.
    if valence == "negative" and arousal >= 0.5 and empathy_flag:
        return True
    # Multiple low-rating feedback signals.
    low_ratings = [
        s for s in feedback_signals
        if isinstance(s, dict) and isinstance(s.get("rating"), (int, float))
        and float(s["rating"]) <= 2
    ]
    if len(low_ratings) >= 2:
        return True
    # Arousal above crisis threshold regardless of valence.
    if arousal >= 0.75:
        return True
    return False


def derive_tone_guidance(valence: str, arousal: float) -> str:
    """Pick a downstream tone label from valence and arousal bucket."""
    bucket = "high" if arousal >= 0.5 else "low"
    key = f"{valence}_{bucket}"
    return _TONE_GUIDANCE.get(key, "neutral")


def compute_confidence(
    *,
    content_len: int,
    pos: int,
    neg: int,
    feedback_count: int,
) -> float:
    """Confidence in the emotional assessment."""
    if content_len == 0:
        return 0.30
    signal_count = pos + neg + feedback_count
    if signal_count >= 5:
        return 0.85
    if signal_count >= 3:
        return 0.75
    if signal_count >= 1:
        return 0.65
    return 0.50


def _merge_valence_with_hint(text_valence: str, narrative_tone: str) -> str:
    """Optionally bias valence using the narrative tone label from Phase 23."""
    hint = _TONE_VALENCE_HINTS.get(str(narrative_tone or "").lower())
    if not hint or hint == text_valence:
        return text_valence
    # If hint and text agree on direction, keep text result.
    # If they conflict, fall through to "mixed".
    if text_valence == "neutral":
        return hint
    if hint == "neutral":
        return text_valence
    return "mixed"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EmotionalAffectiveMemoryAgent(BaseAgent):
    """Phase 42: Tag memories with emotional valence and drive escalation.

    Scores content for emotional valence, arousal intensity, distress
    empathy cues, and escalation necessity. Outputs a tone_guidance string
    that downstream agents (NarrativeSynthesis, HumanReviewQueue) can use
    to calibrate their responses.
    """

    name = "EmotionalAffectiveMemoryAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["FeedbackIntegrationAgent", "HumanReviewQueueAgent", "NarrativeSynthesisAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        valence = outputs.get("emotional_valence")
        if valence not in {"positive", "negative", "neutral", "mixed"}:
            raise ValueError(f"emotional_valence must be one of positive/negative/neutral/mixed, got {valence!r}")

        arousal = outputs.get("arousal_level")
        if not isinstance(arousal, (int, float)) or not (0.0 <= float(arousal) <= 1.0):
            raise ValueError("arousal_level must be a float in [0, 1]")

        if not isinstance(outputs.get("empathy_flag"), bool):
            raise ValueError("empathy_flag must be a bool")

        if not isinstance(outputs.get("escalation_recommended"), bool):
            raise ValueError("escalation_recommended must be a bool")

        if not isinstance(outputs.get("tone_guidance"), str):
            raise ValueError("tone_guidance must be a str")

        conf = outputs.get("confidence")
        if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be a float in [0, 1]")

    def _heuristic(
        self,
        *,
        content: str,
        feedback_signals: list[dict[str, Any]],
        narrative_tone: str,
        urgency_score: float,
    ) -> dict[str, Any]:
        pos, neg = score_valence(content)
        text_valence = classify_valence(pos, neg)
        valence = _merge_valence_with_hint(text_valence, narrative_tone)

        arousal = compute_arousal(content, feedback_signals)
        empathy_flag = check_empathy_flag(content, valence, arousal)
        escalation_recommended = check_escalation(
            valence=valence,
            arousal=arousal,
            empathy_flag=empathy_flag,
            urgency_score=urgency_score,
            feedback_signals=feedback_signals,
        )
        tone_guidance = derive_tone_guidance(valence, arousal)
        confidence = compute_confidence(
            content_len=len(content),
            pos=pos,
            neg=neg,
            feedback_count=len(feedback_signals),
        )

        return {
            "emotional_valence": valence,
            "arousal_level": arousal,
            "empathy_flag": empathy_flag,
            "escalation_recommended": escalation_recommended,
            "tone_guidance": tone_guidance,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}

        content = str(memory.get("content") or "")
        feedback_signals = list(enrichment.get("feedback_signals") or [])
        narrative_tone = str(enrichment.get("narrative_tone") or "")
        urgency_score = float(enrichment.get("urgency_score") or 0.0)

        context_settings = context.get("settings") or {}
        strategy = context_settings.get("EMOTIONAL_AFFECTIVE_STRATEGY")
        if not strategy:
            strategy = context_settings.get("AGENT_STRATEGY")
        if not strategy:
            strategy = getattr(settings, "EMOTIONAL_AFFECTIVE_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                content=content,
                feedback_signals=feedback_signals,
                narrative_tone=narrative_tone,
                urgency_score=urgency_score,
            )
        else:
            prompt = (
                "You are an emotional intelligence engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Analyse the memory content and feedback signals. Return an emotional "
                "assessment with the following fields:\n"
                "- emotional_valence: 'positive' | 'negative' | 'neutral' | 'mixed'\n"
                "- arousal_level: float 0..1 (intensity of emotional activation)\n"
                "- empathy_flag: bool (true if distress or empathy cues present)\n"
                "- escalation_recommended: bool (true if human review is warranted)\n"
                "- tone_guidance: str (suggested tone for downstream agents)\n"
                "- confidence: float 0..1\n"
                "- rationale: brief explanation\n\n"
                f"CONTENT: {content[:500]}\n"
                f"FEEDBACK_SIGNALS: {feedback_signals[:5]}\n"
                f"NARRATIVE_TONE: {narrative_tone}\n"
                f"URGENCY_SCORE: {urgency_score}\n"
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
                and resp.get("emotional_valence") in {"positive", "negative", "neutral", "mixed"}
                and isinstance(resp.get("arousal_level"), (int, float))
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    content=content,
                    feedback_signals=feedback_signals,
                    narrative_tone=narrative_tone,
                    urgency_score=urgency_score,
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
