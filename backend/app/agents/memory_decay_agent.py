"""Memory Decay & Relevance Aging — Phase 15.

Annotates each enriched memory with a freshness score that decays
exponentially over time, using a domain-specific half-life. Freshness
is boosted by reference count and recency of last access.

The score feeds downstream agents (conflict detection, world model, silo
propagation) so that stale signals are automatically deprioritised.

Inputs (via context["memory"]):
  - created_at:       ISO timestamp string when the memory was created
  - content:          raw memory content (for LLM path)

Inputs (via context["memory"]["enrichment"]):
  - domain:           domain string from Phase 6 (SemanticNormalizationAgent)
  - reference_count:  int — how many times this memory has been referenced
  - last_accessed_at: ISO timestamp of the most recent access (optional)

Outputs:
  - freshness_score:  float 0..1
  - decay_rate:       float (λ, per-day exponential rate)
  - age_days:         float
  - half_life:        float (days)
  - reference_boost:  float
  - recency_boost:    float
  - expires_at:       ISO timestamp string (3× half-life from created_at)
  - is_stale:         bool (freshness_score < stale_threshold)
  - domain:           str
  - confidence:       float
  - rationale:        "heuristic" | "llm"
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# Domain-specific half-lives in days.  Shorter = decays faster.
_DOMAIN_HALF_LIFE: dict[str, float] = {
    "incident": 7.0,
    "support": 14.0,
    "sprint": 14.0,
    "deployment": 21.0,
    "sales": 30.0,
    "engineering": 45.0,
    "project": 60.0,
    "finance": 90.0,
    "hr": 90.0,
    "strategy": 180.0,
    "document": 365.0,
}
_DEFAULT_HALF_LIFE = 30.0
_DEFAULT_STALE_THRESHOLD = 0.20

# Reference boost: each reference adds this fraction, capped at the max.
_REFERENCE_BOOST_PER_REF = 0.02
_MAX_REFERENCE_BOOST = 0.20

# Recency boost: linear from MAX_RECENCY_BOOST (accessed today) to 0
# at RECENCY_WINDOW_DAYS days ago.
_MAX_RECENCY_BOOST = 0.10
_RECENCY_WINDOW_DAYS = 7.0


def domain_half_life(domain: str) -> float:
    """Return the half-life in days for the given domain (case-insensitive)."""
    return _DOMAIN_HALF_LIFE.get(str(domain or "").lower(), _DEFAULT_HALF_LIFE)


def compute_decay_rate(*, half_life: float) -> float:
    """Compute λ = ln(2) / t_half so that freshness(t_half) ≈ 0.5."""
    hl = max(float(half_life), 0.01)
    return round(math.log(2) / hl, 6)


def compute_base_freshness(*, age_days: float, decay_rate: float) -> float:
    """Exponential decay: e^(-λ × age_days), clamped to [0, 1]."""
    score = math.exp(-float(decay_rate) * max(float(age_days), 0.0))
    return round(min(max(score, 0.0), 1.0), 4)


def compute_reference_boost(*, reference_count: int) -> float:
    """Each reference adds _REFERENCE_BOOST_PER_REF, capped at _MAX_REFERENCE_BOOST."""
    boost = max(int(reference_count or 0), 0) * _REFERENCE_BOOST_PER_REF
    return round(min(boost, _MAX_REFERENCE_BOOST), 4)


def compute_recency_boost(
    *, last_accessed_at: datetime | None, now: datetime
) -> float:
    """Linear decay from _MAX_RECENCY_BOOST (just accessed) to 0 at _RECENCY_WINDOW_DAYS."""
    if last_accessed_at is None:
        return 0.0
    days_since = (now - last_accessed_at).total_seconds() / 86400.0
    if days_since < 0 or days_since >= _RECENCY_WINDOW_DAYS:
        return 0.0
    boost = _MAX_RECENCY_BOOST * (1.0 - days_since / _RECENCY_WINDOW_DAYS)
    return round(min(boost, _MAX_RECENCY_BOOST), 4)


def compute_freshness_score(
    *, base: float, reference_boost: float, recency_boost: float
) -> float:
    """Sum base + boosts, capped at 1.0."""
    return round(min(float(base) + float(reference_boost) + float(recency_boost), 1.0), 4)


def compute_expires_at(*, created_at: datetime, half_life: float) -> datetime:
    """Memory is effectively expired at 3× the half-life from creation."""
    return created_at + timedelta(days=float(half_life) * 3)


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO datetime string or passthrough a datetime object."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


class MemoryDecayAgent(BaseAgent):
    """Phase 15: Annotate memories with freshness score and staleness flag.

    Inputs (via context["memory"]):
      - created_at:       ISO timestamp of memory creation
      - content:          raw text (for LLM path)

    Inputs (via context["memory"]["enrichment"]):
      - domain:           domain string (Phase 6)
      - reference_count:  int
      - last_accessed_at: ISO timestamp of last access (optional)

    Outputs:
      - freshness_score, decay_rate, age_days, half_life,
        reference_boost, recency_boost, expires_at, is_stale,
        domain, confidence, rationale
    """

    name = "MemoryDecayAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SemanticNormalizationAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}
        fs = outputs.get("freshness_score")
        if not isinstance(fs, (int, float)):
            raise ValueError("freshness_score must be a float")
        if not (0.0 <= float(fs) <= 1.0):
            raise ValueError("freshness_score must be between 0 and 1")
        if not isinstance(outputs.get("is_stale"), bool):
            raise ValueError("is_stale must be a bool")
        age = outputs.get("age_days")
        if not isinstance(age, (int, float)) or float(age) < 0:
            raise ValueError("age_days must be a non-negative float")
        if not isinstance(outputs.get("expires_at"), str):
            raise ValueError("expires_at must be a string")

    def _get_stale_threshold(self) -> float:
        raw = getattr(settings, "MEMORY_DECAY_STALE_THRESHOLD", None)
        try:
            return float(raw) if raw is not None else _DEFAULT_STALE_THRESHOLD
        except (TypeError, ValueError):
            return _DEFAULT_STALE_THRESHOLD

    def _heuristic(
        self,
        *,
        created_at: datetime,
        last_accessed_at: datetime | None,
        reference_count: int,
        domain: str,
        now: datetime,
    ) -> dict[str, Any]:
        half_life = domain_half_life(domain)
        decay_rate = compute_decay_rate(half_life=half_life)
        age_days = max((now - created_at).total_seconds() / 86400.0, 0.0)

        base = compute_base_freshness(age_days=age_days, decay_rate=decay_rate)
        ref_boost = compute_reference_boost(reference_count=reference_count)
        rec_boost = compute_recency_boost(last_accessed_at=last_accessed_at, now=now)
        freshness_score = compute_freshness_score(
            base=base, reference_boost=ref_boost, recency_boost=rec_boost
        )

        stale_threshold = self._get_stale_threshold()
        is_stale = freshness_score < stale_threshold
        expires_at = compute_expires_at(created_at=created_at, half_life=half_life)

        # Confidence scales with freshness: fresh memories are more reliable.
        confidence = round(min(0.90, 0.50 + freshness_score * 0.40), 3)

        return {
            "freshness_score": freshness_score,
            "decay_rate": decay_rate,
            "age_days": round(age_days, 2),
            "half_life": half_life,
            "reference_boost": ref_boost,
            "recency_boost": rec_boost,
            "expires_at": expires_at.isoformat(),
            "is_stale": is_stale,
            "domain": domain,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}
        content = memory.get("content", "")

        now = datetime.now(timezone.utc)
        created_at = _parse_dt(memory.get("created_at")) or now
        last_accessed_at = _parse_dt(enrichment.get("last_accessed_at"))
        reference_count = int(enrichment.get("reference_count") or 0)
        domain = str(enrichment.get("domain") or "")

        strategy = getattr(settings, "MEMORY_DECAY_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                created_at=created_at,
                last_accessed_at=last_accessed_at,
                reference_count=reference_count,
                domain=domain,
                now=now,
            )
        else:
            prompt = (
                "You are a memory relevance scoring engine for an enterprise Cognitive OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Score the freshness of this memory based on its age, domain, and access patterns.\n\n"
                f"CREATED_AT: {created_at.isoformat()}\n"
                f"DOMAIN: {domain}\n"
                f"REFERENCE_COUNT: {reference_count}\n"
                f"LAST_ACCESSED_AT: {last_accessed_at.isoformat() if last_accessed_at else 'unknown'}\n"
                f"NOW: {now.isoformat()}\n"
                f"CONTENT: {content[:300]}\n\n"
                "Return JSON with keys:\n"
                "- freshness_score: float 0..1\n"
                "- decay_rate: float (per-day decay rate)\n"
                "- age_days: float\n"
                "- half_life: float (days)\n"
                "- reference_boost: float\n"
                "- recency_boost: float\n"
                "- expires_at: ISO timestamp string\n"
                "- is_stale: bool\n"
                "- domain: str\n"
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
                and isinstance(resp.get("freshness_score"), (int, float))
                and isinstance(resp.get("is_stale"), bool)
                and isinstance(resp.get("expires_at"), str)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    created_at=created_at,
                    last_accessed_at=last_accessed_at,
                    reference_count=reference_count,
                    domain=domain,
                    now=now,
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
