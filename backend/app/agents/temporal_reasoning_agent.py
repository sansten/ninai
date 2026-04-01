"""Temporal Reasoning Engine — Phase 17.

Annotates each enriched memory with temporal context: its position in
time, the event sequence it belongs to, the trend it represents, and
whether its timing is anomalous relative to related memories.

Depends on:
  - SemanticNormalizationAgent (Phase 6)  — domain
  - EntityResolutionAgent (Phase 7)       — entities
  - MemoryDecayAgent (Phase 15)           — age_days, freshness_score, is_stale

Inputs (via context["memory"]):
  - content:           raw text of the memory
  - created_at:        ISO timestamp of creation

Inputs (via context["memory"]["enrichment"]):
  - domain:            domain string (Phase 6)
  - entities:          resolved entities dict (Phase 7)
  - age_days:          float (Phase 15)
  - freshness_score:   float (Phase 15)
  - is_stale:          bool (Phase 15)
  - related_memories:  list[dict] — pre-fetched temporally-adjacent memories
                       each: {id, created_at, domain, freshness_score}

Outputs:
  - temporal_position:   "recent" | "historical" | "archival"
  - sequence_role:       "initiating" | "escalating" | "resolving" | "recurring" | "isolated"
  - trend_signal:        "rising" | "falling" | "stable" | "anomalous"
  - sequence_gap_days:   float — median gap in days to related memories (0.0 if none)
  - is_sequence_anomaly: bool — timing breaks expected spacing pattern
  - temporal_cluster:    str — ISO year-week, e.g. "2026-W10"
  - velocity_score:      float 0..1 — density of related activity in last 30 days
  - confidence:          float
  - rationale:           "heuristic" | "llm"
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.agents.llm.ollama_breaker import create_ollama_client
from app.core.config import settings


# Age thresholds for temporal position (days).
_RECENT_THRESHOLD = 7.0
_HISTORICAL_THRESHOLD = 90.0

# Velocity: count related memories within this window.
_VELOCITY_WINDOW_DAYS = 30.0
# Count of related memories in the window at which velocity_score saturates.
_MAX_VELOCITY_COUNT = 20

# Flag timing as anomalous when gap deviates more than this many std devs
# from the median of the cluster's gaps.
_ANOMALY_STD_MULTIPLIER = 2.0

# Minimum related memories needed to infer a meaningful trend.
_MIN_RELATED_FOR_TREND = 3

_VALID_POSITIONS = frozenset({"recent", "historical", "archival"})
_VALID_ROLES = frozenset({"initiating", "escalating", "resolving", "recurring", "isolated"})
_VALID_TRENDS = frozenset({"rising", "falling", "stable", "anomalous"})


# ---------------------------------------------------------------------------
# Pure helper functions (tested independently)
# ---------------------------------------------------------------------------

def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO datetime string or pass through a datetime object."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def classify_temporal_position(age_days: float) -> str:
    """Bucket a memory by age into recent / historical / archival."""
    age = max(float(age_days or 0.0), 0.0)
    if age < _RECENT_THRESHOLD:
        return "recent"
    if age < _HISTORICAL_THRESHOLD:
        return "historical"
    return "archival"


def compute_temporal_cluster(created_at: datetime) -> str:
    """Return an ISO year-week string, e.g. '2026-W10'."""
    iso = created_at.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def compute_sequence_gaps(
    related_memories: list[dict[str, Any]], reference_dt: datetime
) -> list[float]:
    """Compute absolute gap in days between this memory and each related one.

    Returns a sorted list of non-negative gap values.  Entries without a
    parseable created_at are skipped.
    """
    gaps: list[float] = []
    for mem in related_memories or []:
        dt = _parse_dt(mem.get("created_at"))
        if dt is None:
            continue
        gap = abs((reference_dt - dt).total_seconds()) / 86400.0
        gaps.append(round(gap, 3))
    return sorted(gaps)


def compute_velocity_score(
    related_memories: list[dict[str, Any]],
    *,
    now: datetime,
    window_days: float = _VELOCITY_WINDOW_DAYS,
) -> float:
    """Measure density of related activity within the last window_days.

    score = min(count / _MAX_VELOCITY_COUNT, 1.0), so 20+ related memories
    in the window yields a score of 1.0.
    """
    count = 0
    for mem in related_memories or []:
        dt = _parse_dt(mem.get("created_at"))
        if dt is None:
            continue
        age = (now - dt).total_seconds() / 86400.0
        if 0.0 <= age <= window_days:
            count += 1
    return round(min(count / _MAX_VELOCITY_COUNT, 1.0), 4)


def detect_trend_signal(
    related_memories: list[dict[str, Any]],
) -> str:
    """Infer a trend from the freshness trajectory of related memories.

    Sorts related memories by created_at and computes mean first-difference
    of freshness_score values.  A noisy signal (high std vs mean) → anomalous.
    Falls back to 'stable' when there is insufficient data.
    """
    scored: list[tuple[datetime, float]] = []
    for mem in related_memories or []:
        dt = _parse_dt(mem.get("created_at"))
        fs = mem.get("freshness_score")
        if dt is not None and isinstance(fs, (int, float)):
            scored.append((dt, float(fs)))

    if len(scored) < _MIN_RELATED_FOR_TREND:
        return "stable"

    scored.sort(key=lambda x: x[0])
    scores = [s for _, s in scored]

    diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0

    # Noisy relative to signal magnitude → anomalous.
    if std_diff > 0.05 and std_diff > abs(mean_diff) * 2:
        return "anomalous"
    if mean_diff > 0.02:
        return "rising"
    if mean_diff < -0.02:
        return "falling"
    return "stable"


def detect_sequence_anomaly(
    gap_days: float,
    sequence_gaps: list[float],
) -> bool:
    """Return True if gap_days deviates significantly from the cluster's gap distribution.

    Requires at least 2 gaps to compute a meaningful standard deviation.
    """
    if len(sequence_gaps) < 2:
        return False
    median = statistics.median(sequence_gaps)
    std = statistics.pstdev(sequence_gaps)
    if std < 0.01:
        return False
    return abs(gap_days - median) > _ANOMALY_STD_MULTIPLIER * std


def classify_sequence_role(
    *,
    sequence_gaps: list[float],
    trend_signal: str,
    is_stale: bool,
    is_anomaly: bool,
    velocity_score: float,
    temporal_position: str,
) -> str:
    """Assign a narrative role to this memory within its sequence.

    Priority:
    1. isolated   — no related memories or timing is anomalous
    2. initiating — recent memory with little prior activity
    3. escalating — activity is ramping up (rising trend)
    4. resolving  — activity is winding down (falling trend)
    5. recurring  — stable pattern with meaningful velocity
    6. isolated   — fallback
    """
    if not sequence_gaps or is_anomaly:
        return "isolated"
    if temporal_position == "recent" and velocity_score < 0.15:
        return "initiating"
    if trend_signal == "rising" and not is_stale:
        return "escalating"
    if trend_signal == "falling":
        return "resolving"
    if trend_signal == "stable" and velocity_score >= 0.30:
        return "recurring"
    return "isolated"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TemporalReasoningAgent(BaseAgent):
    """Phase 17: Annotate memories with temporal context and sequence role.

    Consumes decay metadata (age_days, freshness_score, is_stale) together
    with a pre-fetched list of related memories to compute event-sequence
    position, trend direction, and anomaly flags.
    """

    name = "TemporalReasoningAgent"
    version = "v1"

    def dependencies(self) -> list[str]:
        return ["SemanticNormalizationAgent", "EntityResolutionAgent", "MemoryDecayAgent"]

    def validate_outputs(self, result: AgentResult) -> None:
        if result.status != "success":
            return
        outputs = result.outputs or {}

        pos = outputs.get("temporal_position")
        if pos not in _VALID_POSITIONS:
            raise ValueError(f"invalid temporal_position: {pos!r}")

        role = outputs.get("sequence_role")
        if role not in _VALID_ROLES:
            raise ValueError(f"invalid sequence_role: {role!r}")

        trend = outputs.get("trend_signal")
        if trend not in _VALID_TRENDS:
            raise ValueError(f"invalid trend_signal: {trend!r}")

        gap = outputs.get("sequence_gap_days")
        if not isinstance(gap, (int, float)) or float(gap) < 0:
            raise ValueError("sequence_gap_days must be a non-negative float")

        if not isinstance(outputs.get("is_sequence_anomaly"), bool):
            raise ValueError("is_sequence_anomaly must be a bool")

        if not isinstance(outputs.get("temporal_cluster"), str):
            raise ValueError("temporal_cluster must be a str")

        vs = outputs.get("velocity_score")
        if not isinstance(vs, (int, float)) or not (0.0 <= float(vs) <= 1.0):
            raise ValueError("velocity_score must be a float in [0, 1]")

    def _heuristic(
        self,
        *,
        created_at: datetime,
        age_days: float,
        is_stale: bool,
        related_memories: list[dict[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        temporal_position = classify_temporal_position(age_days)
        temporal_cluster = compute_temporal_cluster(created_at)
        sequence_gaps = compute_sequence_gaps(related_memories, created_at)
        velocity_score = compute_velocity_score(related_memories, now=now)
        trend_signal = detect_trend_signal(related_memories)

        sequence_gap_days = (
            round(statistics.median(sequence_gaps), 3) if sequence_gaps else 0.0
        )
        is_anomaly = detect_sequence_anomaly(sequence_gap_days, sequence_gaps)
        sequence_role = classify_sequence_role(
            sequence_gaps=sequence_gaps,
            trend_signal=trend_signal,
            is_stale=is_stale,
            is_anomaly=is_anomaly,
            velocity_score=velocity_score,
            temporal_position=temporal_position,
        )

        # Confidence: higher when we have more signal
        if is_anomaly:
            confidence = 0.55
        elif not sequence_gaps:
            confidence = 0.60
        elif trend_signal == "anomalous":
            confidence = 0.65
        elif temporal_position == "recent":
            confidence = 0.85
        elif temporal_position == "historical":
            confidence = 0.75
        else:
            confidence = 0.65

        return {
            "temporal_position": temporal_position,
            "sequence_role": sequence_role,
            "trend_signal": trend_signal,
            "sequence_gap_days": sequence_gap_days,
            "is_sequence_anomaly": is_anomaly,
            "temporal_cluster": temporal_cluster,
            "velocity_score": velocity_score,
            "confidence": confidence,
            "rationale": "heuristic",
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started_at = datetime.now(timezone.utc)
        trace_id = (context.get("runtime") or {}).get("job_id")
        memory = context.get("memory") or {}
        enrichment = memory.get("enrichment") or {}
        content = str(memory.get("content") or "")

        wall_now = datetime.now(timezone.utc)
        created_at = _parse_dt(memory.get("created_at")) or wall_now
        age_days = float(enrichment.get("age_days") or 0.0)
        # Derive an event-relative "now" for deterministic velocity scoring in tests
        # and for replayed historical contexts where age_days is provided.
        now = created_at + timedelta(days=max(age_days, 0.0))
        is_stale = bool(enrichment.get("is_stale", False))
        related_memories = list(enrichment.get("related_memories") or [])

        strategy = getattr(settings, "TEMPORAL_REASONING_STRATEGY", None)
        if not strategy:
            strategy = getattr(settings, "AGENT_STRATEGY", "llm")
        strategy = str(strategy or "llm").strip().lower()

        outputs: dict[str, Any]

        if strategy == "heuristic" or not content:
            outputs = self._heuristic(
                created_at=created_at,
                age_days=age_days,
                is_stale=is_stale,
                related_memories=related_memories,
                now=now,
            )
        else:
            rel_json = related_memories[:5]
            prompt = (
                "You are a temporal reasoning engine for an enterprise memory OS. "
                "Output JSON only. Do not hallucinate.\n\n"
                "Classify the temporal context of this memory.\n\n"
                f"CREATED_AT: {created_at.isoformat()}\n"
                f"AGE_DAYS: {age_days}\n"
                f"IS_STALE: {is_stale}\n"
                f"RELATED_MEMORIES: {rel_json}\n"
                f"CONTENT: {content[:300]}\n\n"
                "Return JSON with keys:\n"
                "- temporal_position: one of recent|historical|archival\n"
                "- sequence_role: one of initiating|escalating|resolving|recurring|isolated\n"
                "- trend_signal: one of rising|falling|stable|anomalous\n"
                "- sequence_gap_days: non-negative float\n"
                "- is_sequence_anomaly: bool\n"
                "- temporal_cluster: ISO year-week string e.g. '2026-W10'\n"
                "- velocity_score: float 0..1\n"
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
                and resp.get("temporal_position") in _VALID_POSITIONS
                and resp.get("sequence_role") in _VALID_ROLES
                and resp.get("trend_signal") in _VALID_TRENDS
                and isinstance(resp.get("is_sequence_anomaly"), bool)
                and isinstance(resp.get("temporal_cluster"), str)
            ):
                outputs = resp
            else:
                outputs = self._heuristic(
                    created_at=created_at,
                    age_days=age_days,
                    is_stale=is_stale,
                    related_memories=related_memories,
                    now=now,
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
